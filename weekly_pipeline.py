import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from db.operations import fetch_news, save_news
from get_news.fetch_fulltext import read as fetch_fulltext
from utils.chat_completion import THINKING_MODEL, achat_complete
from db.operations import save_performance_record
from utils.llm import choose_scenario
from utils.backtest import compute_backtest_metrics
from utils.utils import trim_to_chars
from utils.data import get_data, prepare_datasets

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def setup_logger(run_id: Optional[str] = None, name: str = "weekly_pipeline", log_dir: str = "logs") -> logging.Logger:
    logger = logging.getLogger(run_id or name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if run_id:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{run_id}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def get_week_window(anchor_date: Optional[datetime] = None) -> Tuple[str, str]:
    dt = anchor_date or datetime.now()
    # Accept str/date/datetime.
    if isinstance(dt, str):
        dt = pd.to_datetime(dt, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"Invalid anchor_date string: {anchor_date}")
        dt = dt.to_pydatetime()
    elif not isinstance(dt, datetime):
        # date-like
        dt = pd.to_datetime(dt, errors="coerce").to_pydatetime()

    dt = datetime(dt.year, dt.month, dt.day)
    monday = dt - timedelta(days=dt.weekday())
    sunday = monday + timedelta(days=6)
    return monday.strftime("%Y-%m-%d"), sunday.strftime("%Y-%m-%d")


def _safe_json_extract(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    s = raw.find("{")
    e = raw.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(raw[s : e + 1])
        except Exception:
            pass
    return {}


def _parse_selected_ids(raw: str, cap: int) -> List[int]:
    js = _safe_json_extract(raw)

    # Preferred format: {"selected_ids":[1,2,...]}
    ids = js.get("selected_ids")
    if isinstance(ids, list):
        out = []
        for x in ids:
            try:
                out.append(int(x))
            except Exception:
                continue
        return _dedup_keep(out)[:cap]

    # Compatible format: {"selected":[{"id":1}, ...]}
    selected = js.get("selected")
    if isinstance(selected, list):
        out = []
        for it in selected:
            if isinstance(it, dict):
                try:
                    out.append(int(it.get("id")))
                except Exception:
                    continue
        return _dedup_keep(out)[:cap]

    nums = [int(x) for x in re.findall(r"\b\d+\b", raw or "")]
    return _dedup_keep(nums)[:cap]


def _dedup_keep(xs: List[int]) -> List[int]:
    out = []
    seen = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _norm_probs(prob: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    clean = {}
    for k in keys:
        try:
            clean[k] = float(prob.get(k, 0.0))
        except Exception:
            clean[k] = 0.0
    s = sum(max(v, 0.0) for v in clean.values())
    if s <= 0:
        p = 1.0 / len(keys)
        return {k: p for k in keys}
    return {k: max(clean[k], 0.0) / s for k in keys}


def _parse_weekly_regime_output(raw: str) -> Dict[str, Any]:
    js = _safe_json_extract(raw)

    direction = js.get("direction_probs") if isinstance(js, dict) else {}
    vol = js.get("volatility_probs") if isinstance(js, dict) else {}

    direction = _norm_probs(direction if isinstance(direction, dict) else {}, ["bull", "bear", "range"])
    vol = _norm_probs(vol if isinstance(vol, dict) else {}, ["low_vol", "medium_vol", "high_vol"])

    out = {
        "direction_probs": direction,
        "volatility_probs": vol,
        "strategy_background": str(js.get("strategy_background", "")).strip() if isinstance(js, dict) else "",
        "key_drivers": js.get("key_drivers", []) if isinstance(js, dict) else [],
        "main_risks": js.get("main_risks", []) if isinstance(js, dict) else [],
    }
    return out


def build_candidate_prompt_items(news_rows: List[Dict[str, Any]]) -> str:
    lines = ["Candidate news in this week:", ""]
    for idx, it in enumerate(news_rows, start=1):
        title = trim_to_chars((it.get("title") or "").strip(), 280)
        src = (it.get("source") or "").strip()
        dt = it.get("published_at")
        dt_text = str(dt)[:19] if dt is not None else "(none)"
        kw = (it.get("keyword") or "").strip()
        lines.append(
            f"[{idx}] Date: {dt_text}\n"
            f"Source: {src or '(none)'}\n"
            f"Keyword: {kw or '(none)'}\n"
            f"Title: {title or '(none)'}\n"
        )
    return "\n".join(lines)


async def llm_select_articles(
    news_rows: List[Dict[str, Any]],
    *,
    max_pick: int,
    model_name: str,
) -> List[Dict[str, Any]]:
    # Hard cap to keep prompt size bounded.
    if not news_rows:
        return []
    news_rows = news_rows[:200]

    sysmsg = (
        "You are a disciplined financial news selector for Hang Seng Index (HSI) options trading.\n"
        "Select only the headlines with clear predictive value for next-week market direction, volatility, "
        "liquidity, policy transmission, or risk sentiment.\n"
        "Avoid generic recap and low-impact stories.\n"
        "Return JSON only with schema: {\"selected_ids\": [1,2,...]}.\n"
        f"Do not select more than {max_pick} items."
    )
    user = build_candidate_prompt_items(news_rows) + f"\nSelect <= {max_pick} items. Return JSON only."
    try:
        raw = await achat_complete(model_name, sysmsg, user, temperature=0.15)
        picked_ids = _parse_selected_ids(raw, cap=max_pick)
    except Exception as e:
        # Fallback: shrink candidate set to reduce request risk/size.
        # Note: we intentionally do not crash the whole weekly timeline.
        fallback_n = min(len(news_rows), 80)
        if fallback_n <= 0:
            return []
        try:
            user2 = build_candidate_prompt_items(news_rows[:fallback_n]) + f"\nSelect <= {max_pick} items. Return JSON only."
            raw2 = await achat_complete(model_name, sysmsg, user2, temperature=0.15)
            picked_ids = _parse_selected_ids(raw2, cap=max_pick)
        except Exception:
            return []
    
    picked: List[Dict[str, Any]] = []
    seen = set()
    for i in picked_ids:
        if i in seen:
            continue
        seen.add(i)
        if 1 <= i <= len(news_rows):
            picked.append(news_rows[i - 1])
    return picked


async def enrich_selected_with_fulltext(
    selected: List[Dict[str, Any]],
    *,
    max_concurrency: int = 3,
    summary_chars: int = 500,
    content_chars: int = 2200,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        url = (item.get("url") or "").strip()
        text = ""
        async with sem:
            if url:
                try:
                    text = await asyncio.to_thread(fetch_fulltext, url)
                except Exception:
                    text = ""

        text = (text or "").strip()
        if not text:
            text = (
                (item.get("summary") or "").strip()
                or (item.get("content") or "").strip()
                or (item.get("title") or "").strip()
            )

        item = dict(item)
        item["summary"] = text[:summary_chars]
        item["content"] = text[:content_chars]
        return item

    tasks = [asyncio.create_task(_one(x)) for x in selected]
    return await asyncio.gather(*tasks)


def build_news_context_for_weekly_llm(selected: List[Dict[str, Any]], max_items: int = 25) -> str:
    rows = selected[:max_items]
    lines = []
    lines.append("## Weekly News Context")
    lines.append("The following news was pre-selected by an LLM and enriched with article excerpts.")
    lines.append("")

    for i, it in enumerate(rows, start=1):
        lines.append(f"[{i}] Date: {str(it.get('published_at'))[:19] or '(none)'}")
        lines.append(f"Source: {(it.get('source') or '').strip() or '(none)'}")
        lines.append(f"Title: {(it.get('title') or '').strip() or '(none)'}")
        lines.append(f"Excerpt: {trim_to_chars((it.get('summary') or '').strip(), 650) or '(none)'}")
        lines.append("")
    return "\n".join(lines).strip()


async def llm_weekly_market_analysis(
    news_context: str,
    *,
    week_start: str,
    week_end: str,
    model_name: str,
) -> Dict[str, Any]:
    sysmsg = (
        "You are a quantitative trader specializing in HSI regime analysis.\n"
        "Use the weekly news context as supplementary evidence and estimate next-week market regime.\n"
        "Return JSON only with keys:\n"
        "{\n"
        '  "direction_probs": {"bull":..., "bear":..., "range":...},\n'
        '  "volatility_probs": {"low_vol":..., "medium_vol":..., "high_vol":...},\n'
        '  "key_drivers": ["..."],\n'
        '  "main_risks": ["..."],\n'
        '  "strategy_background": "..."\n'
        "}\n"
        "Rules:\n"
        "- Probabilities must each sum to 1.\n"
        "- strategy_background should be concise Chinese for strategy context."
    )
    user = (
        f"Week window (Mon-Sun): {week_start} to {week_end}\n"
        "Forecast horizon: next week\n\n"
        f"{news_context}\n\n"
        "Return JSON only."
    )
    raw = await achat_complete(model_name, sysmsg, user, temperature=0.14)
    out = _parse_weekly_regime_output(raw)

    # Reuse existing scenario mapping logic from current pipeline style.
    scenario_text = json.dumps(
        {
            "direction_probs": out["direction_probs"],
            "volatility_probs": out["volatility_probs"],
        },
        ensure_ascii=False,
    )
    out["scenario"] = choose_scenario(scenario_text)
    out["raw_llm_text"] = raw
    return out


async def run_weekly_pipeline(
    *,
    anchor_date: Optional[datetime] = None,
    candidate_limit: int = 250,
    max_pick_for_fulltext: int = 40,
    keyword: Optional[str] = None,
    model_name: str = THINKING_MODEL,
    max_concurrency: int = 3,
    save_selected_fulltext_to_db: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Weekly pipeline (Mon-Sun):
    1) Fetch weekly candidate headlines from DB
    2) LLM-1 dynamically selects important articles (<= threshold)
    3) Fetch selected article full text and optionally write back to DB
    4) LLM-2 outputs weekly regime + strategy background
    """
    if logger is None:
        logger = setup_logger()
    t0 = time.perf_counter()
    week_start, week_end = get_week_window(anchor_date)

    logger.info(
        "WEEKLY PIPELINE START | week_start=%s | week_end=%s | candidate_limit=%d | max_pick_for_fulltext=%d | model=%s",
        week_start,
        week_end,
        candidate_limit,
        max_pick_for_fulltext,
        model_name,
    )

    candidates = fetch_news(
        start_date=week_start,
        end_date=week_end,
        limit=candidate_limit,
        keyword=keyword,
    )
    logger.info("candidates fetched=%d", len(candidates))

    if not candidates:
        return {
            "week_start": week_start,
            "week_end": week_end,
            "selected_count": 0,
            "status": "no_news_in_db",
        }
    selected = await llm_select_articles(
        candidates,
        max_pick=max_pick_for_fulltext,
        model_name=model_name,
    )
    logger.info("llm selected count=%d", len(selected))

    if not selected:
        return {
            "week_start": week_start,
            "week_end": week_end,
            "candidate_count": len(candidates),
            "selected_count": 0,
            "status": "llm_selected_none",
        }

    enriched = await enrich_selected_with_fulltext(
        selected,
        max_concurrency=max_concurrency,
    )

    if save_selected_fulltext_to_db:
        # save_news uses upsert by url_hash and will update summary/content.
        save_news(enriched)
        logger.info("selected fulltext saved to db | rows=%d", len(enriched))

    context = build_news_context_for_weekly_llm(enriched, max_items=min(len(enriched), max_pick_for_fulltext))
    try:
        analysis = await llm_weekly_market_analysis(
            context,
            week_start=week_start,
            week_end=week_end,
            model_name=model_name,
        )
    except Exception:
        logger.exception("LLM weekly market analysis failed | week_start=%s | week_end=%s", week_start, week_end)
        return {
            "week_start": week_start,
            "week_end": week_end,
            "candidate_count": len(candidates),
            "selected_count": len(enriched),
            "selected_urls": [x.get("url") for x in enriched if x.get("url")],
            "analysis": {},
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "status": "llm_analysis_failed",
        }

    result = {
        "week_start": week_start,
        "week_end": week_end,
        "candidate_count": len(candidates),
        "selected_count": len(enriched),
        "selected_urls": [x.get("url") for x in enriched if x.get("url")],
        "analysis": analysis,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "status": "ok",
    }
    logger.info(
        "WEEKLY PIPELINE END | status=%s | selected=%d | scenario=%s | elapsed=%.3fs",
        result["status"],
        result["selected_count"],
        analysis.get("scenario"),
        result["elapsed_sec"],
    )
    return result


def _summarize_full_mtm(full_mtm_df: pd.DataFrame, account_value: float = 1_000_000.0) -> Dict[str, Any]:
    if full_mtm_df is None or full_mtm_df.empty:
        return {
            "n_days": 0,
            "cum_pnl": np.nan,
            "return_pct": np.nan,
            "ann_return_pct": np.nan,
            "ann_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
            "win_rate_daily": np.nan,
        }

    df = full_mtm_df.copy()
    df["portfolio_daily_ret"] = df["portfolio_daily_pnl"] / account_value

    n_days = len(df)
    cum_pnl = float(df["portfolio_total_pnl"].iloc[-1])
    ret_pct = cum_pnl / account_value

    ann_return = (1 + ret_pct) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else np.nan
    ann_vol = df["portfolio_daily_ret"].std(ddof=1) * np.sqrt(252) if n_days > 1 else np.nan
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan

    equity = account_value + df["portfolio_total_pnl"]
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1
    mdd = float(drawdown.min()) if len(drawdown) else np.nan

    win_rate_daily = float((df["portfolio_daily_pnl"] > 0).mean()) if n_days > 0 else np.nan
    return {
        "n_days": int(n_days),
        "cum_pnl": cum_pnl,
        "return_pct": ret_pct,
        "ann_return_pct": ann_return,
        "ann_vol_pct": ann_vol,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd,
        "win_rate_daily": win_rate_daily,
    }


def get_weekly_windows_from_option_df(
    option_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Build weekly windows using trading days in option_df.

    For each decision week (Mon-Sun), we backtest the NEXT week:
      entry_date = first trading day in next week
      exit_date  = last  trading day in next week

    We only keep windows whose entry/exit are fully within [start_date, end_date].
    Returns: [(decision_date(Sun), entry_date, exit_date), ...]
    """
    if option_df is None or option_df.empty:
        return []

    df = option_df.copy()
    if "Date" not in df.columns:
        raise KeyError("option_df must contain 'Date' column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])

    trading_days = sorted(df["Date"].unique())
    if not trading_days:
        return []

    start_ts = pd.to_datetime(start_date, errors="coerce").normalize()
    end_ts = pd.to_datetime(end_date, errors="coerce").normalize()
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError(f"Invalid start/end date: {start_date}, {end_date}")

    # decision_monday is the Monday of the week containing start_ts
    decision_monday = start_ts - pd.Timedelta(days=int(start_ts.weekday()))
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    decision_monday = decision_monday.normalize()
    while decision_monday <= end_ts:
        decision_week_end = decision_monday + pd.Timedelta(days=6)  # Sunday
        next_week_start = decision_week_end + pd.Timedelta(days=1)
        next_week_end = next_week_start + pd.Timedelta(days=6)

        # trading days in next week intersected with [start_ts, end_ts]
        entries = [d for d in trading_days if (next_week_start <= d <= next_week_end) and (start_ts <= d <= end_ts)]
        if not entries:
            decision_monday += pd.Timedelta(days=7)
            continue

        entry_date = entries[0]
        exit_date = entries[-1]

        if entry_date < start_ts or exit_date > end_ts:
            decision_monday += pd.Timedelta(days=7)
            continue

        # Keep backtests whose full horizon lies in the evaluation interval.
        if entry_date <= exit_date:
            windows.append((decision_week_end, entry_date, exit_date))

        decision_monday += pd.Timedelta(days=7)

    return windows


async def run_full_weekly_timeline(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    model_name: str = "deepseek-reasoner",
    account_value: float = 1_000_000.0,
    risk_pct: float = 0.01,
    save_db: bool = True,
    # News pipeline params
    candidate_limit: int = 250,
    max_pick_for_fulltext: int = 35,
    keyword: Optional[str] = None,
    save_selected_fulltext_to_db: bool = True,
    max_concurrency: int = 3,
) -> Dict[str, Any]:
    """
    Weekly full-year backtest:
    week decision (Mon-Sun news) -> LLM => scenario -> next-week option MTM backtest.
    """
    run_id = f"weekly_timeline_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    logger = setup_logger(run_id=run_id)

    t0 = time.perf_counter()
    logger.info(
        "RUN START | run_id=%s | start_date=%s | end_date=%s | model_name=%s | account_value=%.2f | risk_pct=%.4f | save_db=%s",
        run_id,
        start_date,
        end_date,
        model_name,
        account_value,
        risk_pct,
        save_db,
    )

    try:
        # 1) Load raw data & prepare option_df
        t_data = time.perf_counter()
        price_chain_df, hsi_df, vhsi_df, future = get_data()
        logger.debug("RAW DATA LOADED | elapsed=%.3fs", time.perf_counter() - t_data)

        t_prepare = time.perf_counter()
        price_chain_df_clean, hsi_df_clean, vhsi_df_clean, index_df = prepare_datasets(
            price_chain_df, hsi_df, vhsi_df, future
        )
        logger.debug("PREPARE DATASETS DONE | elapsed=%.3fs", time.perf_counter() - t_prepare)

        # option_df used by the MTM engine (signals generation needs option columns)
        merged_df = pd.merge(price_chain_df_clean, index_df, on="Date", how="left")
        merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce").dt.normalize()
        merged_df = merged_df.dropna(subset=["Date"])

        # 2) Build weekly windows (decision week -> next week backtest window)
        windows = get_weekly_windows_from_option_df(merged_df, start_date=start_date, end_date=end_date)
        logger.info("Weekly windows built | n_windows=%d", len(windows))
        if not windows:
            raise ValueError("No weekly windows found. Check your option_df data range.")

        weekly_rows: List[Dict[str, Any]] = []
        mtm_parts: List[pd.DataFrame] = []
        cumulative_offset = 0.0

        # 3) Weekly loop
        for i, (decision_date, entry_date, exit_date) in enumerate(
            tqdm(windows, desc="Running weekly timeline", total=len(windows)),
            start=1,
        ):
            logger.info(
                "[%02d/%02d] WEEK START | decision_date=%s | entry_date=%s | exit_date=%s | cumulative_offset_before=%.2f",
                i,
                len(windows),
                decision_date.date(),
                entry_date.date(),
                exit_date.date(),
                cumulative_offset,
            )

            # 3.1) News => scenario
            news_result = await run_weekly_pipeline(
                anchor_date=decision_date.to_pydatetime(),
                candidate_limit=candidate_limit,
                max_pick_for_fulltext=max_pick_for_fulltext,
                keyword=keyword,
                model_name=model_name,
                max_concurrency=max_concurrency,
                save_selected_fulltext_to_db=save_selected_fulltext_to_db,
                logger=logger,
            )

            if news_result.get("status") != "ok":
                logger.warning(
                    "[%s] news_result not ok | status=%s | week_start=%s | week_end=%s",
                    decision_date.date(),
                    news_result.get("status"),
                    news_result.get("week_start"),
                    news_result.get("week_end"),
                )
                continue

            analysis = news_result.get("analysis") or {}
            scenario = analysis.get("scenario")
            direction_probs = analysis.get("direction_probs") or {}
            volatility_probs = analysis.get("volatility_probs") or {}
            strategy_background = analysis.get("strategy_background") or ""
            key_drivers = analysis.get("key_drivers") or []
            main_risks = analysis.get("main_risks") or []

            if not scenario:
                logger.warning("[%s] scenario missing, skip backtest.", decision_date.date())
                continue

            # 3.2) Backtest next week
            bt = compute_backtest_metrics(
                scenario=scenario,
                entry_date=entry_date,
                exit_date=exit_date,
                option_df=merged_df,
                account_value=account_value,
                risk_pct=risk_pct,
            )

            metrics = bt["metrics"]
            mtm_df = bt["mtm_df"]
            signals = bt["signals"]
            strategy_type = bt["strategy_type"]

            # 3.3) MTM post-process + accumulate pnl
            if mtm_df is not None and not mtm_df.empty:
                mtm_df = mtm_df.copy()
                mtm_df["decision_date"] = decision_date
                mtm_df["entry_date"] = entry_date
                mtm_df["exit_date"] = exit_date
                mtm_df["scenario"] = scenario
                mtm_df["strategy_type"] = strategy_type

                prev_offset = cumulative_offset
                mtm_df["portfolio_total_pnl"] = mtm_df["Total_PnL"] + cumulative_offset
                cumulative_offset = float(mtm_df["portfolio_total_pnl"].iloc[-1])
                mtm_parts.append(mtm_df)
                logger.info(
                    "[%s] WEEK BACKTEST DONE | scenario=%s | cum_pnl=%s | return_pct=%s | sharpe=%s | max_dd=%s | elapsed_cum=%.2fs",
                    decision_date.date(),
                    scenario,
                    metrics.get("cum_pnl"),
                    metrics.get("return_pct"),
                    metrics.get("sharpe"),
                    metrics.get("max_drawdown_pct"),
                    time.perf_counter() - t0,
                )
            else:
                logger.warning("[%s] Empty mtm_df returned from backtest.", decision_date.date())

            # 3.4) DB save
            if save_db:
                reasoning_text = "\n".join(
                    [
                        f"Decision date: {decision_date.date()}",
                        f"Chosen scenario: {scenario}",
                        f"Strategy type: {strategy_type}",
                        f"Key drivers: {key_drivers}",
                        f"Main risks: {main_risks}",
                        f"Strategy background:\n{strategy_background}",
                    ]
                )
                save_performance_record(
                    run_id=run_id,
                    decision_date=decision_date,
                    final_text=analysis,
                    performance_metrics=metrics,
                    reasoning=reasoning_text,
                    model=model_name,
                )

            weekly_rows.append(
                {
                    "decision_date": decision_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "direction_probs": direction_probs,
                    "volatility_probs": volatility_probs,
                    "scenario": scenario,
                    "strategy_type": strategy_type,
                    "signals": signals,
                    "cum_pnl": metrics.get("cum_pnl"),
                    "return_pct": metrics.get("return_pct"),
                    "ann_return_pct": metrics.get("ann_return_pct"),
                    "ann_vol_pct": metrics.get("ann_vol_pct"),
                    "sharpe": metrics.get("sharpe"),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "win_rate_daily": metrics.get("win_rate_daily"),
                    "news_selected_count": news_result.get("selected_count"),
                    "news_week_start": news_result.get("week_start"),
                    "news_week_end": news_result.get("week_end"),
                }
            )

        # 4) Assemble
        weekly_df = pd.DataFrame(weekly_rows)
        logger.info("Assemble done | weekly_df rows=%d", len(weekly_df))

        if mtm_parts:
            full_mtm_df = pd.concat(mtm_parts, ignore_index=True)
            full_mtm_df = full_mtm_df.sort_values("Date").reset_index(drop=True)
            full_mtm_df["portfolio_daily_pnl"] = (
                full_mtm_df["portfolio_total_pnl"].diff().fillna(full_mtm_df["portfolio_total_pnl"])
            )
        else:
            full_mtm_df = pd.DataFrame()

        overall_metrics = _summarize_full_mtm(full_mtm_df=full_mtm_df, account_value=account_value)
        logger.info("OVERALL METRICS=%s", overall_metrics)
        logger.info("RUN END | run_id=%s | total_elapsed=%.3fs", run_id, time.perf_counter() - t0)

        return {
            "run_id": run_id,
            "weekly_df": weekly_df,
            "full_mtm_df": full_mtm_df,
            "overall_metrics": overall_metrics,
        }

    except Exception as e:
        logger.exception("RUN FAILED | run_id=%s | error=%s", run_id, e)
        raise


async def main():
    result = await run_full_weekly_timeline(
        start_date="2024-01-01",
        end_date="2024-12-31",
        model_name="deepseek-reasoner",
        account_value=1_000_000,
        risk_pct=0.01,
        save_db=True,
        candidate_limit=250,
        max_pick_for_fulltext=35,
        keyword=None,
        save_selected_fulltext_to_db=True,
        max_concurrency=3,
    )

    print(result["run_id"])
    print(result["weekly_df"][[ "decision_date", "scenario", "strategy_type", "cum_pnl", "sharpe" ]])
    print(result["overall_metrics"])

    out_path = "./outputs"
    os.makedirs(out_path, exist_ok=True)
    run_id = result["run_id"]
    result["weekly_df"].to_csv(os.path.join(out_path, f"weekly_summary_{run_id}.csv"), index=False)
    result["full_mtm_df"].to_csv(os.path.join(out_path, f"full_weekly_mtm_path_{run_id}.csv"), index=False)


if __name__ == "__main__":
    asyncio.run(main())
