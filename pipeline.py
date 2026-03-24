import os
import re
import json
import math
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple
import asyncio 
import numpy as np
import pandas as pd
import time
import traceback
from tqdm.auto import tqdm
from utils.data import *
from utils.context import (
    build_news_context_from_db,
    # fetch_last_performance,
    # build_last_experience_block,
)
from utils.llm import call_llm, choose_scenario
from utils.backtest import compute_backtest_metrics
from db.operations import save_performance_record
from get_news.reader_model import refine_for_market
from get_news.keyword_model import suggest_keywords_for_hsi

def _to_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x).normalize()


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def _extract_json_from_text_local(text: str) -> Dict[str, Any]:
    """
    Local lightweight parser for logging only.
    """
    if text is None:
        return {}

    text = str(text).strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            pass

    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            pass

    return {"raw_text": text}


def _prediction_to_dict(final_text: Any) -> Dict[str, Any]:
    if isinstance(final_text, dict):
        return final_text
    if isinstance(final_text, str):
        return _extract_json_from_text_local(final_text)
    return {}


def _safe_json(obj, max_len: int = 3000):
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    if len(text) > max_len:
        text = text[:max_len] + f"... [truncated {len(text) - max_len} chars]"
    return text

def _df_debug_summary(name: str, df: pd.DataFrame, date_col: str = "Date", max_cols: int = 20) -> str:
    if df is None:
        return f"{name}: None"

    parts = [f"{name}: shape={df.shape}"]

    cols = list(df.columns)
    if len(cols) > max_cols:
        parts.append(f"cols={cols[:max_cols]}...(+{len(cols)-max_cols} more)")
    else:
        parts.append(f"cols={cols}")

    if not df.empty and date_col in df.columns:
        try:
            d = pd.to_datetime(df[date_col], errors="coerce")
            parts.append(f"{date_col}_min={d.min()}")
            parts.append(f"{date_col}_max={d.max()}")
            parts.append(f"{date_col}_na={int(d.isna().sum())}")
        except Exception as e:
            parts.append(f"{date_col}_parse_error={e}")

    na_top = df.isna().sum()
    na_top = na_top[na_top > 0].sort_values(ascending=False).head(10)
    if not na_top.empty:
        parts.append(f"top_na={na_top.to_dict()}")

    return " | ".join(parts)

def _series_range_summary(name: str, s: pd.Series) -> str:
    if s is None or len(s) == 0:
        return f"{name}: empty"
    s2 = pd.to_datetime(s, errors="coerce")
    return f"{name}: count={len(s2)}, min={s2.min()}, max={s2.max()}, na={int(s2.isna().sum())}"



def get_monthly_windows_from_features(
    option_df: pd.DataFrame,
    agg_features: pd.DataFrame,
    start_date,
    end_date,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Build monthly windows aligned with month-end features.

    Returns:
        [(feature_date, entry_date, exit_date), ...]

    Logic:
    - feature_date = one row from agg_features['decision_date'] (month-end)
    - entry_date   = first trading day of the NEXT month
    - exit_date    = last trading day of the NEXT month
    - only keep windows whose entry_date is within [start_date, end_date]
    """
    df = option_df.copy()
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    feature_dates = (
        pd.to_datetime(agg_features["decision_date"])
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )

    windows = []

    for feature_date in feature_dates:
        next_month = feature_date + pd.offsets.MonthBegin(1)

        month_mask = df[date_col].dt.to_period("M") == next_month.to_period("M")
        month_dates = df.loc[month_mask, date_col].drop_duplicates().sort_values()

        if month_dates.empty:
            continue

        entry_date = month_dates.min()
        exit_date = month_dates.max()

        if entry_date < start_date or entry_date > end_date:
            continue

        windows.append((feature_date, entry_date, exit_date))

    return windows


def previous_calendar_month_window(decision_date) -> Tuple[str, str]:
    decision_date = _to_ts(decision_date)
    month_start = decision_date.replace(day=1)
    prev_month_end = month_start - pd.Timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)
    return prev_month_start.strftime("%Y-%m-%d"), prev_month_end.strftime("%Y-%m-%d")


def setup_logger(run_id: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_id}.log"), encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def summarize_full_mtm(full_mtm_df: pd.DataFrame,
                       account_value: float = 1_000_000.0) -> Dict[str, Any]:
    if full_mtm_df.empty:
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


async def run_full_timeline(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    model_name: str = "deepseek-reasoner",
    account_value: float = 1_000_000.0,
    risk_pct: float = 0.01,
    save_db: bool = True,
    ) -> Dict[str, Any]:
    """
    Main monthly loop for the full 2024 timeline.
    Assumes data.py handles all one-time data preparation.
    """
    run_id = f"timeline_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    logger = setup_logger(run_id)

    t0 = time.perf_counter()

    logger.info(
        "RUN START | run_id=%s | start_date=%s | end_date=%s | model_name=%s | account_value=%.2f | risk_pct=%.4f | save_db=%s",
        run_id, start_date, end_date, model_name, account_value, risk_pct, save_db
    )

    try:
        # ============================================================
        # 1) Load raw data
        # ============================================================
        t_data = time.perf_counter()
        price_chain_df, hsi_df, vhsi_df, future = get_data()
        logger.debug("RAW DATA LOADED | elapsed=%.3fs", time.perf_counter() - t_data)
        logger.debug(_df_debug_summary("price_chain_df", price_chain_df, date_col="Date"))
        logger.debug(_df_debug_summary("hsi_df", hsi_df, date_col="Date"))
        logger.debug(_df_debug_summary("vhsi_df", vhsi_df, date_col="Date"))
        logger.debug(_df_debug_summary("future", future, date_col="Date"))

        # ============================================================
        # 2) Prepare datasets
        # ============================================================
        t_prepare = time.perf_counter()
        price_chain_df_clean, hsi_df_clean, vhsi_df_clean, index_df = prepare_datasets(
            price_chain_df, hsi_df, vhsi_df, future
        )
        logger.debug("PREPARE DATASETS DONE | elapsed=%.3fs", time.perf_counter() - t_prepare)
        logger.debug(_df_debug_summary("price_chain_df_clean", price_chain_df_clean, date_col="Date"))
        logger.debug(_df_debug_summary("hsi_df_clean", hsi_df_clean, date_col="Date"))
        logger.debug(_df_debug_summary("vhsi_df_clean", vhsi_df_clean, date_col="Date"))
        logger.debug(_df_debug_summary("index_df", index_df, date_col="Date"))

        merged_df = pd.merge(price_chain_df_clean, index_df, on="Date", how="left")
        logger.debug(_df_debug_summary("merged_df", merged_df, date_col="Date"))

        # ============================================================
        # 3) Build features
        # ============================================================
        t_feat = time.perf_counter()
        index_features = add_index_features(merged_df)
        logger.debug("INDEX FEATURES DONE | elapsed=%.3fs", time.perf_counter() - t_feat)
        logger.debug(_df_debug_summary("index_features", index_features, date_col="Date"))

        t_smirk = time.perf_counter()
        smirk_daily = fit_iv_surface_shape_daily(
            df=price_chain_df_clean,
            price_col="SettlementPrice",
            iv_col="ImpliedVolatility",
            rate_col=None,
            r_default=0.0,
            dte_col="dte",
            x_band=np.inf,
            min_points=5,
        )
        logger.debug("SMIRK DAILY DONE | elapsed=%.3fs", time.perf_counter() - t_smirk)
        logger.debug(_df_debug_summary("smirk_daily", smirk_daily, date_col="Date"))

        t_opt_feat = time.perf_counter()
        option_features = add_option_shape_daily_features(shape_df=smirk_daily)
        logger.debug("OPTION FEATURES DONE | elapsed=%.3fs", time.perf_counter() - t_opt_feat)
        logger.debug(_df_debug_summary("option_features", option_features, date_col="Date"))

        t_agg = time.perf_counter()
        agg_features = build_periodic_features(
            index_features=index_features,
            option_features=option_features
        )
        agg_features["decision_date"] = pd.to_datetime(agg_features["decision_date"])
        logger.debug("AGG FEATURES DONE | elapsed=%.3fs", time.perf_counter() - t_agg)
        logger.debug(_df_debug_summary("agg_features", agg_features, date_col="decision_date"))
        logger.debug(_series_range_summary("agg_features.decision_date", agg_features["decision_date"]))


        # ============================================================
        # 4) Build windows
        # ============================================================
        t_window = time.perf_counter()
        windows = get_monthly_windows_from_features(
            option_df=merged_df,
            agg_features=agg_features,
            start_date=start_date,
            end_date=end_date,
        )
        logger.debug("WINDOWS BUILT | elapsed=%.3fs | n_windows=%d", time.perf_counter() - t_window, len(windows))
        logger.debug("WINDOWS SAMPLE HEAD=%s", _safe_json(windows[:5]))
        logger.debug("WINDOWS SAMPLE TAIL=%s", _safe_json(windows[-5:] if windows else []))


        if not windows:
            logger.error("No monthly trading windows found. start_date=%s end_date=%s", start_date, end_date)
            raise ValueError("No monthly trading windows found.")

        logger.info("Timeline ready | run_id=%s | n_windows=%d", run_id, len(windows))

        monthly_rows = []
        mtm_parts = []
        cumulative_offset = 0.0

        # ============================================================
        # 5) Monthly loop
        # ============================================================
        for i, (decision_date, entry_date, exit_date) in enumerate(
            tqdm(windows, desc="Running monthly timeline", total=len(windows)),
            start=1,
        ):
            month_t0 = time.perf_counter()

            logger.info(
                "[%02d/%02d] MONTH START | decision_date=%s | entry_date=%s | exit_date=%s | cumulative_offset_before=%.2f",
                i, len(windows), decision_date, entry_date, exit_date, cumulative_offset
            )

            try:
                # ----------------------------
                # News context
                # ----------------------------
                news_start, news_end = previous_calendar_month_window(decision_date)
                logger.debug(
                    "[%s] news window | news_start=%s | news_end=%s",
                    decision_date.date(), news_start, news_end
                )

                # ----------------------------
                # Refresh market news into DB first
                # ----------------------------
                t_refine = time.perf_counter()
                kwq_bundle= await suggest_keywords_for_hsi(use_static=True)
                refined_rows = await refine_for_market(
                    country_code="HK",
                    anchor_date=decision_date.date() if hasattr(decision_date, "date") else pd.to_datetime(decision_date).date(),
                    lookback_months=LOOKBACK_LEN,
                    kwq_bundle=kwq_bundle,   # <- 这里要确保它存在
                )

                logger.debug(
                    "[%s] refine_for_market done | elapsed=%.3fs | added_rows=%d",
                    decision_date.date(),
                    time.perf_counter() - t_refine,
                    len(refined_rows) if refined_rows else 0,
                )

                # ----------------------------
                # Build DB-backed news context
                # ----------------------------
                t_news = time.perf_counter()
                news_context = await build_news_context_from_db(
                    start_date=news_start,
                    end_date=news_end,
                    keyword=None,
                    limit=100,
                    fulltext_chars=300,
                )

                logger.debug(
                    "[%s] news context built | elapsed=%.3fs | chars=%d | preview=%s",
                    decision_date.date(),
                    time.perf_counter() - t_news,
                    len(news_context) if news_context else 0,
                    (news_context[:500] + "...") if news_context and len(news_context) > 500 else news_context
                )

                # ----------------------------
                # LLM call
                # ----------------------------
                t_llm = time.perf_counter()
                reasoning, final_text = call_llm(
                    features_df=agg_features,
                    as_of_date=decision_date,
                    context=news_context,
                )
                logger.debug(
                    "[%s] LLM done | elapsed=%.3fs | reasoning_len=%d | final_text_len=%d",
                    decision_date.date(),
                    time.perf_counter() - t_llm,
                    len(reasoning) if reasoning else 0,
                    len(final_text) if final_text else 0,
                )
                logger.debug("[%s] LLM final_text=%s", decision_date.date(), final_text)
                logger.debug(
                    "[%s] LLM reasoning preview=%s",
                    decision_date.date(),
                    (reasoning[:1000] + "...") if reasoning and len(reasoning) > 1000 else reasoning
                )

                # ----------------------------
                # Parse prediction
                # ----------------------------
                pred_dict = _prediction_to_dict(final_text)
                logger.debug("[%s] pred_dict=%s", decision_date.date(), _safe_json(pred_dict))

                scenario = choose_scenario(final_text)
                logger.debug("[%s] scenario chosen=%s", decision_date.date(), scenario)

                direction_probs = pred_dict.get("direction_probs")
                volatility_probs = pred_dict.get("volatility_probs")
                logger.debug(
                    "[%s] probs | direction=%s | volatility=%s",
                    decision_date.date(),
                    _safe_json(direction_probs),
                    _safe_json(volatility_probs),
                )

                # ----------------------------
                # Backtest
                # ----------------------------
                t_bt = time.perf_counter()
                bt = compute_backtest_metrics(
                    scenario=scenario,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    option_df=merged_df,
                    account_value=account_value,
                    risk_pct=risk_pct,
                )
                logger.debug("[%s] backtest done | elapsed=%.3fs", decision_date.date(), time.perf_counter() - t_bt)

                metrics = bt["metrics"]
                mtm_df = bt["mtm_df"]
                signals = bt["signals"]
                strategy_type = bt["strategy_type"]

                logger.debug("[%s] strategy_type=%s", decision_date.date(), strategy_type)
                logger.debug("[%s] signals=%s", decision_date.date(), _safe_json(signals))
                logger.debug("[%s] metrics=%s", decision_date.date(), _safe_json(metrics))
                logger.debug(_df_debug_summary(f"mtm_df[{decision_date.date()}]", mtm_df, date_col="Date"))

                # ----------------------------
                # MTM post-process
                # ----------------------------
                if not mtm_df.empty:
                    mtm_df = mtm_df.copy()

                    mtm_date_col = _pick_col(mtm_df, ["Date", "date"])
                    logger.debug("[%s] mtm_date_col=%s", decision_date.date(), mtm_date_col)

                    mtm_df[mtm_date_col] = pd.to_datetime(mtm_df[mtm_date_col]).dt.normalize()

                    mtm_df["decision_date"] = decision_date
                    mtm_df["entry_date"] = entry_date
                    mtm_df["exit_date"] = exit_date
                    mtm_df["scenario"] = scenario
                    mtm_df["strategy_type"] = strategy_type

                    prev_offset = cumulative_offset
                    mtm_df["portfolio_total_pnl"] = mtm_df["Total_PnL"] + cumulative_offset
                    cumulative_offset = float(mtm_df["portfolio_total_pnl"].iloc[-1])

                    logger.debug(
                        "[%s] MTM merged | rows=%d | prev_offset=%.2f | new_offset=%.2f | pnl_start=%.2f | pnl_end=%.2f",
                        decision_date.date(),
                        len(mtm_df),
                        prev_offset,
                        cumulative_offset,
                        float(mtm_df["portfolio_total_pnl"].iloc[0]),
                        float(mtm_df["portfolio_total_pnl"].iloc[-1]),
                    )

                    logger.debug(
                        "[%s] MTM head=%s",
                        decision_date.date(),
                        _safe_json(mtm_df.head(3).to_dict(orient="records"))
                    )
                    logger.debug(
                        "[%s] MTM tail=%s",
                        decision_date.date(),
                        _safe_json(mtm_df.tail(3).to_dict(orient="records"))
                    )

                    mtm_parts.append(mtm_df)
                else:
                    logger.warning("[%s] Empty mtm_df returned from backtest.", decision_date.date())

                # ----------------------------
                # reasoning text
                # ----------------------------
                reasoning_text = "\n\n".join([
                    f"Decision date: {decision_date.date()}",
                    f"Chosen scenario: {scenario}",
                    f"Strategy type: {strategy_type}",
                    f"LLM reasoning:\n{reasoning}" if reasoning else "LLM reasoning: None",
                ])

                # ----------------------------
                # DB save
                # ----------------------------
                if save_db:
                    t_db = time.perf_counter()
                    save_performance_record(
                        run_id=run_id,
                        decision_date=decision_date,
                        final_text=pred_dict,
                        performance_metrics=metrics,
                        reasoning=reasoning_text,
                        model=model_name,
                    )
                    logger.debug("[%s] DB save done | elapsed=%.3fs", decision_date.date(), time.perf_counter() - t_db)

                logger.info(
                    "[%s] MONTH END | scenario=%s | strategy=%s | cum_pnl=%s | return_pct=%s | sharpe=%s | max_dd=%s | month_elapsed=%.3fs",
                    decision_date.date(),
                    scenario,
                    strategy_type,
                    metrics.get("cum_pnl"),
                    metrics.get("return_pct"),
                    metrics.get("sharpe"),
                    metrics.get("max_drawdown_pct"),
                    time.perf_counter() - month_t0,
                )

                monthly_rows.append({
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
                })

            except Exception as e:
                logger.exception(
                    "[%s] MONTH FAILED | entry_date=%s | exit_date=%s | error=%s",
                    decision_date, entry_date, exit_date, e
                )
                raise

        # ============================================================
        # 6) Final assemble
        # ============================================================
        monthly_df = pd.DataFrame(monthly_rows)
        logger.debug(_df_debug_summary("monthly_df", monthly_df, date_col="decision_date"))

        if mtm_parts:
            full_mtm_df = pd.concat(mtm_parts, ignore_index=True)
            mtm_date_col = _pick_col(full_mtm_df, ["Date", "date"])
            logger.debug("FULL MTM date col=%s", mtm_date_col)

            full_mtm_df = full_mtm_df.sort_values(mtm_date_col).reset_index(drop=True)
            full_mtm_df["portfolio_daily_pnl"] = (
                full_mtm_df["portfolio_total_pnl"]
                .diff()
                .fillna(full_mtm_df["portfolio_total_pnl"])
            )
            logger.debug(_df_debug_summary("full_mtm_df", full_mtm_df, date_col=mtm_date_col))
            logger.debug("full_mtm_df head=%s", _safe_json(full_mtm_df.head(5).to_dict(orient="records")))
            logger.debug("full_mtm_df tail=%s", _safe_json(full_mtm_df.tail(5).to_dict(orient="records")))
        else:
            full_mtm_df = pd.DataFrame()
            logger.warning("No mtm_parts available. full_mtm_df is empty.")

        # ============================================================
        # 7) Overall metrics
        # ============================================================
        t_summary = time.perf_counter()
        overall_metrics = summarize_full_mtm(
            full_mtm_df=full_mtm_df,
            account_value=account_value,
        )
        logger.debug("SUMMARY DONE | elapsed=%.3fs", time.perf_counter() - t_summary)
        logger.info("OVERALL METRICS=%s", _safe_json(overall_metrics))
        logger.info("RUN END | run_id=%s | total_elapsed=%.3fs", run_id, time.perf_counter() - t0)

        return {
            "run_id": run_id,
            "monthly_df": monthly_df,
            "full_mtm_df": full_mtm_df,
            "overall_metrics": overall_metrics,
        }

    except Exception as e:
        logger.exception("RUN FAILED | run_id=%s | error=%s", run_id, e)
        raise
    
    

async def main():
    result = await run_full_timeline(
        start_date="2024-01-01",
        end_date="2024-12-31",
        model_name="deepseek-reasoner",
        account_value=1_000_000,
        risk_pct=0.01,
        save_db=True,
    )

    print(result["run_id"])
    print(result["monthly_df"][[
        "decision_date", "scenario", "strategy_type", "cum_pnl", "sharpe"
    ]])
    print(result["overall_metrics"])
    
    out_path = "./outputs"
    os.makedirs(out_path, exist_ok=True)
    run_id = result["run_id"]
    result["monthly_df"].to_csv(
        os.path.join(out_path, f"monthly_summary_{run_id}.csv"),
        index=False
    )
    result["full_mtm_df"].to_csv(
        os.path.join(out_path, f"full_mtm_path_{run_id}.csv"),
        index=False
    )

if __name__ == "__main__":
    asyncio.run(main())