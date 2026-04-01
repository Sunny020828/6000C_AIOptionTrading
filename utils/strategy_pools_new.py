import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

STRAT_LOG = logging.getLogger(__name__)


@dataclass
class WeeklyStrategyCarry:
    """Cross-week option book state for the weekly timeline backtest."""

    current_scenario: Optional[str] = None
    current_signals: Optional[List[Dict[str, Any]]] = None
    current_meta: Optional[Dict[str, Any]] = None
    current_strategy_type: Optional[str] = None
    current_block_base_portfolio: float = 0.0


def _underlying_close_series(
    option_df: pd.DataFrame, date_col: str = "Date"
) -> Optional[pd.Series]:
    if option_df is None or option_df.empty or "Close" not in option_df.columns:
        return None
    sub = option_df[[date_col, "Close"]].dropna(subset=[date_col, "Close"])
    if sub.empty:
        return None
    sub = sub.copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce").dt.normalize()
    sub = sub.dropna(subset=[date_col])
    return (
        sub.drop_duplicates(subset=[date_col])
        .set_index(date_col)["Close"]
        .sort_index()
        .astype(float)
    )


def _pad_mtm_flat_through_end(
    mtm_df: pd.DataFrame,
    option_df: pd.DataFrame,
    end_date: Any,
    date_col: str = "Date",
) -> pd.DataFrame:
    """After an early risk exit, hold MTM flat through the week end for a continuous path."""
    if mtm_df is None or mtm_df.empty:
        return mtm_df
    end_date = pd.to_datetime(end_date).normalize()
    last_d = pd.to_datetime(mtm_df[date_col].iloc[-1]).normalize()
    if last_d >= end_date:
        return mtm_df
    od_dates = pd.to_datetime(option_df[date_col], errors="coerce").dt.normalize()
    mask = (od_dates >= last_d) & (od_dates <= end_date)
    future_dates = sorted(
        pd.to_datetime(option_df.loc[mask, date_col], errors="coerce")
        .dropna()
        .dt.normalize()
        .unique()
        .tolist()
    )
    future_dates = [d for d in future_dates if d > last_d]
    if not future_dates:
        return mtm_df
    last_pnl = float(mtm_df["Total_PnL"].iloc[-1])
    add_rows = []
    for d in future_dates:
        row = {c: np.nan for c in mtm_df.columns}
        row[date_col] = d
        row["Total_PnL"] = last_pnl
        row["Daily_PnL"] = 0.0
        add_rows.append(row)
    ext = pd.DataFrame(add_rows)
    out = pd.concat([mtm_df, ext], ignore_index=True)
    return out.sort_values(date_col).reset_index(drop=True)


def run_weekly_prediction_cycle(
    option_df: pd.DataFrame,
    *,
    scenario: str,
    direction_probs: Dict[str, Any],
    entry_date: Any,
    exit_date: Any,
    decision_date_saved: Any,
    cumulative_offset: float,
    carry: Optional[WeeklyStrategyCarry] = None,
    account_value: float = 1_000_000.0,
    risk_pct: float = 0.01,
    contract_multiplier: float = 50.0,
    dte_rebalance_threshold_days: int = 10,
    scenario_reversal_prob_threshold: float = 0.33,
    max_contracts: int = 80,
    intraweek_stop_loss_pct: Optional[float] = 0.38,
    intraweek_delta_limit: float = 160.0,
    risk_budget_multiplier: float = 1.25,
) -> Dict[str, Any]:
    """
    One prediction week: rebalance decision, open/adjust book, MTM through entry→exit,
    weekly performance metrics, and updated carry for the next week.

    Parameters
    ----------
    cumulative_offset
        Portfolio total PnL carried into this week (end of prior week).
    carry
        Position/scenario state from the prior week; pass None on the first week.
    intraweek_stop_loss_pct
        If set, flat % of max structural risk to cut the week early (then flat through week-end).
        Pass None to disable.
    risk_budget_multiplier
        Multiplier on (account_value × risk_pct) when sizing contracts (>1 = larger clips).
    """
    if carry is None:
        carry = WeeklyStrategyCarry()

    prev_offset = float(cumulative_offset)
    current_scenario = carry.current_scenario
    current_signals = carry.current_signals
    current_meta = carry.current_meta
    current_strategy_type = carry.current_strategy_type
    current_block_base_portfolio = carry.current_block_base_portfolio

    need_rebalance = False
    rebalance_reason: List[str] = []

    STRAT_LOG.info(
        "week_cycle begin | decision=%s entry=%s exit=%s | llm_scenario=%s | cum_pnl_in=%.2f | "
        "carry_exec_scenario=%s carry_strategy=%s carry_qty_meta=%s",
        decision_date_saved,
        entry_date,
        exit_date,
        scenario,
        prev_offset,
        current_scenario,
        current_strategy_type,
        (current_meta or {}).get("quantity") if current_meta else None,
    )

    if current_signals is None or current_scenario is None:
        need_rebalance = True
        rebalance_reason.append("no_position")
    elif scenario != current_scenario:
        dir_key = str(scenario).split("_", 1)[0].strip().lower()
        p_dir = float(direction_probs.get(dir_key, 0.0) or 0.0)
        STRAT_LOG.info(
            "reversal check | held=%s new=%s | p_%s=%.4f need>=%.4f | direction_probs=%s",
            current_scenario,
            scenario,
            dir_key,
            p_dir,
            scenario_reversal_prob_threshold,
            direction_probs,
        )
        if p_dir >= scenario_reversal_prob_threshold:
            need_rebalance = True
            rebalance_reason.append(
                f"scenario_reversal_p_{dir_key}_ge_{scenario_reversal_prob_threshold}"
            )
        else:
            need_rebalance = False
            rebalance_reason.append(f"scenario_reversal_blocked_p_{dir_key}={p_dir:.3f}")
    else:
        dte_days = compute_dte_days(
            (current_meta or {}).get("expiry") if current_meta else None,
            entry_date,
        )
        STRAT_LOG.info(
            "same_scenario roll | dte=%s threshold<=%s | expiry=%s",
            dte_days,
            dte_rebalance_threshold_days,
            (current_meta or {}).get("expiry") if current_meta else None,
        )
        if dte_days is not None and dte_days <= dte_rebalance_threshold_days:
            need_rebalance = True
            rebalance_reason.append(f"dte_le_{dte_rebalance_threshold_days}")

    risk_exit_reason: Optional[str] = None

    STRAT_LOG.info(
        "rebalance_decision | need=%s reasons=%s",
        need_rebalance,
        rebalance_reason,
    )

    if need_rebalance:
        new_signals = generate_trade_signals(
            scenario=scenario,
            date=entry_date,
            df=option_df,
            account_value=account_value,
            risk_pct=risk_pct,
            max_contracts=max_contracts,
            risk_budget_multiplier=risk_budget_multiplier,
        )
        current_signals = new_signals if new_signals else None
        current_meta = extract_meta_from_signals(current_signals or [])
        current_scenario = scenario
        current_strategy_type = (
            current_meta.get("strategy_type") if current_meta else None
        )
        current_block_base_portfolio = cumulative_offset
        if not current_signals:
            STRAT_LOG.warning(
                "rebalance produced no signals | llm_scenario=%s entry=%s",
                scenario,
                entry_date,
            )
        else:
            STRAT_LOG.info(
                "opened/adjusted | strategy_type=%s meta=%s",
                current_strategy_type,
                {
                    k: current_meta.get(k)
                    for k in (
                        "quantity",
                        "expiry",
                        "underlying_price",
                        "buy_strike",
                        "sell_strike",
                        "put_sell_strike",
                        "call_sell_strike",
                    )
                }
                if current_meta
                else None,
            )
    elif current_signals:
        om = extract_meta_from_signals(current_signals) or {}
        STRAT_LOG.info(
            "carry forward | still %s | signal_open_meta_date=%s strikes=%s qty=%s",
            current_scenario,
            om.get("date"),
            {k: om.get(k) for k in ("buy_strike", "sell_strike", "put_sell_strike", "call_sell_strike")},
            om.get("quantity"),
        )

    date_col = "Date"
    if current_signals:
        mtm_df = mark_signals_to_market(
            option_df=option_df,
            signals=current_signals,
            start_date=entry_date,
            end_date=exit_date,
            contract_multiplier=contract_multiplier,
        )
    else:
        mtm_df = pd.DataFrame()

    if (
        intraweek_stop_loss_pct is not None
        and current_signals
        and mtm_df is not None
        and not mtm_df.empty
    ):
        u = _underlying_close_series(option_df, date_col=date_col)
        if u is not None and not u.empty:
            mtm_try, _exit_d, risk_exit_reason = apply_risk_management(
                mtm_df,
                option_df,
                current_signals,
                entry_date,
                u,
                stop_loss_pct=intraweek_stop_loss_pct,
                delta_limit=intraweek_delta_limit,
                contract_multiplier=contract_multiplier,
            )
            STRAT_LOG.info(
                "intraweek risk | stop_pct=%s delta_cap=%s -> exit=%s",
                intraweek_stop_loss_pct,
                intraweek_delta_limit,
                risk_exit_reason,
            )
            if risk_exit_reason is not None:
                mtm_df = _pad_mtm_flat_through_end(
                    mtm_try, option_df, exit_date, date_col=date_col
                )
            else:
                mtm_df = mtm_try
        else:
            STRAT_LOG.warning(
                "intraweek risk skipped | no underlying Close series on option_df"
            )

    if mtm_df is None or mtm_df.empty:
        mask = (
            (option_df[date_col] >= pd.to_datetime(entry_date))
            & (option_df[date_col] <= pd.to_datetime(exit_date))
        )
        dates = (
            pd.to_datetime(option_df.loc[mask, date_col], errors="coerce")
            .dropna()
            .dt.normalize()
            .unique()
            .tolist()
        )
        dates = sorted(dates)
        base = current_block_base_portfolio
        mtm_df = pd.DataFrame(
            {
                "Date": dates,
                "Total_PnL": [prev_offset - base] * len(dates),
                "Daily_PnL": [0.0] * len(dates),
            }
        )

    mtm_df = mtm_df.copy()
    mtm_df = mtm_df.sort_values("Date").reset_index(drop=True)
    mtm_df["decision_date"] = decision_date_saved
    mtm_df["entry_date"] = entry_date
    mtm_df["exit_date"] = exit_date
    mtm_df["scenario"] = current_scenario
    mtm_df["strategy_type"] = current_strategy_type

    mtm_df["portfolio_total_pnl"] = mtm_df["Total_PnL"] + current_block_base_portfolio
    mtm_df["portfolio_total_pnl"] = pd.to_numeric(
        mtm_df["portfolio_total_pnl"], errors="coerce"
    ).ffill()
    mtm_df["portfolio_daily_pnl"] = mtm_df["portfolio_total_pnl"].diff()
    if len(mtm_df) > 0:
        mtm_df.loc[0, "portfolio_daily_pnl"] = (
            mtm_df.loc[0, "portfolio_total_pnl"] - prev_offset
        )

    tmp = mtm_df[["portfolio_total_pnl", "portfolio_daily_pnl"]].copy()
    tmp = tmp.rename(
        columns={
            "portfolio_total_pnl": "Total_PnL",
            "portfolio_daily_pnl": "Daily_PnL",
        }
    )
    from utils.backtest import summarize_mtm_df

    metrics = summarize_mtm_df(tmp, account_value=account_value)

    new_cumulative_offset = (
        float(mtm_df["portfolio_total_pnl"].iloc[-1]) if len(mtm_df) else prev_offset
    )

    STRAT_LOG.info(
        "week_cycle end | week_pnl=%.2f cum_pnl=%.2f | mtm_days=%d | metrics_cum=%s",
        new_cumulative_offset - prev_offset,
        new_cumulative_offset,
        len(mtm_df),
        metrics.get("cum_pnl"),
    )

    next_carry = WeeklyStrategyCarry(
        current_scenario=current_scenario,
        current_signals=current_signals,
        current_meta=current_meta,
        current_strategy_type=current_strategy_type,
        current_block_base_portfolio=current_block_base_portfolio,
    )

    return {
        "mtm_df": mtm_df,
        "metrics": metrics,
        "new_cumulative_offset": new_cumulative_offset,
        "carry": next_carry,
        "need_rebalance": need_rebalance,
        "rebalance_reason": rebalance_reason,
        "executed_scenario": current_scenario,
        "strategy_type": current_strategy_type,
        "signals": current_signals,
        "intraweek_risk_exit": risk_exit_reason,
    }


def get_options_for_date(
    df: pd.DataFrame,
    date,
    exact_only: bool = False,
    forward_only: bool = True,
    max_shift_days: int | None = 10,
    return_actual_date: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Timestamp | None]:
    """
    Get option data for a target date.

    Rules
    -----
    1. Try exact date first.
    2. If exact_only=False and exact date has no rows:
       - forward_only=True: use the nearest later available trading date
       - forward_only=False: use the nearest available trading date in either direction
    3. If max_shift_days is not None, only allow fallback dates within that distance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Date'.
    date : str / datetime-like
        Target date.
    exact_only : bool
        If True, only return exact-match rows.
    forward_only : bool
        If True, only search forward when exact date is missing.
    max_shift_days : int | None
        Maximum allowed calendar-day shift.
    return_actual_date : bool
        If True, return (subset_df, actual_date_used).

    Returns
    -------
    pd.DataFrame
        Rows for the selected date.
    or
    (pd.DataFrame, pd.Timestamp | None)
        If return_actual_date=True.
    """
    if "Date" not in df.columns:
        raise KeyError(f"'Date' not found in columns: {df.columns.tolist()}")

    target_date = pd.to_datetime(date).normalize()

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["Date"])

    # exact match first
    exact = out[out["Date"] == target_date].copy()
    if not exact.empty:
        return (exact, target_date) if return_actual_date else exact

    if exact_only:
        return (pd.DataFrame(), None) if return_actual_date else pd.DataFrame()

    available_dates = sorted(out["Date"].drop_duplicates())
    if not available_dates:
        return (pd.DataFrame(), None) if return_actual_date else pd.DataFrame()

    chosen_date = None

    if forward_only:
        future_dates = [d for d in available_dates if d >= target_date]
        if future_dates:
            chosen_date = future_dates[0]
    else:
        chosen_date = min(available_dates, key=lambda d: abs((d - target_date).days))

    if chosen_date is None:
        return (pd.DataFrame(), None) if return_actual_date else pd.DataFrame()

    if max_shift_days is not None:
        if abs((chosen_date - target_date).days) > max_shift_days:
            return (pd.DataFrame(), None) if return_actual_date else pd.DataFrame()

    selected = out[out["Date"] == chosen_date].copy()
    return (selected, chosen_date) if return_actual_date else selected




def get_target_expiry(day_data: pd.DataFrame, min_dte: int = 21, max_dte: int = 50):
    """
    Pick an expiry whose dte sits in [min_dte, max_dte], preferring the band mid-point.
    Default 21–50 aligns better with weekly holds than a tight 30–45 window.

    If the band is empty, use the smallest dte >= min_dte; if still none, fall back to
    the largest available dte so we still trade rather than skipping the week.
    """
    if day_data.empty:
        return None

    expiries = day_data[["ExpDate", "dte"]].drop_duplicates()
    mid = (min_dte + max_dte) / 2.0
    candidates = expiries[(expiries["dte"] >= min_dte) & (expiries["dte"] <= max_dte)]

    if not candidates.empty:
        target = candidates.iloc[(candidates["dte"] - mid).abs().argmin()]["ExpDate"]
    else:
        candidates = expiries[expiries["dte"] >= min_dte]
        if not candidates.empty:
            target = candidates.nsmallest(1, "dte")["ExpDate"].iloc[0]
        else:
            fallback = expiries.nlargest(1, "dte")
            target = fallback["ExpDate"].iloc[0] if not fallback.empty else None

    return target


def find_closest_strike(strikes, target):
    """Return the strike price closest to the target value."""
    return min(strikes, key=lambda x: abs(x - target))


def black_delta(S, K, T, r, sigma, option_type):
    """
    Black-76 style delta approximation.
    option_type: 'call' or 'put'
    """
    if T <= 0 or sigma <= 0 or pd.isna(sigma):
        return np.nan

    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)


def estimate_delta(row, S, r=0.02):
    """Estimate Delta from one option row."""
    T = row["dte"] / 365.0
    sigma = row.get("ImpliedVolatility", np.nan)

    if pd.isna(sigma) or sigma <= 0:
        return np.nan

    opt_type = str(row["Type"]).lower()
    if opt_type == "c":
        opt_type = "call"
    elif opt_type == "p":
        opt_type = "put"

    return black_delta(S, row["StrikePrice"], T, r, sigma, opt_type)

def calculate_portfolio_delta(
    option_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    current_date,
    S,
    contract_col="Series",
    date_col="Date",
    multiplier=50
):
    """
    Calculate portfolio NET DELTA exposure
    """
    df = option_df.copy()

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    current_date = pd.to_datetime(current_date).normalize()

    legs, _ = signals_to_position_legs(signals, multiplier)

    if not legs:
        return 0.0

    total_delta = 0.0

    for leg in legs:
        contract = leg["contract"]

        sub = df[
            (df[contract_col] == contract) &
            (df[date_col] == current_date)
        ]

        if sub.empty:
            continue

        row = sub.iloc[0]

        delta = estimate_delta(row, S)

        if pd.isna(delta):
            continue

        total_delta += delta * leg["qty"] * leg["sign"] * multiplier

    return total_delta

def filter_liquid_options(df: pd.DataFrame, min_volume: int = 10, min_oi: int = 10) -> pd.DataFrame:
    """
    Liquidity filtering.
    Here Open is treated as open interest, consistent with your current code.
    """
    out = df.copy()

    if "Volume" in out.columns:
        vol = pd.to_numeric(out["Volume"], errors="coerce")
        if vol.notna().any():
            out = out[vol >= min_volume]

    # if "Open" in out.columns:
    #     oi = pd.to_numeric(out["Open"], errors="coerce")
    #     if oi.notna().any():
    #         out = out[oi >= min_oi]

    return out


def iron_condor_budget_loss_points(
    put_max_loss_pts: float,
    call_max_loss_pts: float,
    *,
    second_leg_fraction: float = 0.18,
) -> float:
    """
    Capital-at-risk for sizing / stops on an iron condor.

    Summing both vertical max-loss points **double counts**: in a one-sided move you
    typically lose ~one wing’s risk (the breached side), not both wing maxima at once.
    We budget ``max(wings) + second_leg_fraction * min(wings)`` (small residual on the
    other side). This was inflating $/contract (e.g. 301 pts → qty stuck at 1).
    """
    a = max(float(put_max_loss_pts), float(call_max_loss_pts))
    b = min(float(put_max_loss_pts), float(call_max_loss_pts))
    return float(a + second_leg_fraction * b)


def _effective_vertical_max_loss(
    *,
    is_credit: bool,
    spread_width: float,
    net_premium: float,
) -> float:
    """
    Robust max loss per vertical (index points, before multiplier).

    Credit: max loss ≈ width - credit received; floor when quotes imply width < credit.
    Debit: max loss ≈ debit paid; floor when debit is zero/negative from bad data.
    """
    w = max(float(spread_width), 1e-9)
    if is_credit:
        raw = w - float(net_premium)
        floor_risk = max(0.05 * w, 1.0)
        return max(raw, floor_risk)
    raw = float(net_premium)
    floor_risk = max(0.02 * w, 0.5)
    return max(raw, floor_risk)


def calculate_position_size(
    max_loss_per_trade: float,
    account_value: float = 1_000_000,
    risk_pct: float = 0.01,
    max_contracts: int = 40,
    risk_budget_multiplier: float = 1.0,
    silent: bool = False,
) -> int:
    """
    Contracts from risk budget (max loss per contract × qty <= risk_amount).
    Caps size so tiny per-contract risk does not explode quantity.
    """
    risk_amount = account_value * risk_pct * risk_budget_multiplier
    if max_loss_per_trade <= 0:
        if not silent:
            STRAT_LOG.warning(
                "position_size skip | non-positive max_loss_per_contract=%s",
                max_loss_per_trade,
            )
        return 0
    raw = int(risk_amount // max_loss_per_trade)
    qty = max(raw, 1)
    out = min(qty, max_contracts)
    if not silent:
        STRAT_LOG.info(
            "position_size | risk_budget=%.2f (acct×risk_pct×mult) | max_loss/contract=%.4f | "
            "raw_qty=%s -> capped_qty=%s (cap=%s)",
            risk_amount,
            max_loss_per_trade,
            raw,
            out,
            max_contracts,
        )
    return out


def generate_single_spread(
    scenario_type,
    opt_type,
    delta_buy_target,
    delta_sell_target,
    S,
    subset,
    use_pct,
    contract_multiplier,
    account_value,
    risk_pct,
    max_contracts: int = 40,
    risk_budget_multiplier: float = 1.0,
    silent_sizing: bool = False,
):
    """
    Generate one vertical spread.
    Returns dict or None.
    """
    subset = subset.copy()

    if subset.empty:
        return None

    # 1) Strike selection
    if use_pct:
        if opt_type == "call":
            buy_pct = 0.02 if abs(delta_buy_target) > 0.3 else 0.03
            sell_pct = 0.05 if abs(delta_sell_target) > 0.15 else 0.06
            buy_target = S * (1 + buy_pct)
            sell_target = S * (1 + sell_pct)
        else:
            buy_pct = 0.02 if abs(delta_buy_target) > 0.3 else 0.03
            sell_pct = 0.05 if abs(delta_sell_target) > 0.15 else 0.06
            buy_target = S * (1 - buy_pct)
            sell_target = S * (1 - sell_pct)

        strikes = sorted(subset["StrikePrice"].dropna().unique())
        if len(strikes) < 2:
            return None

        k_buy = find_closest_strike(strikes, buy_target)
        k_sell = find_closest_strike(strikes, sell_target)

    else:
        subset["Delta_abs"] = subset["Delta"].abs()

        buy_candidates = subset[subset["Delta_abs"] <= abs(delta_buy_target) * 1.2]
        if buy_candidates.empty:
            buy_candidates = subset.nsmallest(1, "Delta_abs")
        k_buy = buy_candidates.loc[
            (buy_candidates["Delta_abs"] - abs(delta_buy_target)).abs().idxmin(),
            "StrikePrice",
        ]

        sell_candidates = subset[subset["Delta_abs"] >= abs(delta_sell_target) * 0.8]
        if sell_candidates.empty:
            sell_candidates = subset.nlargest(1, "Delta_abs")
        k_sell = sell_candidates.loc[
            (sell_candidates["Delta_abs"] - abs(delta_sell_target)).abs().idxmin(),
            "StrikePrice",
        ]

    buy_contract = subset[subset["StrikePrice"] == k_buy].iloc[0]
    sell_contract = subset[subset["StrikePrice"] == k_sell].iloc[0]

    # 2) Force distinct strikes
    if buy_contract["StrikePrice"] == sell_contract["StrikePrice"]:
        strikes = sorted(subset["StrikePrice"].dropna().unique())
        if len(strikes) < 2:
            return None

        idx = strikes.index(k_buy)
        if opt_type == "call":
            if k_buy <= k_sell:
                if idx > 0:
                    k_buy = strikes[idx - 1]
                elif idx < len(strikes) - 1:
                    k_buy = strikes[idx + 1]
            else:
                if idx < len(strikes) - 1:
                    k_buy = strikes[idx + 1]
                elif idx > 0:
                    k_buy = strikes[idx - 1]
        else:
            if k_buy >= k_sell:
                if idx < len(strikes) - 1:
                    k_buy = strikes[idx + 1]
                elif idx > 0:
                    k_buy = strikes[idx - 1]
            else:
                if idx > 0:
                    k_buy = strikes[idx - 1]
                elif idx < len(strikes) - 1:
                    k_buy = strikes[idx + 1]

        buy_contract = subset[subset["StrikePrice"] == k_buy].iloc[0]

    # 3) Liquidity filter
    buy_df = filter_liquid_options(pd.DataFrame([buy_contract]))
    sell_df = filter_liquid_options(pd.DataFrame([sell_contract]))
    if buy_df.empty or sell_df.empty:
        return None

    buy_contract = buy_df.iloc[0]
    sell_contract = sell_df.iloc[0]

    # 4) Prices
    price_col = "SettlementPrice"
    if price_col not in buy_contract.index:
        return None

    buy_price = float(buy_contract[price_col])
    sell_price = float(sell_contract[price_col])

    # 5) Max loss (floored for stable sizing)
    spread_width = abs(float(k_sell) - float(k_buy))
    if "credit" in scenario_type:
        net_collect = sell_price - buy_price
        max_loss_per_spread = _effective_vertical_max_loss(
            is_credit=True, spread_width=spread_width, net_premium=net_collect
        )
    else:
        net_collect = buy_price - sell_price
        max_loss_per_spread = _effective_vertical_max_loss(
            is_credit=False, spread_width=spread_width, net_premium=net_collect
        )

    max_loss_per_contract = max_loss_per_spread * contract_multiplier
    quantity = calculate_position_size(
        max_loss_per_contract,
        account_value=account_value,
        risk_pct=risk_pct,
        max_contracts=max_contracts,
        risk_budget_multiplier=risk_budget_multiplier,
        silent=silent_sizing,
    )

    if quantity == 0:
        return None

    return {
        "buy_contract": buy_contract,
        "sell_contract": sell_contract,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "k_buy": float(k_buy),
        "k_sell": float(k_sell),
        "quantity": int(quantity),
        "max_loss_per_spread": float(max_loss_per_spread),
        "scenario_type": scenario_type,
    }


def generate_trade_signals(
    scenario,
    date,
    df,
    account_value=1_000_000,
    risk_pct=0.01,
    max_contracts: int = 40,
    risk_budget_multiplier: float = 1.0,
):
    """
    Generate trading signals using original column names.
    """
    STRAT_LOG.info(
        "generate_trade_signals | scenario=%s date=%s acct=%.0f risk_pct=%.4f mult=%.3f max_c=%s",
        scenario,
        date,
        account_value,
        risk_pct,
        risk_budget_multiplier,
        max_contracts,
    )
    df = df.copy()

    required_cols = ["Date", "ExpDate", "StrikePrice", "Type", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # normalize types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["ExpDate"] = pd.to_datetime(df["ExpDate"], errors="coerce").dt.normalize()
    date = pd.to_datetime(date).normalize()

    numeric_cols = [
        "StrikePrice", "SettlementPrice", "ImpliedVolatility", "dte", "Close",
        "Volume", "Open", "High", "Low", "Returns", "F",
        "Open_vhsi", "High_vhsi", "Low_vhsi", "Close_vhsi"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "ExpDate", "StrikePrice", "Type", "dte", "Close"])

    df["Type"] = df["Type"].astype(str).str.lower()
    df = df[df["Type"].isin(["call", "put", "c", "p"])]
    df["Type"] = df["Type"].replace({"c": "call", "p": "put"})

    day_data, actual_trade_date = get_options_for_date(df, date, return_actual_date=True)
    if day_data.empty:
        STRAT_LOG.warning("no option rows | requested_date=%s", date)
        return []

    if "dte" in day_data.columns:
        _dte = pd.to_numeric(day_data["dte"], errors="coerce")
        ok = _dte.fillna(0) > 0
        if bool(ok.any()):
            n0, n1 = len(day_data), int(ok.sum())
            day_data = day_data.loc[ok].copy()
            if n1 < n0:
                STRAT_LOG.info(
                    "day_data dte filter | dropped_zero_or_nan_dte rows=%s -> %s",
                    n0,
                    n1,
                )
        else:
            STRAT_LOG.warning(
                "day_data all dte<=0 | keeping rows anyway | date=%s",
                date,
            )

    S = float(day_data["Close"].iloc[0])

    target_exp = get_target_expiry(day_data)
    if target_exp is None:
        STRAT_LOG.warning("no target expiry | date=%s S=%.2f", date, S)
        return []
    STRAT_LOG.info(
        "option snapshot | trade_date=%s underlying=%.4f target_exp=%s dte_row0=%s",
        actual_trade_date or date,
        S,
        target_exp,
        day_data["dte"].iloc[0] if "dte" in day_data.columns else None,
    )

    # High-vol: slightly wider short wings / tighter bodies vs low-vol (tail + gamma aware).
    scenario_config = {
        "bull_highvol": {"type": "put_credit_spread", "opt": "put", "buy_delta": 0.10, "sell_delta": 0.24},
        "bull_lowvol": {"type": "call_debit_spread", "opt": "call", "buy_delta": 0.30, "sell_delta": 0.15},
        "bull_medianvol": {"type": "call_debit_spread", "opt": "call", "buy_delta": 0.40, "sell_delta": 0.20},
        "bear_highvol": {"type": "call_credit_spread", "opt": "call", "buy_delta": 0.10, "sell_delta": 0.24},
        "bear_lowvol": {"type": "put_debit_spread", "opt": "put", "buy_delta": -0.30, "sell_delta": -0.15},
        "bear_medianvol": {"type": "put_debit_spread", "opt": "put", "buy_delta": -0.40, "sell_delta": -0.20},
        "range_highvol": {
            "type": "iron_condor",
            "opt": None,
            "put_buy_delta": 0.08,
            "put_sell_delta": 0.20,
            "call_buy_delta": 0.08,
            "call_sell_delta": 0.20,
        },
        "range_lowvol": {
            "type": "iron_condor",
            "put_buy_delta": 0.05,
            "put_sell_delta": 0.14,
            "call_buy_delta": 0.05,
            "call_sell_delta": 0.14,
        },
        "range_medianvol": {
            "type": "iron_condor",
            "opt": None,
            "put_buy_delta": 0.08,
            "put_sell_delta": 0.22,
            "call_buy_delta": 0.08,
            "call_sell_delta": 0.22,
        },
    }

    if scenario not in scenario_config:
        STRAT_LOG.warning("unknown scenario=%s date=%s", scenario, date)
        return []

    config = scenario_config[scenario]
    contract_multiplier = 50

    if config["type"] == "iron_condor":
        put_subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == "put")].copy()
        call_subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == "call")].copy()

        if put_subset.empty or call_subset.empty:
            STRAT_LOG.warning("iron_condor missing leg | date=%s", date)
            return []

        put_subset["Delta"] = put_subset.apply(lambda row: estimate_delta(row, S), axis=1)
        call_subset["Delta"] = call_subset.apply(lambda row: estimate_delta(row, S), axis=1)

        use_pct_put = put_subset["Delta"].isna().all()
        use_pct_call = call_subset["Delta"].isna().all()

        put_result = generate_single_spread(
            scenario_type="put_credit_spread",
            opt_type="put",
            delta_buy_target=config["put_buy_delta"],
            delta_sell_target=config["put_sell_delta"],
            S=S,
            subset=put_subset,
            use_pct=use_pct_put,
            contract_multiplier=contract_multiplier,
            account_value=account_value,
            risk_pct=risk_pct,
            max_contracts=max_contracts,
            risk_budget_multiplier=risk_budget_multiplier,
            silent_sizing=True,
        )

        call_result = generate_single_spread(
            scenario_type="call_credit_spread",
            opt_type="call",
            delta_buy_target=config["call_buy_delta"],
            delta_sell_target=config["call_sell_delta"],
            S=S,
            subset=call_subset,
            use_pct=use_pct_call,
            contract_multiplier=contract_multiplier,
            account_value=account_value,
            risk_pct=risk_pct,
            max_contracts=max_contracts,
            risk_budget_multiplier=risk_budget_multiplier,
            silent_sizing=True,
        )

        if put_result is None or call_result is None:
            STRAT_LOG.warning("iron_condor strike select failed | date=%s", date)
            return []

        put_ml = float(put_result["max_loss_per_spread"])
        call_ml = float(call_result["max_loss_per_spread"])
        naive_sum_pts = put_ml + call_ml
        budget_pts = iron_condor_budget_loss_points(put_ml, call_ml)
        combined_contract = budget_pts * contract_multiplier
        qty = calculate_position_size(
            combined_contract,
            account_value=account_value,
            risk_pct=risk_pct,
            max_contracts=max_contracts,
            risk_budget_multiplier=risk_budget_multiplier,
        )
        if qty < 1:
            return []

        STRAT_LOG.info(
            "iron_condor sized | put_ml=%.4f call_ml=%.4f naive_sum_pts=%.4f budget_pts=%.4f "
            "| $/contract=%.2f qty=%s (sum would be $%.2f /contract)",
            put_ml,
            call_ml,
            naive_sum_pts,
            budget_pts,
            combined_contract,
            qty,
            naive_sum_pts * contract_multiplier,
        )

        signals = [
            {
                "action": "SELL",
                "contract": put_result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "put_sell",
                "price_est": put_result["sell_price"],
            },
            {
                "action": "BUY",
                "contract": put_result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "put_buy",
                "price_est": put_result["buy_price"],
            },
            {
                "action": "SELL",
                "contract": call_result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "call_sell",
                "price_est": call_result["sell_price"],
            },
            {
                "action": "BUY",
                "contract": call_result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "call_buy",
                "price_est": call_result["buy_price"],
            },
            {
                "meta": {
                    "date": date,
                    "scenario": scenario,
                    "strategy_type": "iron_condor",
                    "underlying_price": S,
                    "expiry": target_exp,
                    "put_sell_strike": put_result["k_sell"],
                    "put_buy_strike": put_result["k_buy"],
                    "call_sell_strike": call_result["k_sell"],
                    "call_buy_strike": call_result["k_buy"],
                    "quantity": qty,
                    "put_max_loss": put_ml,
                    "call_max_loss": call_ml,
                    "budget_max_loss_per_spread": budget_pts,
                }
            },
        ]
        return signals

    else:
        opt_type = config["opt"]
        subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == opt_type)].copy()

        if subset.empty:
            STRAT_LOG.warning("empty subset | opt=%s date=%s", opt_type, date)
            return []

        subset["Delta"] = subset.apply(lambda row: estimate_delta(row, S), axis=1)
        use_pct = subset["Delta"].isna().all()

        result = generate_single_spread(
            scenario_type=config["type"],
            opt_type=opt_type,
            delta_buy_target=config["buy_delta"],
            delta_sell_target=config["sell_delta"],
            S=S,
            subset=subset,
            use_pct=use_pct,
            contract_multiplier=contract_multiplier,
            account_value=account_value,
            risk_pct=risk_pct,
            max_contracts=max_contracts,
            risk_budget_multiplier=risk_budget_multiplier,
        )

        if result is None:
            STRAT_LOG.warning("vertical spread selection failed | date=%s type=%s", date, config["type"])
            return []

        qty = result["quantity"]
        STRAT_LOG.info(
            "vertical spread | type=%s k_buy=%s k_sell=%s max_loss_pts=%.4f qty=%s",
            config["type"],
            result.get("k_buy"),
            result.get("k_sell"),
            result.get("max_loss_per_spread"),
            qty,
        )
        signals = []

        if "credit" in config["type"]:
            signals.append({
                "action": "SELL",
                "contract": result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "sell_leg",
                "price_est": result["sell_price"],
            })
            signals.append({
                "action": "BUY",
                "contract": result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "buy_leg",
                "price_est": result["buy_price"],
            })
        else:
            signals.append({
                "action": "BUY",
                "contract": result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "buy_leg",
                "price_est": result["buy_price"],
            })
            signals.append({
                "action": "SELL",
                "contract": result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "sell_leg",
                "price_est": result["sell_price"],
            })

        signals.append({
            "meta": {
                "date": date,
                "scenario": scenario,
                "strategy_type": config["type"],
                "underlying_price": S,
                "expiry": target_exp,
                "buy_strike": result["k_buy"],
                "sell_strike": result["k_sell"],
                "quantity": qty,
                "max_loss_per_spread": result["max_loss_per_spread"],
                "budget_max_loss_per_spread": float(result["max_loss_per_spread"]),
            }
        })

        return signals


def signals_to_position_legs(signals: List[Dict[str, Any]], contract_multiplier: float = 50.0):
    """
    Convert signals into position legs.
    """
    legs = []
    meta = None

    for sig in signals:
        if "meta" in sig:
            meta = sig["meta"]
            continue

        action = sig["action"].upper()
        sign = 1 if action == "BUY" else -1

        legs.append({
            "contract": sig["contract"],
            "sign": sign,
            "qty": float(sig["quantity"]),
            "entry_price": float(sig["price_est"]),
            "leg": sig.get("leg"),
            "strategy": sig.get("strategy"),
            "multiplier": contract_multiplier,
        })

    return legs, meta


def mark_signals_to_market(
    option_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    start_date,
    end_date,
    price_col: str = "SettlementPrice",
    contract_col: str = "Series",
    date_col: str = "Date",
    contract_multiplier: float = 50.0,
) -> pd.DataFrame:
    """
    Generic MTM engine using original column names.
    pnl_leg_t = sign * (price_t - entry_price) * qty * multiplier
    """
    df = option_df.copy()

    if date_col not in df.columns:
        raise KeyError(f"{date_col} not found in option_df. Available columns: {df.columns.tolist()}")
    if contract_col not in df.columns:
        raise KeyError(f"{contract_col} not found in option_df. Available columns: {df.columns.tolist()}")
    if price_col not in df.columns:
        raise KeyError(f"{price_col} not found in option_df. Available columns: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    legs, meta = signals_to_position_legs(signals, contract_multiplier=contract_multiplier)
    if not legs:
        return pd.DataFrame()

    all_dates = sorted(
        df.loc[(df[date_col] >= start_date) & (df[date_col] <= end_date), date_col].unique()
    )
    if not all_dates:
        return pd.DataFrame()

    contract_prices = {}
    for leg in legs:
        c = leg["contract"]
        sub = df.loc[df[contract_col] == c, [date_col, price_col]].copy()
        sub = sub.sort_values(date_col).drop_duplicates(subset=[date_col])
        sub = sub.set_index(date_col)[price_col]
        contract_prices[c] = sub

    rows = []
    last_price = {leg["contract"]: leg["entry_price"] for leg in legs}

    for d in all_dates:
        total_pnl = 0.0
        row = {"Date": d}

        for i, leg in enumerate(legs, start=1):
            c = leg["contract"]
            px_series = contract_prices.get(c)

            if px_series is not None and d in px_series.index and pd.notna(px_series.loc[d]):
                px = float(px_series.loc[d])
                last_price[c] = px
            else:
                px = last_price[c]

            pnl = leg["sign"] * (px - leg["entry_price"]) * leg["qty"] * leg["multiplier"]
            row[f"Price_leg_{i}"] = px
            row[f"PnL_leg_{i}"] = pnl
            total_pnl += pnl

        row["Total_PnL"] = total_pnl
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    out["Daily_PnL"] = out["Total_PnL"].diff().fillna(out["Total_PnL"])
    return out

def extract_meta_from_signals(signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    `generate_trade_signals()` appends a dict like {"meta": {...}} at the end.
    This helper extracts and returns that meta.
    """
    if not signals:
        return None
    for sig in signals:
        if isinstance(sig, dict) and "meta" in sig:
            return sig.get("meta")
    return None

def compute_dte_days(expiry: Any, current_date: Any) -> Optional[int]:
    """
    Compute days-to-expiry using meta['expiry'] (typically target option ExpDate) and `current_date`.
    Returns None if either input is invalid.
    """
    if expiry is None or current_date is None:
        return None
    try:
        exp = pd.to_datetime(expiry, errors="coerce").normalize()
        cur = pd.to_datetime(current_date, errors="coerce").normalize()
        if pd.isna(exp) or pd.isna(cur):
            return None
        return int((exp - cur).days)
    except Exception:
        return None



def _underlying_value_on_or_before(
    underlying_series: pd.Series, d: Any
) -> float:
    d = pd.to_datetime(d).normalize()
    s = underlying_series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce").normalize()
    s = s[~s.index.isna()]
    if s.empty:
        return float("nan")
    try:
        if d in s.index:
            return float(s.loc[d])
    except (KeyError, TypeError, ValueError):
        pass
    prior = s.loc[s.index <= d]
    if prior.empty:
        return float("nan")
    return float(prior.iloc[-1])


def apply_risk_management(
    mtm_df: pd.DataFrame,
    option_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    start_date,
    underlying_series: pd.Series,
    stop_loss_pct: float = 0.5,
    delta_limit: float = 300.0,
    contract_multiplier: float = 50.0,
):
    """
    Apply:

    1. FLOATING STOP LOSS (% of max risk)
    2. PORTFOLIO NET DELTA LIMIT

    Returns truncated MTM until exit date
    """

    if mtm_df.empty:
        return mtm_df, None, None

    # get meta info
    meta = [x["meta"] for x in signals if "meta" in x][0]

    mult = float(contract_multiplier)
    bud = meta.get("budget_max_loss_per_spread")
    if bud is not None:
        max_risk = float(bud) * meta["quantity"] * mult
        STRAT_LOG.info(
            "risk_mgmt budget | budget_pts=%.4f qty=%s $=%.2f (legacy sum avoided for IC)",
            float(bud),
            meta["quantity"],
            max_risk,
        )
    elif "put_max_loss" in meta:
        max_risk = (
            iron_condor_budget_loss_points(
                float(meta["put_max_loss"]),
                float(meta["call_max_loss"]),
            )
            * meta["quantity"]
            * mult
        )
    else:
        max_risk = float(meta["max_loss_per_spread"]) * meta["quantity"] * mult

    stop_threshold = -stop_loss_pct * max_risk
    STRAT_LOG.info(
        "risk_mgmt | max_structural_risk=%.2f stop_pct=%.3f stop_level_pnl=%.2f delta_limit=%.0f",
        max_risk,
        stop_loss_pct,
        stop_threshold,
        delta_limit,
    )

    mtm_df = mtm_df.copy()

    for i in range(len(mtm_df)):

        d = mtm_df.iloc[i]["Date"]

        pnl = mtm_df.iloc[i]["Total_PnL"]

        S = _underlying_value_on_or_before(underlying_series, d)

        # ---- STOP LOSS CHECK ----
        if pnl <= stop_threshold:
            STRAT_LOG.warning(
                "risk_mgmt STOP_LOSS | date=%s mtm_pnl=%.2f threshold=%.2f",
                d,
                pnl,
                stop_threshold,
            )
            return mtm_df.iloc[: i + 1], d, "STOP_LOSS"

        # ---- DELTA CHECK ----
        if not np.isnan(S):
            net_delta = calculate_portfolio_delta(
                option_df,
                signals,
                d,
                S,
                multiplier=mult,
            )

            if abs(net_delta) >= delta_limit:
                STRAT_LOG.warning(
                    "risk_mgmt DELTA_BREACH | date=%s net_delta=%.1f limit=%.1f",
                    d,
                    net_delta,
                    delta_limit,
                )
                return mtm_df.iloc[: i + 1], d, f"DELTA_BREACH({int(net_delta)})"

    return mtm_df, None, None
