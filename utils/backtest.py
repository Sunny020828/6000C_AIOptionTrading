import pandas as pd
import numpy as np
import math
from utils.strategy_pools import generate_trade_signals, mark_signals_to_market, signals_to_position_legs
from typing import Optional, Tuple, Dict, Any, List



def compute_backtest_metrics(
    scenario,
    entry_date,
    exit_date,
    option_df,
    account_value=1_000_000,
    risk_pct=0.01
) -> Dict[str, Any]:
    """
    Run strategy generation + MTM + summary metrics.
    Return structure aligned with run_full_timeline.
    """

    signals = []
    mtm_df = pd.DataFrame()
    strategy_type = None

    if scenario is not None:
        signals = generate_trade_signals(
            scenario=scenario,
            date=entry_date,
            df=option_df,
            account_value=account_value,
            risk_pct=risk_pct,
        )

    if signals:
        meta_list = [x["meta"] for x in signals if "meta" in x]
        if meta_list:
            strategy_type = meta_list[-1].get("strategy_type")

        mtm_df = mark_signals_to_market(
            option_df=option_df,
            signals=signals,
            start_date=entry_date,
            end_date=exit_date,
            price_col="SettlementPrice",
            contract_col="Series",
            date_col="Date",
            contract_multiplier=50.0,
        )

    if mtm_df.empty:
        metrics = {
            "n_days": 0,
            "cum_pnl": np.nan,
            "return_pct": np.nan,
            "ann_return_pct": np.nan,
            "ann_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
            "win_rate_daily": np.nan,
        }
        return {
            "metrics": metrics,
            "mtm_df": mtm_df,
            "signals": signals,
            "strategy_type": strategy_type,
        }

    df = mtm_df.copy()
    df["daily_ret"] = df["Daily_PnL"] / account_value

    n_days = len(df)
    cum_pnl = float(df["Total_PnL"].iloc[-1])
    ret_pct = cum_pnl / account_value

    ann_return = (1 + ret_pct) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else np.nan
    ann_vol = df["daily_ret"].std(ddof=1) * np.sqrt(252) if n_days > 1 else np.nan

    if pd.notna(ann_vol) and ann_vol > 0:
        sharpe = ann_return / ann_vol
    else:
        sharpe = np.nan

    equity = account_value + df["Total_PnL"]
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1
    mdd = float(drawdown.min()) if len(drawdown) else np.nan

    win_rate_daily = float((df["Daily_PnL"] > 0).mean()) if n_days > 0 else np.nan

    metrics = {
        "n_days": int(n_days),
        "cum_pnl": cum_pnl,
        "return_pct": ret_pct,
        "ann_return_pct": ann_return,
        "ann_vol_pct": ann_vol,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd,
        "win_rate_daily": win_rate_daily,
    }

    return {
        "metrics": metrics,
        "mtm_df": mtm_df,
        "signals": signals,
        "strategy_type": strategy_type,
    }
    
    
def summarize_mtm_path(
    mtm_df: pd.DataFrame,
    account_value: float = 1_000_000.0,
) -> Dict[str, Any]:
    if mtm_df.empty:
        return {
            "n_days": 0,
            "cum_pnl": None,
            "return_pct": None,
            "ann_return_pct": None,
            "ann_vol_pct": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "win_rate_daily": None,
        }

    df = mtm_df.copy()
    df["daily_ret"] = df["period_daily_pnl"] / account_value

    n_days = len(df)
    cum_pnl = float(df["period_total_pnl"].iloc[-1])
    ret_pct = cum_pnl / account_value

    if n_days > 0:
        ann_return = (1 + ret_pct) ** (252 / n_days) - 1
    else:
        ann_return = np.nan

    ann_vol = df["daily_ret"].std(ddof=1) * np.sqrt(252) if n_days > 1 else np.nan
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan

    equity = account_value + df["period_total_pnl"]
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1
    mdd = float(drawdown.min()) if len(drawdown) else np.nan

    win_rate = float((df["period_daily_pnl"] > 0).mean()) if n_days > 0 else np.nan

    def _safe(x):
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x) if isinstance(x, (np.floating, float)) else x

    return {
        "n_days": int(n_days),
        "cum_pnl": _safe(cum_pnl),
        "return_pct": _safe(ret_pct),
        "ann_return_pct": _safe(ann_return),
        "ann_vol_pct": _safe(ann_vol),
        "sharpe": _safe(sharpe),
        "max_drawdown_pct": _safe(mdd),
        "win_rate_daily": _safe(win_rate),
    }


def run_monthly_backtest(
    scenario: str,
    entry_date,
    exit_date,
    option_df: pd.DataFrame,
    account_value: float = 1_000_000.0,
    risk_pct: float = 0.01,
    contract_multiplier: float = 50.0,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    One monthly strategy backtest.
    Returns:
        mtm_df, metrics, signals, meta
    """
    signals = generate_trade_signals(
        scenario=scenario,
        date=pd.to_datetime(entry_date),
        df=option_df,
        account_value=account_value,
        risk_pct=risk_pct,
    )

    if not signals:
        metrics = {
            "scenario": scenario,
            "start_date": str(pd.to_datetime(entry_date).date()),
            "end_date": str(pd.to_datetime(exit_date).date()),
            "strategy_type": None,
            "signals": [],
            **summarize_mtm_path(pd.DataFrame(), account_value=account_value),
        }
        return pd.DataFrame(), metrics, [], None

    mtm_df = mark_signals_to_market(
        option_df=option_df,
        signals=signals,
        start_date=entry_date,
        end_date=exit_date,
        contract_multiplier=contract_multiplier,
    )

    _, meta = signals_to_position_legs(signals, contract_multiplier=contract_multiplier)
    metrics = summarize_mtm_path(mtm_df, account_value=account_value)

    metrics.update({
        "scenario": scenario,
        "start_date": str(pd.to_datetime(entry_date).date()),
        "end_date": str(pd.to_datetime(exit_date).date()),
        "strategy_type": meta.get("strategy_type") if meta else None,
        "underlying_price": meta.get("underlying_price") if meta else None,
        "expiry": str(pd.to_datetime(meta.get("expiry")).date()) if meta and meta.get("expiry") is not None else None,
        "signals": signals,
    })

    return mtm_df, metrics, signals, meta


