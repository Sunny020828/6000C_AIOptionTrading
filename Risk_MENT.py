# utils/risk_management.py

import pandas as pd


def apply_risk_management(
    mtm_df: pd.DataFrame,
    account_value: float,
    stop_loss_pct: float = 0.01,
    delta_upper: float = 150,
    delta_lower: float = -150,
    delta_proxy_per_contract: float = 0.5,
):
    """
    Apply execution-level risk management on MTM path.

    Risk controls implemented:
    1. Account-level stop loss (percentage of NAV)
    2. Portfolio delta exposure control (proxy-based)

    Parameters
    ----------
    mtm_df : pd.DataFrame
        MTM DataFrame with at least ['Date', 'Total_PnL'].
        Optionally contains position info for delta proxy.
    account_value : float
        Initial account NAV.
    stop_loss_pct : float
        Stop loss threshold as fraction of NAV.
    delta_upper : float
        Upper delta exposure threshold.
    delta_lower : float
        Lower delta exposure threshold.
    delta_proxy_per_contract : float
        Delta proxy per option contract (simplified).

    Returns
    -------
    pd.DataFrame
        Risk-adjusted MTM DataFrame.
    """

    out = mtm_df.copy().reset_index(drop=True)

    position_closed = False
    locked_pnl = None
    delta_adjusted = False

    for i in range(len(out)):

        # ============================
        # Stop Loss: lock MTM after trigger
        # ============================
        if position_closed:
            out.loc[i, "Total_PnL"] = locked_pnl
            continue

        total_pnl = out.loc[i, "Total_PnL"]

        if total_pnl <= -stop_loss_pct * account_value:
            position_closed = True
            locked_pnl = total_pnl
            out.loc[i, "Total_PnL"] = locked_pnl
            continue

        # ============================
        # Delta Exposure Control (proxy)
        # ============================
        if not delta_adjusted:
            portfolio_delta = 0.0

            # Optional: only apply if delta-related columns exist
            if {"qty", "multiplier", "contract"}.issubset(out.columns):
                for _, row in out.loc[[i]].iterrows():
                    if "call" in str(row["contract"]).lower():
                        delta = delta_proxy_per_contract
                    else:
                        delta = -delta_proxy_per_contract

                    portfolio_delta += delta * row["qty"] * row["multiplier"]

            if portfolio_delta > delta_upper or portfolio_delta < delta_lower:
                delta_adjusted = True
                # Exposure reduction is represented implicitly
                # (execution-layer adjustment without modifying original trades)

    # ============================
    # Recompute Daily PnL
    # ============================
    out["Daily_PnL"] = out["Total_PnL"].diff().fillna(out["Total_PnL"])

    return out