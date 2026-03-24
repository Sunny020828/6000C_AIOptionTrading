import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Any


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




def get_target_expiry(day_data: pd.DataFrame, min_dte: int = 30, max_dte: int = 45):
    """
    Select contract month closest to 30-45 days to expiration.
    If none are in range, take the smallest dte >= min_dte.
    """
    if day_data.empty:
        return None

    expiries = day_data[["ExpDate", "dte"]].drop_duplicates()
    candidates = expiries[(expiries["dte"] >= min_dte) & (expiries["dte"] <= max_dte)]

    if not candidates.empty:
        target = candidates.iloc[
            (candidates["dte"] - (min_dte + max_dte) / 2).abs().argmin()
        ]["ExpDate"]
    else:
        candidates = expiries[expiries["dte"] >= min_dte]
        if not candidates.empty:
            target = candidates.nsmallest(1, "dte")["ExpDate"].iloc[0]
        else:
            target = None

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


def calculate_position_size(max_loss_per_trade, account_value=1_000_000, risk_pct=0.01):
    """
    Calculate number of contracts based on risk budget.
    """
    risk_amount = account_value * risk_pct
    if max_loss_per_trade <= 0:
        return 0
    qty = int(risk_amount // max_loss_per_trade)
    return max(qty, 1)


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

    # 5) Max loss
    spread_width = abs(float(k_sell) - float(k_buy))

    if "credit" in scenario_type:
        net_credit = sell_price - buy_price
        max_loss_per_spread = spread_width - net_credit
    else:
        net_debit = buy_price - sell_price
        max_loss_per_spread = net_debit

    max_loss_per_contract = max_loss_per_spread * contract_multiplier
    quantity = calculate_position_size(
        max_loss_per_contract,
        account_value=account_value,
        risk_pct=risk_pct,
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


def generate_trade_signals(scenario, date, df, account_value=1_000_000, risk_pct=0.01):
    """
    Generate trading signals using original column names.
    """
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

    day_data, actual_trade_date = get_options_for_date(df,date, return_actual_date=True)
    if day_data.empty:
        print(f"[{date}] No data")
        return []
    
    S = float(day_data["Close"].iloc[0])

    target_exp = get_target_expiry(day_data)
    if target_exp is None:
        print(f"[{date}] No suitable expiration")
        return []

    scenario_config = {
        "bull_highvol": {"type": "put_credit_spread", "opt": "put", "buy_delta": 0.12, "sell_delta": 0.28},
        "bull_lowvol": {"type": "call_debit_spread", "opt": "call", "buy_delta": 0.32, "sell_delta": 0.17},
        "bull_medianvol": {"type": "call_debit_spread", "opt": "call", "buy_delta": 0.42, "sell_delta": 0.22},
        "bear_highvol": {"type": "call_credit_spread", "opt": "call", "buy_delta": 0.12, "sell_delta": 0.28},
        "bear_lowvol": {"type": "put_debit_spread", "opt": "put", "buy_delta": -0.32, "sell_delta": -0.17},
        "bear_medianvol": {"type": "put_debit_spread", "opt": "put", "buy_delta": -0.42, "sell_delta": -0.22},
        "range_highvol": {
            "type": "iron_condor",
            "opt": None,
            "put_buy_delta": 0.10,
            "put_sell_delta": 0.22,
            "call_buy_delta": 0.10,
            "call_sell_delta": 0.22,
        },
        "range_lowvol": {
            "type": "iron_condor",
            "put_buy_delta": 0.05,
            "put_sell_delta": 0.15,
            "call_buy_delta": 0.05,
            "call_sell_delta": 0.15,
        },
        "range_medianvol": {
            "type": "iron_condor",
            "put_buy_delta": 0.10,
            "put_sell_delta": 0.25,
            "call_buy_delta": 0.10,
            "call_sell_delta": 0.25,
        },
    }

    if scenario not in scenario_config:
        print(f"[{date}] Unknown scenario: {scenario}")
        return []

    config = scenario_config[scenario]
    contract_multiplier = 50

    if config["type"] == "iron_condor":
        put_subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == "put")].copy()
        call_subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == "call")].copy()

        if put_subset.empty or call_subset.empty:
            print(f"[{date}] Iron condor missing put or call options")
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
        )

        if put_result is None or call_result is None:
            print(f"[{date}] Iron condor contract selection failed")
            return []

        qty = min(put_result["quantity"], call_result["quantity"])
        if qty == 0:
            return []

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
                    "put_max_loss": put_result["max_loss_per_spread"],
                    "call_max_loss": call_result["max_loss_per_spread"],
                }
            },
        ]
        return signals

    else:
        opt_type = config["opt"]
        subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == opt_type)].copy()

        if subset.empty:
            print(f"[{date}] No {opt_type} options")
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
        )

        if result is None:
            print(f"[{date}] Spread selection failed")
            return []

        qty = result["quantity"]
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