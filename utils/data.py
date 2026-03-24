import os
from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import os
import math

LOOKBACK_LEN=3
PRED_LEN=1

def get_data():
    price_chain_df=pd.read_csv('data/Monthly_HSI_Options.csv')
    hsi_df=pd.read_csv('data/HSI_data.csv')
    vhsi_df=pd.read_csv('data/VHSI_data.csv')

    def parse_number(x):
        """Parse numbers like '21,123.0' -> 21123.0"""
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace('"','').replace(',','')
        if s == "" or s.lower() in {"nan", "null"}:
            return np.nan
        try:
            return float(s)
        except:
            return np.nan
        


    future_df = pd.read_csv('data/Hang Seng Futures Historical Price Data.csv')
    future_df.columns = future_df.columns.str.replace('"', '', regex=False).str.strip()
    future_df["Date"] = pd.to_datetime(future_df["Date"], dayfirst=True).dt.normalize()

    for col in ["Price","Open","High","Low"]:
        future_df[col] = future_df[col].map(parse_number)


    # 最终你大概率只需要 Date + F(=Price)
    future_df = future_df.rename(columns={"Price":"F"})[["Date","F"]]
    future_df = future_df.sort_values("Date").reset_index(drop=True)
    future=future_df.copy().reset_index(drop=True)
    
    return price_chain_df, hsi_df, vhsi_df, future


def prepare_datasets(price_chain_df: pd.DataFrame,
                     hsi_df: pd.DataFrame,
                     vhsi_df: pd.DataFrame,
                     future: pd.DataFrame,
                     bad_contract_threshold: float = 0.85):
    """Clean and align the three raw dataframes.

    Returns:
        price_chain_df_clean, hsi_df_clean, vhsi_df_clean, index_df, full_df
    """
    # --- price_chain_df ---
    pc = price_chain_df.copy()

    # parse dates
    for col in ["Date", "ExpDate"]:
        if col in pc.columns:
            pc[col] = pd.to_datetime(pc[col], errors="coerce")
            
    pc['dte']=(pc['ExpDate']-pc['Date']).dt.days
    # ensure numeric columns
    num_cols = ["StrikePrice", "SettlementPrice", "ImpliedVolatility", "dte"]
    for col in num_cols:
        if col in pc.columns:
            pc[col] = pd.to_numeric(pc[col], errors="coerce")

    # drop rows with missing mandatory fields
    pc = pc.dropna(subset=["Date"])
    pc = pc.drop_duplicates()

    grp = (
        pc.groupby(["ExpDate", "StrikePrice", "Type"])["SettlementPrice"]
        .agg(
            n_obs=lambda x: x.notna().sum(),
            pct_one=lambda x: np.mean(x.dropna() == 1.0) if x.notna().sum() > 0 else np.nan,
            n_unique=lambda x: x.dropna().nunique()
        )
        .reset_index()
    )
    bad_contracts = grp[(grp["pct_one"] >= bad_contract_threshold)][["ExpDate", "StrikePrice", "Type"]].copy()
    bad_contracts["drop_flag"] = 1
    pc = pd.merge(
        pc,
        bad_contracts,
        on=["ExpDate", "StrikePrice", "Type"],
        how="left",
    )
    pc = pc[pc["drop_flag"] != 1].drop(columns=["drop_flag"])
    
    sort_cols = [c for c in ["Date", "ExpDate", "StrikePrice", "Type"] if c in pc.columns]
    if sort_cols:
        pc = pc.sort_values(sort_cols).reset_index(drop=True)

    # --- hsi_df ---
    hsi = hsi_df.copy()
    if "Date" in hsi.columns:
        hsi["Date"] = pd.to_datetime(hsi["Date"], errors="coerce")
        hsi = hsi.dropna(subset=["Date"])
        hsi = hsi.sort_values("Date")

    for col in ["Close", "High", "Low", "Open", "Volume", "Returns"]:
        if col in hsi.columns:
            hsi[col] = pd.to_numeric(hsi[col], errors="coerce")
            hsi[col] = hsi[col].fillna(method="ffill")

    hsi = hsi.drop_duplicates(subset=["Date"]).reset_index(drop=True)
    hsi=pd.merge(hsi,future,on="Date",how="left")

    
    # --- vhsi_df ---
    vhsi = vhsi_df.copy()
    if "Date" in vhsi.columns:
        vhsi["Date"] = pd.to_datetime(vhsi["Date"], errors="coerce")
        vhsi = vhsi.dropna(subset=["Date"])
        vhsi = vhsi.sort_values("Date")

    for col in ["Open", "High", "Low", "Close"]:
        if col in vhsi.columns:
            vhsi[col] = pd.to_numeric(vhsi[col], errors="coerce")

    vhsi = vhsi.drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # --- merged index dataset (HSI + VHSI) ---
    vhsi_renamed = vhsi.add_suffix("_vhsi")
    if "Date_vhsi" in vhsi_renamed.columns:
        vhsi_renamed = vhsi_renamed.rename(columns={"Date_vhsi": "Date"})

    index_df = pd.merge(
        hsi,
        vhsi_renamed,
        on="Date",
        how="inner",
    )

    return pc, hsi, vhsi, index_df



def add_index_features(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    添加 HSI 和 VHSI 的量价、动量与趋势特征。
    
    增强点：
    1. 引入对数收益率 (Log Returns)，解决简单收益率时间不可加的问题。
    2. 将波动率统一转化为年化形式，量级更直观。
    3. VHSI（作为自带百分比属性的指数）增加绝对差值 (Diff) 特征。
    """
    # 确保时间顺序，避免计算错乱
    df = index_df.copy().sort_values("Date")

    # ==========================================
    # 1. 恒生指数 (HSI) 基础收益率特征
    # ==========================================
    df["hsi_close_lag1"] = df["Close"].shift(1)
    
    # 简单收益率 (供常规计算使用)
    df["hsi_ret_1d"] = df["Close"] / df["hsi_close_lag1"] - 1
    # 对数收益率 (Log Return，统计性质更好，适合求均值和加和)
    df["hsi_log_ret_1d"] = np.log(df["Close"] / df["hsi_close_lag1"])

    # 多时间窗口收益率与波动率
    for win in [5, 10, 20]:
        # N日累计收益率
        df[f"hsi_ret_{win}d"] = df["Close"] / df["Close"].shift(win) - 1
        
        # N日年化滚动波动率 (Shift 1 防止使用当天的收益率预测当天)
        df[f"hsi_vol_{win}d_ann"] = (
            df["hsi_ret_1d"]
            .rolling(window=win, min_periods=win)
            .std()
            .shift(1) 
            * (252 ** 0.5)  # 年化处理
        )

    # ==========================================
    # 2. 恒生指数 (HSI) 趋势与均线特征
    # ==========================================
    for win in [20, 50, 100]:
        # 计算过去的移动平均线 (不包含当天收盘价)
        df[f"hsi_ma_{win}"] = (
            df["Close"]
            .rolling(window=win, min_periods=win)
            .mean()
            .shift(1)
        )
        # 当天收盘价是否站上过去N日均线 (0 or 1)
        df[f"hsi_above_ma_{win}"] = (df["Close"] > df[f"hsi_ma_{win}"]).astype(float)
        
        # 乖离率 (Bias) - 衡量价格偏离均线的程度，比单纯的 0/1 包含更多信息
        df[f"hsi_bias_{win}"] = (df["Close"] - df[f"hsi_ma_{win}"]) / df[f"hsi_ma_{win}"]

    # ==========================================
    # 3. 恒生波幅指数 (VHSI) 动态特征
    # ==========================================
    df["vhsi_close_lag1"] = df["Close_vhsi"].shift(1)
    
    # VHSI 的相对变化率
    df["vhsi_ret_1d"] = df["Close_vhsi"] / df["vhsi_close_lag1"] - 1
    # VHSI 的绝对点数变化 (波动率指数通常看绝对变化也很重要)
    df["vhsi_diff_1d"] = df["Close_vhsi"] - df["vhsi_close_lag1"]

    for win in [5, 10, 20]:
        df[f"vhsi_ret_{win}d"] = df["Close_vhsi"] / df["Close_vhsi"].shift(win) - 1
        # VHSI 自身的滚动波动率
        df[f"vhsi_vol_{win}d"] = (
            df["vhsi_ret_1d"]
            .rolling(window=win, min_periods=win)
            .std()
            .shift(1)
        )

    # ==========================================
    # 4. 波动率体制 (Volatility Regime) 标签
    # ==========================================
    # 使用过去 20 天的 VHSI 均值作为基准
    vhsi_ma_20 = (
        df["Close_vhsi"]
        .rolling(window=20, min_periods=20)
        .mean()
        .shift(1)
    )
    # 标志位：今天的 VHSI 是否高于过去 20 天的平均水平 (高波动环境)
    df["vhsi_high_vol_regime"] = (df["Close_vhsi"] > vhsi_ma_20).astype(float)

    return df

def _safe_to_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c])
    return out


def _prepare_call_put_wide(
    df: pd.DataFrame,
    price_col: str,
) -> pd.DataFrame:
    """
    For each Date x ExpDate x Strike, pivot Call/Put prices side by side.
    """
    required = ["Date", "ExpDate", "StrikePrice", "Type", price_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tmp = df[required].copy()
    tmp["Type"] = tmp["Type"].astype(str).str.lower().str.strip()

    # normalize common labels
    type_map = {
        "call": "call",
        "c": "call",
        "put": "put",
        "p": "put",
    }
    tmp["Type"] = tmp["Type"].map(type_map)

    wide = (
        tmp.pivot_table(
            index=["Date", "ExpDate", "StrikePrice"],
            columns="Type",
            values=price_col,
            aggfunc="first",
        )
        .reset_index()
    )

    # ensure columns exist
    for col in ["call", "put"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["abs_cp_diff"] = (wide["call"] - wide["put"]).abs()
    return wide


def compute_implied_forward_by_parity(
    df: pd.DataFrame,
    price_col: str = "SettlementPrice",
    rate_col: Optional[str] = None,
    r_default: float = 0.0,
    dte_col: str = "dte",
) -> pd.DataFrame:
    """
    Literature-style implied forward F0 for each Date x ExpDate:

    1) find the strike where |Call - Put| is smallest
    2) define ATM strike = that strike
    3) compute F0 = K + exp(r * tau) * (Call - Put)

    Returns one row per Date x ExpDate with:
    - Date, ExpDate
    - atm_strike_parity
    - call_atm, put_atm
    - tau
    - r
    - implied_forward_parity
    - abs_cp_diff_min
    """
    df = _safe_to_datetime(df, ["Date", "ExpDate"]).copy()

    if dte_col not in df.columns:
        raise ValueError(f"Missing '{dte_col}' column.")
    if price_col not in df.columns:
        raise ValueError(f"Missing '{price_col}' column.")

    wide = _prepare_call_put_wide(df, price_col=price_col)

    # need dte / rate at Date x ExpDate x Strike level for the selected ATM strike
    aux_cols = ["Date", "ExpDate", "StrikePrice", dte_col]
    if rate_col is not None and rate_col in df.columns:
        aux_cols.append(rate_col)

    aux = (
        df[aux_cols]
        .drop_duplicates(subset=["Date", "ExpDate", "StrikePrice"])
        .copy()
    )

    wide = wide.merge(aux, on=["Date", "ExpDate", "StrikePrice"], how="left")
    # keep rows with both call and put
    wide = wide.dropna(subset=["call", "put", dte_col]).copy()
    if wide.empty:
        raise ValueError("No valid call-put pairs available to compute implied forward.")

    # pick the strike minimizing |C - P| per Date x ExpDate
    wide = wide.sort_values(
        ["Date", "ExpDate", "abs_cp_diff", "StrikePrice"],
        ascending=[True, True, True, True]
    )

    atm = (
        wide.groupby(["Date", "ExpDate"], as_index=False)
        .first()
        .rename(columns={"StrikePrice": "atm_strike_parity"})
    )

    # tau in years
    atm["tau"] = atm[dte_col] / 365.0

    # risk-free rate
    if rate_col is not None and rate_col in atm.columns:
        atm["r"] = atm[rate_col].fillna(r_default)
    else:
        atm["r"] = r_default

    # literature formula
    atm["implied_forward_parity"] = (
        atm["atm_strike_parity"] + np.exp(atm["r"] * atm["tau"]) * (atm["call"] - atm["put"])
    )

    atm = atm.rename(columns={
        "call": "call_atm",
        "put": "put_atm",
        "abs_cp_diff": "abs_cp_diff_min",
    })

    return atm[
        [
            "Date",
            "ExpDate",
            "atm_strike_parity",
            "call_atm",
            "put_atm",
            dte_col,
            "tau",
            "r",
            "implied_forward_parity",
            "abs_cp_diff_min",
        ]
    ].copy()


def build_otm_smirk_points(
    df: pd.DataFrame,
    forwards_df: pd.DataFrame,
    iv_col: str = "ImpliedVolatility",
    x_band: float = np.inf,
    dte_col: str = "dte",
) -> pd.DataFrame:
    """
    Build OTM smirk fitting points using:
      x = ln(K / F0)

    Keep only:
      - Put when K < F0
      - Call when K > F0
    and only in the near-ATM band abs(x) <= x_band.
    """
    df = _safe_to_datetime(df, ["Date", "ExpDate"]).copy()

    required = ["Date", "ExpDate", "StrikePrice", "Type", iv_col, dte_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.merge(
        forwards_df[["Date", "ExpDate", "implied_forward_parity", "atm_strike_parity", "tau", "r"]],
        on=["Date", "ExpDate"],
        how="left",
    ).copy()

    out = out.dropna(subset=["implied_forward_parity", iv_col]).copy()
    out["Type_norm"] = out["Type"].astype(str).str.lower().str.strip()
    out["Type_norm"] = out["Type_norm"].replace({
        "c": "call",
        "call": "call",
        "p": "put",
        "put": "put",
    })

    out["x"] = np.log(out["StrikePrice"] / out["implied_forward_parity"])
    # print(out["x"])
    # OTM rule from the paper
    is_otm_put = (out["StrikePrice"] < out["implied_forward_parity"]) & (out["Type_norm"] == "put")
    is_otm_call = (out["StrikePrice"] > out["implied_forward_parity"]) & (out["Type_norm"] == "call")
    out["is_otm_selected"] = is_otm_put | is_otm_call

    out = out[out["is_otm_selected"]].copy()
    out = out[out["x"].abs() <= x_band].copy()

    # remove impossible / dirty rows
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["x", iv_col])

    return out.sort_values(["Date", "ExpDate", "StrikePrice", "Type_norm"]).copy()


def _fit_quadratic_with_anchor(
    x: np.ndarray,
    iv: np.ndarray,
    iv_atm: float,
) -> Tuple[float, float, float, float]:
    """
    Fit:
        IV(x) = a + b x + c x^2
    with a forced to equal iv_atm.

    Then:
        y = IV - iv_atm = b x + c x^2

    Returns:
        a, b, c, rmse
    """
    x = np.asarray(x, dtype=float)
    iv = np.asarray(iv, dtype=float)

    y = iv - iv_atm
    X = np.column_stack([x, x**2])

    # OLS on [b, c]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, c = beta[0], beta[1]

    fitted = iv_atm + b * x + c * x**2
    rmse = float(np.sqrt(np.mean((iv - fitted) ** 2)))

    return float(iv_atm), float(b), float(c), rmse


def fit_iv_surface_shape_daily(
    df: pd.DataFrame,
    price_col: str = "SettlementPrice",
    iv_col: str = "ImpliedVolatility",
    rate_col: Optional[str] = None,
    r_default: float = 0.0,
    dte_col: str = "dte",
    x_band: float = np.inf,
    min_points: int = 5,
) -> pd.DataFrame:
    """
    Main pipeline:

    Step 1: compute implied forward F0 by literature parity method
    Step 2: x = ln(K / F0)
    Step 3: keep near-ATM strikes |x| <= x_band
    Step 4: keep OTM puts for K < F0 and OTM calls for K > F0
    Step 5: fit IV(x) = a + b x + c x^2, forcing a = ATM IV at parity strike

    Returns one row per Date x ExpDate with:
    - Date, ExpDate, dte, tau, r
    - implied_forward_parity
    - atm_strike_parity
    - atm_iv
    - level, slope, curvature
    - n_points, rmse
    """
    df = _safe_to_datetime(df, ["Date", "ExpDate"]).copy()

    # ---------- Step 1 ----------
    forwards = compute_implied_forward_by_parity(
        df=df,
        price_col=price_col,
        rate_col=rate_col,
        r_default=r_default,
        dte_col=dte_col,
    )

    # ATM IV at the parity-selected strike:
    atm_iv_df = (
        df.merge(
            forwards[["Date", "ExpDate", "atm_strike_parity"]],
            left_on=["Date", "ExpDate", "StrikePrice"],
            right_on=["Date", "ExpDate", "atm_strike_parity"],
            how="inner",
        )
        .copy()
    )

    # if both call and put exist at ATM strike, take their average IV
    atm_iv_df["Type_norm"] = atm_iv_df["Type"].astype(str).str.lower().str.strip()
    atm_iv_df["Type_norm"] = atm_iv_df["Type_norm"].replace({
        "c": "call",
        "call": "call",
        "p": "put",
        "put": "put",
    })

    atm_iv_agg = (
        atm_iv_df.groupby(["Date", "ExpDate", "atm_strike_parity"], as_index=False)[iv_col]
        .mean()
        .rename(columns={iv_col: "atm_iv"})
    )

    forwards = forwards.merge(
        atm_iv_agg,
        on=["Date", "ExpDate", "atm_strike_parity"],
        how="left",
    )

    # ---------- Steps 2-4 ----------
    pts = build_otm_smirk_points(
        df=df,
        forwards_df=forwards,
        iv_col=iv_col,
        x_band=x_band,
        dte_col=dte_col,
    )

    # ---------- Step 5 ----------
    results = []
    for (date, ExpDate), g in pts.groupby(["Date", "ExpDate"], sort=True):
        row_f = forwards.loc[
            (forwards["Date"] == date) & (forwards["ExpDate"] == ExpDate)
        ]
        if row_f.empty:
            continue

        frow = row_f.iloc[0]
        atm_iv = frow.get("atm_iv", np.nan)

        if pd.isna(atm_iv):
            continue

        g = g.dropna(subset=["x", iv_col]).copy()
        if len(g) < min_points:
            continue

        try:
            a, b, c, rmse = _fit_quadratic_with_anchor(
                x=g["x"].to_numpy(),
                iv=g[iv_col].to_numpy(),
                iv_atm=float(atm_iv),
            )

        except np.linalg.LinAlgError:
            continue

        # simple R^2
        fitted = a + b * g["x"].to_numpy() + c * (g["x"].to_numpy() ** 2)
        y = g[iv_col].to_numpy()
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot

        results.append({
            "Date": date,
            "ExpDate": ExpDate,
            dte_col: int(frow[dte_col]),
            "tau": float(frow["tau"]),
            "r": float(frow["r"]),
            "implied_forward_parity": float(frow["implied_forward_parity"]),
            "atm_strike_parity": float(frow["atm_strike_parity"]),
            "atm_iv": float(atm_iv),
            "level": float(a)/100.0,
            "slope": float(b)/100.0,
            "curvature": float(c)/100.0,
            "n_points": int(len(g)),
            "rmse": float(rmse),
            "r2": float(r2) if pd.notna(r2) else np.nan,
            "x_min": float(g["x"].min()),
            "x_max": float(g["x"].max()),
        })

    res = pd.DataFrame(results).sort_values(["Date", "ExpDate"]).reset_index(drop=True)
    return res


def build_smirk_points_with_fit(
    df: pd.DataFrame,
    fitted_df: pd.DataFrame,
    price_col: str = "SettlementPrice",
    iv_col: str = "ImpliedVolatility",
    rate_col: Optional[str] = None,
    r_default: float = 0.0,
    dte_col: str = "dte",
    x_band: float = np.inf,
) -> pd.DataFrame:
    """
    Optional helper: return fitting points joined with fitted level/slope/curvature,
    useful for plotting diagnostics.
    """
    forwards = compute_implied_forward_by_parity(
        df=df,
        price_col=price_col,
        rate_col=rate_col,
        r_default=r_default,
        dte_col=dte_col,
    )
    pts = build_otm_smirk_points(
        df=df,
        forwards_df=forwards,
        iv_col=iv_col,
        x_band=x_band,
        dte_col=dte_col,
    )
    out = pts.merge(
        fitted_df[["Date", "ExpDate", "level", "slope", "curvature"]],
        on=["Date", "ExpDate"],
        how="left",
    )
    out["iv_fit"] = out["level"] + out["slope"] * out["x"] + out["curvature"] * (out["x"] ** 2)
    return out



import pandas as pd
import numpy as np


def add_option_shape_daily_features(
    shape_df: pd.DataFrame,
    date_col: str = "Date",
    expiry_col: str = "ExpDate",
    dte_col: str = "dte",
    min_points: int = 20,
    min_r2: float = 0.6,
    add_normalized: bool = False,
    norm_window: int = 252,
    clip_z: float = 5.0,
) -> pd.DataFrame:
    """
    Convert Date x Expiry IV-shape panel into Date-level daily option shape features.

    Input columns expected:
    - Date
    - ExpDate
    - dte
    - level
    - slope
    - curvature
    - n_points
    - r2

    Output:
    One row per Date with near/next/far shape features and term differences.
    Optionally also adds rolling normalized (z-score) versions of selected features.
    """

    df = shape_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[expiry_col] = pd.to_datetime(df[expiry_col]).dt.normalize()

    required = [date_col, expiry_col, dte_col, "level", "slope", "curvature", "n_points", "r2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # quality filter
    df = df[(df["r2"] >= min_r2) & (df["n_points"] >= min_points)].copy()

    base_cols = [
        date_col,
        "level_near", "slope_near", "curvature_near", "dte_near",
        "level_next", "slope_next", "curvature_next", "dte_next",
        "level_far", "slope_far", "curvature_far", "dte_far",
        "term_level_next_near", "term_slope_next_near", "term_curvature_next_near",
        "term_level_far_near", "term_slope_far_near", "term_curvature_far_near",
    ]

    if df.empty:
        return pd.DataFrame(columns=base_cols)

    df = df.sort_values([date_col, dte_col, expiry_col]).copy()

    rows = []
    for d, g in df.groupby(date_col, sort=True):
        g = g.sort_values([dte_col, expiry_col]).reset_index(drop=True)

        row = {date_col: d}

        # near
        if len(g) >= 1:
            row["level_near"] = g.loc[0, "level"]
            row["slope_near"] = g.loc[0, "slope"]
            row["curvature_near"] = g.loc[0, "curvature"]
            row["dte_near"] = g.loc[0, dte_col]
        else:
            row["level_near"] = np.nan
            row["slope_near"] = np.nan
            row["curvature_near"] = np.nan
            row["dte_near"] = np.nan

        # next
        if len(g) >= 2:
            row["level_next"] = g.loc[1, "level"]
            row["slope_next"] = g.loc[1, "slope"]
            row["curvature_next"] = g.loc[1, "curvature"]
            row["dte_next"] = g.loc[1, dte_col]
        else:
            row["level_next"] = np.nan
            row["slope_next"] = np.nan
            row["curvature_next"] = np.nan
            row["dte_next"] = np.nan

        # far
        if len(g) >= 3:
            row["level_far"] = g.loc[2, "level"]
            row["slope_far"] = g.loc[2, "slope"]
            row["curvature_far"] = g.loc[2, "curvature"]
            row["dte_far"] = g.loc[2, dte_col]
        else:
            row["level_far"] = np.nan
            row["slope_far"] = np.nan
            row["curvature_far"] = np.nan
            row["dte_far"] = np.nan

        # term differences
        row["term_level_next_near"] = (
            row["level_next"] - row["level_near"]
            if pd.notna(row["level_next"]) and pd.notna(row["level_near"]) else np.nan
        )
        row["term_slope_next_near"] = (
            row["slope_next"] - row["slope_near"]
            if pd.notna(row["slope_next"]) and pd.notna(row["slope_near"]) else np.nan
        )
        row["term_curvature_next_near"] = (
            row["curvature_next"] - row["curvature_near"]
            if pd.notna(row["curvature_next"]) and pd.notna(row["curvature_near"]) else np.nan
        )

        row["term_level_far_near"] = (
            row["level_far"] - row["level_near"]
            if pd.notna(row["level_far"]) and pd.notna(row["level_near"]) else np.nan
        )
        row["term_slope_far_near"] = (
            row["slope_far"] - row["slope_near"]
            if pd.notna(row["slope_far"]) and pd.notna(row["slope_near"]) else np.nan
        )
        row["term_curvature_far_near"] = (
            row["curvature_far"] - row["curvature_near"]
            if pd.notna(row["curvature_far"]) and pd.notna(row["curvature_near"]) else np.nan
        )

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(date_col).reset_index(drop=True)

    # ---------- rolling normalized features ----------
    if add_normalized:
        norm_cols = [
            "level_near", "slope_near", "curvature_near",
            "level_next", "slope_next", "curvature_next",
            "level_far", "slope_far", "curvature_far",
            "term_level_next_near", "term_slope_next_near", "term_curvature_next_near",
            "term_level_far_near", "term_slope_far_near", "term_curvature_far_near",
        ]

        out = out.sort_values(date_col).reset_index(drop=True)

        for col in norm_cols:
            s = out[col].astype(float)

            rolling_mean = s.shift(1).rolling(norm_window, min_periods=max(20, norm_window // 5)).mean()
            rolling_std = s.shift(1).rolling(norm_window, min_periods=max(20, norm_window // 5)).std()

            z = (s - rolling_mean) / rolling_std
            if clip_z is not None:
                z = z.clip(-clip_z, clip_z)

            out[f"{col}_z"] = z

    return out


def summarize_index_window(idx_win: pd.DataFrame) -> dict:
    out = {}
    
    # ==========================================
    # 1. 基础价格与区间累计表现
    # ==========================================
    out["hsi_last_close"] = idx_win["Close"].iloc[-1]
    
    # 区间累计收益率 (使用首尾价格计算，最真实的表现)
    out["hsi_lookback_ret"] = idx_win["Close"].iloc[-1] / idx_win["Close"].iloc[0] - 1
    
    # ==========================================
    # 2. 核心修复：使用对数收益率计算均值
    # ==========================================
    # 修复了之前算术平均导致的正负抵消问题
    if "hsi_log_ret_1d" in idx_win.columns:
        out["hsi_mean_1d_log_ret"] = idx_win["hsi_log_ret_1d"].mean()
    else:
        out["hsi_mean_1d_ret"] = idx_win["hsi_ret_1d"].mean() # 仅作兼容保留

    # ==========================================
    # 3. 波动率与风险调整收益 (Sharpe)
    # ==========================================
    daily_std = idx_win["hsi_ret_1d"].std()
    out["hsi_std_1d_ret"] = daily_std
    out["hsi_realized_vol"] = daily_std * (252 ** 0.5)

    # 衍生特征：夏普比率 (年化收益 / 年化波动，假设无风险利率为0)
    # 这对预测模型来说，比单纯的收益率或波动率更有效
    if out["hsi_realized_vol"] > 0 and "hsi_mean_1d_log_ret" in out:
        # 对数收益率年化只需乘以 252
        annualized_ret = out["hsi_mean_1d_log_ret"] * 252 
        out["hsi_sharpe"] = annualized_ret / out["hsi_realized_vol"]
    else:
        out["hsi_sharpe"] = 0.0

    # ==========================================
    # 4. 趋势、站上均线比例与乖离率 (Bias)
    # ==========================================
    for win in [20, 50, 100]:
        # 站上均线的时间占比
        flag_col = f"hsi_above_ma_{win}"
        if flag_col in idx_win.columns:
            out[f"hsi_frac_above_ma_{win}"] = idx_win[flag_col].mean()
            
        # 吸收新特征：窗口期的平均乖离率 (反映该区间整体是超买还是超卖)
        bias_col = f"hsi_bias_{win}"
        if bias_col in idx_win.columns:
            out[f"hsi_mean_bias_{win}"] = idx_win[bias_col].mean()

    # ==========================================
    # 5. VHSI 恐慌指数特征
    # ==========================================
    out["vhsi_mean"] = idx_win["Close_vhsi"].mean() / 100.0
    out["vhsi_std"] = idx_win["Close_vhsi"].std() / 100.0
    
    # 吸收新逻辑：窗口期内 VHSI 的净变化点数 (恐慌情绪是加剧还是缓解了)
    out["vhsi_net_change"] = idx_win["Close_vhsi"].iloc[-1] - idx_win["Close_vhsi"].iloc[0]
    
    out["vhsi_high_vol_regime_frac"] = idx_win["vhsi_high_vol_regime"].mean()

    return out


def summarize_option_window(opt_win: pd.DataFrame) -> dict:
    """
    Summarize daily option shape features over the lookback window.

    Expected input: one row per Date, with columns like
    - level_near, slope_near, curvature_near
    - level_next, slope_next, curvature_next
    - level_far, slope_far, curvature_far
    - term_level_next_near, term_slope_next_near, term_curvature_next_near
    - term_level_far_near, term_slope_far_near, term_curvature_far_near
    """
    out = {}
    if opt_win is None or opt_win.empty:
        return out

    feature_cols = [
        "level_near", "slope_near", "curvature_near",
        "level_next", "slope_next", "curvature_next",
        "level_far", "slope_far", "curvature_far",
        "term_level_next_near", "term_slope_next_near", "term_curvature_next_near",
        "term_level_far_near", "term_slope_far_near", "term_curvature_far_near",
    ]

    for col in feature_cols:
        if col not in opt_win.columns:
            continue

        s = opt_win[col].dropna()
        if s.empty:
            continue

        out[f"{col}_mean"] = s.mean()
        out[f"{col}_std"] = s.std()
        out[f"{col}_last"] = s.iloc[-1]

        # 20-trading-day change
        if len(s) >= 21:
            out[f"{col}_chg_20d"] = s.iloc[-1] - s.iloc[-21]
        else:
            out[f"{col}_chg_20d"] = np.nan

    # optional: quality diagnostics
    if "level_near" in opt_win.columns:
        out["option_obs_count"] = len(opt_win)

    return out


def build_periodic_features(index_features: pd.DataFrame,
                            option_features: Optional[pd.DataFrame] = None,
                            lookback: int = LOOKBACK_LEN,
                            pred_len: int = PRED_LEN) -> pd.DataFrame:
    """
    Build one row per decision date.
    - lookback: months of history used for features (e.g. 3)
    - pred_len: months ahead for prediction horizon (e.g. 1)
    """
    idx = index_features.sort_values("Date").copy()
    opt = option_features.sort_values("Date").copy()

    # use last trading day of each month as decision date
    month_ends = (
        idx.groupby(idx["Date"].dt.to_period("M"))["Date"]
           .max()
           .sort_values()
           .tolist()
    )

    rows = []
    for d in month_ends:
        d = pd.to_datetime(d)
        lookback_start = d - pd.DateOffset(months=lookback)

        idx_win = idx[(idx["Date"] > lookback_start) & (idx["Date"] <= d)]
        opt_win = opt[(opt["Date"] > lookback_start) & (opt["Date"] <= d)]

        if len(idx_win) < 20:
            continue  # not enough history

        idx_summary = summarize_index_window(idx_win)
        opt_summary = summarize_option_window(opt_win)

        row = {
            "decision_date": d,
            "lookback_months": lookback,
            "pred_len_months": pred_len,
        }
        row.update(idx_summary)
        row.update(opt_summary)
        rows.append(row)

    features = pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)
    return features