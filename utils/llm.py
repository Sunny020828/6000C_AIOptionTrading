
import json
import re
import pandas as pd
from openai import OpenAI
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from utils.data import LOOKBACK_LEN, PRED_LEN
from utils.utils import API_KEY

def build_regime_prompt(
    features_df: pd.DataFrame,
    as_of_date,
    news_context: str = "",
    lookback: int = LOOKBACK_LEN,
    pred_len: int = PRED_LEN,
    ):
    as_of_date = pd.to_datetime(as_of_date)
    df = features_df.copy()
    
    # 筛选对应的 lookback 和 pred_len 窗口
    if lookback is not None:
        df = df[df["lookback_months"] == lookback]
    if pred_len is not None:
        df = df[df["pred_len_months"] == pred_len]

    df = df[df["decision_date"] <= as_of_date].sort_values("decision_date")
    if df.empty:
        raise ValueError(f"No feature row found on or before {as_of_date}")

    row = df.iloc[-1]
    
    # --- 核心数据预处理：将极小值转化为 LLM 可理解的百分比/年化值 ---
    # 之前困扰你的 1d_ret 均值，现在建议展示年化对数收益率
    # 假设原本是 0.0004 -> 展示为 10.08% (0.1008)
    ann_log_ret = row.get('hsi_mean_1d_log_ret', 0) * 252 
    
    system_msg = f"""You are a quantitative trader specializing in equity index regime analysis.

Task:
Estimate the probability distribution of market direction and volatility regimes for the NEXT {pred_len} month(s).

Decision Guidance (Quantitative Logic):
- HSI Sharpe Ratio: Values > 1.0 suggest strong risk-adjusted Bull; < -1.0 suggest strong Bear.
- Annualized Log Return: Reflects the theoretical compounded growth rate. Positive is Bull, Negative is Bear.
- Mean Bias (20d/50d): Reflects distance from moving averages. High positive bias (> 3-5%) might indicate overbought (Range/Bear risk), while high negative bias indicates oversold.
- VHSI Net Change: Positive values indicate rising fear, supporting High_Vol and Bear regimes.
- Option Surface: Negative 'slope' indicates downside hedging demand (Skew). High 'curvature' suggests tail-risk pricing.

Language Rules:
- Input market news context may contain both English and Chinese.
- You must interpret multilingual input consistently.
- Output must be in English only.
- Return JSON only, with English keys and English regime labels.
- Do not include any explanation outside the JSON.


Output Requirements:
- Output valid JSON only.
- Structure: {{"direction_probs": {{"bull":..., "bear":..., "range":...}}, "volatility_probs": {{"low_vol":..., "medium_vol":..., "high_vol":...}}}}
- Sum of each category must be exactly 1.0.
"""

    user_msg = f"""Decision date: {row['decision_date'].date()}
Forecast horizon: next {pred_len} month(s)
Lookback window: last {lookback} month(s).

Aggregated HSI Features:
- Last HSI Close: {row['hsi_last_close']:.2f}
- HSI Lookback Total Return: {row['hsi_lookback_ret']:.2%}
- HSI Annualized Log Return: {ann_log_ret:.2%} (Theoretical drift)
- HSI Sharpe Ratio: {row.get('hsi_sharpe', 0):.3f}
- HSI Realized Volatility (Ann.): {row['hsi_realized_vol']:.2%}
- Avg. Price Bias (20d MA): {row.get('hsi_mean_bias_20', 0):.2%}
- Fraction of days HSI > MA20/MA50/MA100: {row['hsi_frac_above_ma_20']:.1%}, {row['hsi_frac_above_ma_50']:.1%}, {row['hsi_frac_above_ma_100']:.1%}

Aggregated VHSI & Option Surface:
- VHSI Mean Level: {row['vhsi_mean']:.4f}
- VHSI Net Change in Window: {row.get('vhsi_net_change', 0):.2f} pts
- High-Vol Regime Fraction: {row['vhsi_high_vol_regime_frac']:.1%}
- IV Slope (Near-term Mean): {row.get('slope_near_mean', float('nan')):.4f}
- IV Curvature (Near-term Mean): {row.get('curvature_near_mean', float('nan')):.4f}
- IV Level 20d Change: {row.get('level_near_chg_20d', float('nan')):.4f}
"""

    if news_context and str(news_context).strip():
        user_msg += "\nThe news context may be bilingual Chinese and English. Interpret it normally, but output English JSON only."
        user_msg += f"\nSupplementary News Context:\n{news_context.strip()}\n"

    user_msg += "\nBased on the above, provide the JSON regime probability estimates."
    
    return system_msg, user_msg
  
  
def call_llm(features_df, context, as_of_date="2024-10-01"):
    system_msg, user_msg = build_regime_prompt(features_df, as_of_date=as_of_date, news_context=context)
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    model='deepseek-reasoner'
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        stream=False,
        temperature=0.14,
    )

    msg = response.choices[0].message

    reasoning = getattr(msg, "reasoning_content", None)
    final_text = msg.content

    # print("=== reasoning ===")
    # print(reasoning)

    # print("\n=== final answer ===")
    # print(final_text)
    return reasoning, final_text


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from LLM text.
    """
    text = (text or "").strip()

    # Try full text first
    try:
        return json.loads(text)
    except Exception:
        pass

    # fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # first {...} block
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    raise ValueError("Failed to parse JSON from LLM output.")


def normalize_llm_probs(obj: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Normalize keys and probabilities.
    """
    direction = obj.get("direction_probs", {}) or {}
    vol = obj.get("volatility_probs", {}) or {}

    dir_map = {
        "bull": "bull",
        "bear": "bear",
        "range": "range",
    }

    vol_map = {
        "low_vol": "low_vol",
        "low": "low_vol",
        "medium_vol": "medium_vol",
        "medium": "medium_vol",
        "median_vol": "medium_vol",
        "mid_vol": "medium_vol",
        "high_vol": "high_vol",
        "high": "high_vol",
    }

    d = {dir_map[k]: float(v) for k, v in direction.items() if k in dir_map}
    v = {vol_map[k]: float(v) for k, v in vol.items() if k in vol_map}

    for key in ["bull", "bear", "range"]:
        d.setdefault(key, 0.0)
    for key in ["low_vol", "medium_vol", "high_vol"]:
        v.setdefault(key, 0.0)

    d_sum = sum(d.values())
    v_sum = sum(v.values())

    if d_sum > 0:
        d = {k: val / d_sum for k, val in d.items()}
    if v_sum > 0:
        v = {k: val / v_sum for k, val in v.items()}

    return {
        "direction_probs": d,
        "volatility_probs": v
    }


def choose_scenario(final_text: str,
                    method: str = "joint") -> str:
    """
    Convert marginals into scenario key used by strategy_config.
    Important:
    - strategy pool uses 'medianvol'
    - LLM output uses 'medium_vol'
    """
    json=extract_json_from_text(final_text)
    prob_obj=normalize_llm_probs(json)
    d = prob_obj["direction_probs"]
    v = prob_obj["volatility_probs"]

    vol_name_map = {
        "low_vol": "lowvol",
        "medium_vol": "medianvol",
        "high_vol": "highvol",
    }

    if method == "argmax_pair":
        best_dir = max(d.items(), key=lambda x: x[1])[0]
        best_vol = max(v.items(), key=lambda x: x[1])[0]
        return f"{best_dir}_{vol_name_map[best_vol]}"

    # default: joint product
    candidates = []
    for dk, dv in d.items():
        for vk, vv in v.items():
            candidates.append((f"{dk}_{vol_name_map[vk]}", dv * vv))

    candidates = sorted(candidates, key=lambda x: (-x[1], x[0]))
    return candidates[0][0]