import os
import re
import urllib.parse
import datetime as dt
from typing import List, Dict, Optional, Any
import pandas as pd
import math
import json
import numpy as np

GOOGLE_HL = os.getenv("GOOGLE_HL", "en")
GOOGLE_GL = os.getenv("GOOGLE_GL", "US")
GOOGLE_CEID = os.getenv("GOOGLE_CEID", "US:en")

TARGET_RECENT = 20
DATE_RANGE_START = dt.date(2024, 1, 1)
DATE_RANGE_END = dt.date(2026, 3, 1)

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.14"))
TOP_P       = float(os.getenv("TOP_P", "0.9"))   # kept if your backend honors it; not set explicitly in API call
SEED        = int(os.getenv("SEED", "7"))

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "600"))
MAX_CONN     = int(os.getenv("MAX_CONN", "8"))      # concurrency for fetch & reader stage
TARGET_RECENT = int(os.getenv("TARGET_RECENT", "20"))  # stop harvesting when this many recent items found
READER_RECENT_LIMIT = int(os.getenv("READER_RECENT_LIMIT", "12")) # max articles to read per market

HTML_TAG_RE   = re.compile(r"<[^>]+>")
SCRIPT_STYLE  = re.compile(r"(?is)<(script|style).*?>.*?</\1>") 
WS_RE         = re.compile(r"\s+")    
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; 3Model-News/1.0)")

TABLE_NEWS= "news"
TABLE_PERFORMANCE = "performance" 

API_KEY='sk-ef018c3ada414210851f7578ae15b15b'


def clean_html_to_text(html: str) -> str:
    if not html: return ""
    txt = SCRIPT_STYLE.sub(" ", html)
    txt = HTML_TAG_RE.sub(" ", txt)
    txt = WS_RE.sub(" ", txt)
    return txt.strip()

def trim_to_chars(s: str, limit: int = 600) -> str:
    if not s: return s
    n = 0.0
    out = []
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f':
            n += 1.0
        else:
            n += 0.5
        if n > limit:
            break
        out.append(ch)
    return "".join(out).strip()



def to_json_safe(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
        return obj.isoformat()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]

    return obj


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try to parse final_text into JSON.
    If parsing fails, store it as {"raw_text": text}.
    """
    if text is None:
        return {}

    text = str(text).strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return to_json_safe(parsed)
        return {"value": to_json_safe(parsed)}
    except Exception:
        pass

    import re

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                return to_json_safe(parsed)
            return {"value": to_json_safe(parsed)}
        except Exception:
            pass

    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                return to_json_safe(parsed)
            return {"value": to_json_safe(parsed)}
        except Exception:
            pass

    return {"raw_text": text}


def _to_ts(x):
    return pd.to_datetime(x).normalize()


def get_first_trading_day_each_month(option_df: pd.DataFrame,
                                     start_date,
                                     end_date) -> List[pd.Timestamp]:
    df = option_df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    start_date = _to_ts(start_date)
    end_date = _to_ts(end_date)

    dates = sorted(df.loc[(df['date'] >= start_date) & (df['date'] <= end_date), 'date'].unique())
    if not dates:
        return []

    dates = pd.Series(pd.to_datetime(dates))
    month_firsts = dates.groupby(dates.dt.to_period("M")).min().tolist()
    return [pd.Timestamp(x).normalize() for x in month_firsts]


def get_last_trading_day_before(option_df: pd.DataFrame, dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    df = option_df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    s = df.loc[df['date'] < dt, 'date']
    if s.empty:
        return None
    return s.max()


def get_last_trading_day_in_month(option_df: pd.DataFrame, month_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    df = option_df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    target_period = month_ts.to_period("M")
    s = df.loc[df['date'].dt.to_period("M") == target_period, 'date']
    if s.empty:
        return None
    return s.max()