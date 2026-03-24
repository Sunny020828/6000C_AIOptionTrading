from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json 
from db.init import get_connection
from utils.utils import *
from utils.utils import _extract_json_from_text

def normalize_datetime(dt_value: Optional[Union[str, datetime]]) -> Optional[str]:
    if dt_value is None:
        return None

    if isinstance(dt_value, datetime):
        return dt_value.strftime("%Y-%m-%d %H:%M:%S")

    if not isinstance(dt_value, str):
        return None

    text = dt_value.strip()
    if not text:
        return None

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    return None


def save_news_one(conn, news: Dict[str, Any]) -> None:
    sql = f"""
    INSERT INTO {TABLE_NEWS} (
        title,
        url,
        url_hash,
        source,
        published_at,
        keyword,
        summary,
        content,
        relevance_score
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        title = VALUES(title),
        url = VALUES(url),
        source = VALUES(source),
        published_at = VALUES(published_at),
        keyword = VALUES(keyword),
        summary = VALUES(summary),
        content = VALUES(content),
        relevance_score = VALUES(relevance_score)
    """

    values = (
        news.get("title"),
        news.get("url"),
        news.get("url_hash"),
        news.get("source"),
        normalize_datetime(news.get("published_at") or news.get("approx_date")),
        news.get("keyword"),
        news.get("summary"),
        news.get("content"),
        news.get("relevance_score", 1.0),
    )

    cursor = conn.cursor()
    cursor.execute(sql, values)
    cursor.close()


def save_news(items: Union[Dict[str, Any], List[Dict[str, Any]]]) -> int:
    if isinstance(items, dict):
        items = [items]

    if not items:
        return 0

    conn = get_connection()
    affected = 0

    try:
        for item in items:
            if not item:
                continue
            save_news_one(conn, item)
            affected += 1

        conn.commit()
        print(f"[save_news] affected={affected}")
        return affected

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
        


def fetch_news(start_date: Union[str, datetime],
               end_date: Union[str, datetime],
               limit: Optional[int] = None,
               keyword: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    sql = f"SELECT * FROM {TABLE_NEWS} WHERE 1=1"
    params = []

    if start_date:
        sql += " AND published_at >= %s"
        params.append(normalize_datetime(start_date))

    if end_date:
        sql += " AND published_at <= %s"
        params.append(normalize_datetime(end_date))

    if keyword:
        sql += " AND keyword = %s"
        params.append(keyword)

    sql += " ORDER BY published_at DESC"

    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    cursor.execute(sql, params)
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return results



def save_performance_record(
    run_id: str,
    decision_date,
    final_text: Union[str, Dict[str, Any], None],
    performance_metrics: Union[Dict[str, Any], str, None],
    reasoning: Optional[str] = None,
    model: str = "deepseek-reasoner",
) -> int:
    """
    Insert one row into performance table.

    Parameters
    ----------
    run_id : str
    decision_date : str | datetime
        Monthly decision date. Will be written into prediction_json.
    final_text : str | dict | None
        Raw LLM output or parsed dict.
        prediction_json will be built inside this function.
    performance_metrics : dict | str | None
    reasoning : str | None
    model : str

    Returns
    -------
    int
        inserted row id
    """
    conn = get_connection()
    cur = None

    try:
        if final_text is None:
            return None

        if isinstance(final_text, dict):
            parsed = to_json_safe(final_text)
        elif isinstance(final_text, str):
            parsed = _extract_json_from_text(final_text)
        else:
            parsed = None

        if parsed is None:
            return None

        direction_probs = parsed.get("direction_probs") if isinstance(parsed, dict) else None
        volatility_probs = parsed.get("volatility_probs") if isinstance(parsed, dict) else None

        chosen_scenario = None
        if isinstance(parsed, dict):
            chosen_scenario = (
                parsed.get("chosen_scenario")
                or parsed.get("scenario")
            )

        payload = {
            "decision_date": str(pd.to_datetime(decision_date).date()) if decision_date is not None else None,
            "direction_probs": direction_probs,
            "volatility_probs": volatility_probs,
            "chosen_scenario": chosen_scenario,
            "raw_llm_json": parsed,
        }
        prediction_json=to_json_safe(payload)

        if isinstance(performance_metrics, dict):
            perf_json = to_json_safe(performance_metrics)
        elif isinstance(performance_metrics, str):
            perf_json = _extract_json_from_text(performance_metrics)
        else:
            perf_json = None

        sql = f"""
            INSERT INTO {TABLE_PERFORMANCE} (
                run_id,
                created_at,
                model,
                reasoning,
                prediction_json,
                performance_metrics
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        created_at = dt.datetime.now()

        cur = conn.cursor()
        cur.execute(
            sql,
            (
                run_id,
                created_at,
                model,
                reasoning,
                json.dumps(prediction_json, ensure_ascii=False) if prediction_json is not None else None,
                json.dumps(perf_json, ensure_ascii=False) if perf_json is not None else None,
            ),
        )
        inserted_id = cur.lastrowid
        conn.commit()
        return inserted_id

    except Exception:
        conn.rollback()
        raise

    finally:
        if cur is not None:
            cur.close()
        conn.close()
