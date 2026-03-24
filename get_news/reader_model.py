# File: src/loansys_llm_cro/models/reader_model.py

import asyncio
import datetime
import json
import os
import re
import hashlib
from typing import List, Dict, Any, Optional

from dateutil.relativedelta import relativedelta

from get_news.urls import *
from get_news.news_search import harvest_news
from get_news.keyword_model import build_topic_queries

from utils.utils import *
from utils.chat_completion import achat_complete, READER_MODEL
from db.operations import save_news_one
from db.init import get_connection
from utils.utils import TABLE_NEWS


# ============================================================
# Config
# ============================================================

# reader 强制走 deepseek api，不再用本地 ollama
# READER_MODEL = os.getenv("READER_REMOTE_MODEL", "deepseek-reasoner")

# 每次送给模型的候选上限，避免 prompt 太大
READER_CANDIDATE_LIMIT = int(os.getenv("READER_CANDIDATE_LIMIT", "25"))

# 让模型最多挑多少篇
READER_TOP_K = int(os.getenv("READER_TOP_K", "20"))


# ============================================================
# Prompt
# ============================================================

READER_SYSTEM = """
You are a disciplined financial news selector for the Hang Seng Index (HSI) options market.

Your task:
From a batch of candidate news items, choose the TOP most useful items for forecasting the HSI options market.

Classify each selected item into exactly one predefined news_type:
- macro: macro driver such as China fiscal or monetary policy, PBOC, Fed spillover, tariffs, trade tensions, sanctions, export controls, geopolitics, growth shock, or property policy.
- spillover: transmission into Hong Kong equities through HKD, HIBOR, Stock Connect, southbound flows, HKEX activity, cross-border liquidity, or clear China-to-Hong Kong market spillover.
- index_direct: direct market confirmation through HSI direction, sector leadership, heavyweight movers, VHSI, implied or realized volatility, options sentiment, skew, hedging demand, or risk regime.

Selection principles:
- Prefer causal and forward-looking information over backward-looking market recap.
- Prefer articles useful for forecasting HSI direction, volatility, liquidity, or options regime.
- Prefer Hong Kong / China market transmission over generic world macro commentary.
- Prefer market-wide or index-relevant impact over isolated stories.
- Across the selected set, try to cover macro + spillover + index_direct whenever the candidate pool allows it.
- Avoid generic market close summaries, descriptive recap, lifestyle, entertainment, sports, crime, and non-market local stories.

Return JSON only with EXACT schema:
{
  "selected": [
    {"id": 1, "news_type": "macro"},
    {"id": 2, "news_type": "spillover"},
    {"id": 3, "news_type": "index_direct"}
  ]
}
"""


# ============================================================
# Helpers
# ============================================================

def url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def safe_json_extract(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}

    raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    s, e = raw.find("{"), raw.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(raw[s:e + 1])
        except Exception:
            pass

    return {}


def parse_selected_ids(js: Dict[str, Any], raw: str) -> List[int]:
    ids = js.get("selected_ids")
    if isinstance(ids, list):
        out = []
        for x in ids:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out

    # fallback: regex extract
    m = re.search(r'"selected_ids"\s*:\s*\[([^\]]*)\]', raw, flags=re.I | re.S)
    if m:
        nums = re.findall(r'\d+', m.group(1))
        return [int(x) for x in nums]

    # fallback: whole raw
    nums = re.findall(r'\b\d+\b', raw)
    return [int(x) for x in nums[:READER_TOP_K]]


def dedup_by_url_hash(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        source_url = (it.get("origin_url") or it.get("url") or "").strip()
        if not source_url:
            continue

        try:
            canon_url = canonicalize_url(source_url) or source_url
        except Exception:
            canon_url = source_url

        chash = url_hash(canon_url)
        if chash in seen:
            continue
        seen.add(chash)

        it = dict(it)
        it["_canon_url"] = canon_url
        it["_url_hash"] = chash
        out.append(it)
    return out


# ============================================================
# Prompt builder
# ============================================================

def build_batch_selection_user(
    items: List[Dict[str, Any]],
    keywords: List[str],
    *,
    topic: str = "Hang Seng Index",
    country_code: str = "HK",
) -> str:
    keyword_text = ", ".join([k for k in keywords if k][:30]) or "(none)"

    header = [
        f"Topic: {topic}",
        f"Country code: {country_code}",
        "Market: Hong Kong",
        f"Keywords List: {keyword_text}",
        f"Please select the top {READER_TOP_K} most relevant news items.",
        "",
        "Candidate news items:",
        ""
    ]

    blocks = []
    for idx, item in enumerate(items, start=1):
        title = trim_to_chars((item.get("title") or "").strip(), 300)
        abstract = trim_to_chars((item.get("abstract") or "").strip(), 600)
        approx_date = item.get("approx_date") or "(none)"
        source_api = item.get("source_api") or item.get("source") or "rss"
        query = (item.get("query") or "").strip() or "(none)"

        block = (
            f"[{idx}]\n"
            f"Date: {approx_date}\n"
            f"Source API: {source_api}\n"
            f"Query: {query}\n"
            f"Title: {title or '(none)'}\n"
            f"Abstract: {abstract or '(none)'}\n"
        )
        blocks.append(block)

    footer = "\nReturn JSON only."

    return "\n".join(header) + "\n".join(blocks) + footer


# ============================================================
# LLM batch selector
# ============================================================

async def select_top_articles_with_llm(
    items: List[Dict[str, Any]],
    keywords: List[str],
    *,
    topic: str,
    country_code: str,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    user = build_batch_selection_user(
        items=items,
        keywords=keywords,
        topic=topic,
        country_code=country_code,
    )

    try:
        raw = await achat_complete(
            READER_MODEL,
            READER_SYSTEM,
            user,
        )
    except Exception as e:
        print(f"[reader][batch-select-error] model={READER_MODEL} err={repr(e)}")
        return []

    try:
        js = safe_json_extract(raw)
        selected_ids = parse_selected_ids(js, raw)

        if not selected_ids:
            print("[reader][batch-select-empty] no ids parsed")
            return []

        # 保留顺序、去重、过滤越界
        seen = set()
        picked = []
        for i in selected_ids:
            if i in seen:
                continue
            seen.add(i)
            if 1 <= i <= len(items):
                picked.append(items[i - 1])

        return picked[:READER_TOP_K]

    except Exception as e:
        print(f"[reader][batch-select-parse-failed] err={repr(e)}")
        return []


# ============================================================
# Main pipeline
# ============================================================

async def refine_for_market(
    country_code: str,
    *,
    anchor_date: Optional[datetime.date] = None,
    lookback_months: int = 3,
    kwq_bundle: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    sd = anchor_date - relativedelta(months=lookback_months) if anchor_date else None
    ed = anchor_date

    rss = (kwq_bundle or {}).get("rss", {}) or {}
    keywords: List[str] = [str(x).strip() for x in (rss.get("keywords") or []) if str(x).strip()]
    queries_by_type = rss.get("queries_by_type") or {}

    # topic 可以继续保留；如果 build_topic_queries 还想复用，就从 bundle 里的 keywords 生成
    if keywords:
        try:
            topic, _ = build_topic_queries(keywords, [])
        except Exception:
            topic = "HSI options market news"
    else:
        topic = "HSI options market news"

    known_url_hashes: set[str] = set()
    try:
        conn = get_connection(include_database=True)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT url_hash FROM {TABLE_NEWS};")
                for (db_url_hash,) in cur.fetchall():
                    if db_url_hash:
                        known_url_hashes.add(str(db_url_hash))
        finally:
            conn.close()
    except Exception as e:
        print(f"[reader][known-hash-load-failed] {e}")

    try:
        target_recent = 24
        stop_after_valid = 30
        type_targets = {
            "macro": 8,
            "spillover": 8,
            "index_direct": 8,
        }
        locale_soft_targets = {
            "en_hk": 10,
            "zh_cn": 7,
            "zh_hk": 7,
        }
        raw_rss = harvest_news(
            kwq_bundle=kwq_bundle or {},
            target_recent=target_recent,
            max_queries=None,             
            anchor_date=anchor_date,
            lookback_months=lookback_months,
            stop_after_valid=stop_after_valid,
            type_targets=type_targets,
            locale_soft_targets=locale_soft_targets,
            verbose=True,
            # 不要再传 language="en"
            # 不要再强制单一 country_code/language 模式
        )
    except Exception as e:
        print(f"[refine][RSS-error] {e}")
        return []

    # 1) 日期过滤
    items_sorted = sorted(raw_rss, key=lambda x: str(x.get("approx_date")), reverse=True)
    pool = [it for it in items_sorted if in_range(it.get("approx_date"), sd, ed)]
    if not pool:
        print(f"[reader] no items in date window {sd}→{ed}; skipping.")
        return []

    # 2) 去重
    pool = dedup_by_url_hash(pool)

    # 3) 去掉数据库里已有的，优先让模型看新的
    new_pool = [it for it in pool if it.get("_url_hash") not in known_url_hashes]
    old_pool = [it for it in pool if it.get("_url_hash") in known_url_hashes]

    candidate_pool = (new_pool + old_pool)[:READER_CANDIDATE_LIMIT]

    if not candidate_pool:
        print("[reader] candidate pool is empty after dedup/filter.")
        return []

    print(
        f"[reader] batch-select candidates={len(candidate_pool)} "
        f"(new={len(new_pool)}, old={len(old_pool)}, top_k={READER_TOP_K})"
    )

    selected = await select_top_articles_with_llm(
        items=candidate_pool,
        keywords=keywords,   # 现在从 bundle 来
        topic=topic,
        country_code=country_code or "HK",
    )

    if not selected:
        print("[reader] LLM selected no articles.")
        return []

    valid_this_run: List[Dict[str, Any]] = []

    conn = None
    try:
        conn = get_connection()

        for item in selected:
            canon_url = item.get("_canon_url") or (item.get("origin_url") or item.get("url") or "").strip()
            chash = item.get("_url_hash") or (url_hash(canon_url) if canon_url else "")

            if not canon_url or not chash:
                continue

            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            approx_date = item.get("approx_date") or None
            source = publisher_from_url(canon_url) or item.get("source_api") or item.get("source") or "rss"

            # 优先记录具体 query；没有的话退到 news_type；再不行退到 bundle keywords
            keyword = (
                (item.get("query") or "").strip()
                or (item.get("news_type") or "").strip()
                or ", ".join(keywords[:10])
            )

            row = {
                "title": title,
                "url": canon_url,
                "url_hash": chash,
                "source": source,
                "approx_date": approx_date,
                "keyword": keyword,
                "summary": abstract or title,
                "content": abstract or title,
                "relevance_score": None,
                "is_new": (chash not in known_url_hashes),

                # 这些字段如果你表里暂时没有，就先别存 DB，但保留在返回值里很有用
                "lang": item.get("lang"),
                "news_type": item.get("news_type"),
                "query_lang": item.get("query_lang"),
                "search_language": item.get("search_language"),
                "search_country_code": item.get("search_country_code"),
                "search_locale": item.get("search_locale"),
            }

            try:
                save_news_one(conn, row)
                conn.commit()
                known_url_hashes.add(chash)
            except Exception as e:
                conn.rollback()
                print(f"[reader][db-save-failed] {canon_url} | err={e}")
                continue

            valid_this_run.append(row)

            tag = "added-new" if row["is_new"] else "added-known"
            print(
                f"[reader][{tag}] "
                f"type={row.get('news_type')} "
                f"lang={row.get('lang')} "
                f"{canon_url} | {title[:80]}"
            )

    finally:
        if conn is not None:
            conn.close()

    valid_this_run.sort(key=lambda x: str(x.get("approx_date")), reverse=True)
    return valid_this_run