# File: src/loansys_llm_cro/news_search.py
import re
import datetime
from typing import List, Dict, Any, Optional
import urllib.parse
from dateutil.relativedelta import relativedelta
import httpx
import feedparser
from email.utils import parsedate_to_datetime
from get_news.fetch_fulltext import get_google_params, get_origin_url
from utils.utils import *
from get_news.urls import *



# ============================================================
# Helpers
# ============================================================

def _to_iso_date(published: str, published_parsed=None) -> Optional[str]:
    """
    Convert RSS published fields into YYYY-MM-DD if possible.
    """
    if published_parsed:
        try:
            y, m, d = published_parsed.tm_year, published_parsed.tm_mon, published_parsed.tm_mday
            return datetime.date(y, m, d).isoformat()
        except Exception:
            pass

    if published:
        try:
            dt = parsedate_to_datetime(published)
            return dt.date().isoformat()
        except Exception:
            return None
    return None



def _safe_decode_google_news_url(rss_url: str) -> Optional[str]:
    """
    Decode a Google News RSS redirect URL into the original publisher URL.
    Returns None if decoding fails.
    """
    try:
        source, sign, ts = get_google_params(rss_url)
        return get_origin_url(source, sign, ts)
    except Exception as e:
        print(f"[decode-origin-url-error] {e} | rss_url={rss_url}")
        return None


def _append_if_in_range(
    it: Dict[str, Any],
    buf: List[Dict[str, Any]],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> None:
    approx = it.get("approx_date")
    if not in_range(approx, start_date, end_date):
        return

    row = {
        "title": it.get("title", ""),
        "url": it.get("url", ""),
        "origin_url": it.get("origin_url"),
        "abstract": it.get("abstract", ""),
        "approx_date": approx,
        "source": it.get("source", "google_rss"),
    }

    for k in [
        "query",
        "lang",
        "news_type",
        "query_lang",
        "search_language",
        "search_country_code",
        "search_locale",
    ]:
        if it.get(k) is not None:
            row[k] = it.get(k)

    buf.append(row)
    
    

# ============================================================
# Google News RSS
# ============================================================

def google_news_rss(
    q: str,
    when: Optional[str] = None,
    *,
    country_code: str = "",
    language: Optional[str] = None,
    query_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch Google News RSS results for a query.
    """
    base = "https://news.google.com/rss/search"

    # Use full locale strings for Google News.
    lang = (language or GOOGLE_HL or "en-US").strip()

    # Infer market if not explicitly provided
    gl = (country_code or GOOGLE_GL or "US").strip().upper()

    # ceid usually uses country:language-root, e.g. US:en / HK:zh-Hant / HK:en
    lang_root = lang.split("-")[0].lower()
    ceid = f"{gl}:{lang_root}"

    q_full = q.strip()
    if when:
        q_full = f"{q_full} when:{when}"

    params = {
        "q": q_full,
        "hl": lang,
        "gl": gl,
        "ceid": ceid,
    }
    url = base + "?" + urllib.parse.urlencode(params)

    try:
        with httpx.Client(
            timeout=15.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": lang,
            },
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
        fp = feedparser.parse(resp.text)
    except Exception as e:
        print(f"[rss-fetch-error] {e} on {url}")
        return []

    items: List[Dict[str, Any]] = []
    lang_tag = "zh" if lang.lower().startswith("zh") else "en"

    for ent in fp.entries:
        link = ent.get("link") or ""
        title = (ent.get("title") or "").strip()
        summary = (ent.get("summary") or "").strip()

        approx = (
            _to_iso_date(ent.get("published"), ent.get("published_parsed"))
            or infer_date_from_url_or_text(link, summary or title)
        )

        items.append({
            "title": title,
            "url": link,
            "origin_url": _safe_decode_google_news_url(link),
            "abstract": re.sub(r"<[^>]+>", " ", summary).strip(),
            "approx_date": approx if approx and re.match(r"^\d{4}-\d{2}-\d{2}$", str(approx)) else None,
            "source": "rss",
            "lang": lang_tag,
            "query": query_label or q,
        })

    return items


# ============================================================
# Public API
# ============================================================

def harvest_news(
    kwq_bundle: Dict[str, Any],
    target_recent: int = 24,
    max_queries: Optional[int] = None,
    anchor_date: Optional[datetime.date] = None,
    lookback_months: int = 1,
    *,
    search_locales: Optional[List[Dict[str, str]]] = None,
    stop_after_valid: Optional[int] = None,
    type_targets: Optional[Dict[str, int]] = None,
    locale_soft_targets: Optional[Dict[str, int]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Balanced RSS harvesting:
    - hard quotas by news_type
    - soft quotas by search_locale
    - supports query_plan from build_hsi_kwq_bundle()
    """

    mode = (kwq_bundle or {}).get("rss", {}) or {}
    query_plan: List[Dict[str, Any]] = list(mode.get("query_plan") or [])

    # backward compatibility: if query_plan absent, fallback to flat queries
    if not query_plan:
        flat_queries = [
            str(q).strip() for q in (mode.get("queries") or []) if str(q).strip()
        ]
        query_plan = []
        for q in flat_queries:
            query_plan.append({
                "query": q,
                "news_type": None,
                "query_lang": None,
                "locales": search_locales or [
                    {"name": "en_hk", "language": "en-US", "country_code": "HK"},
                    {"name": "zh_cn", "language": "zh-CN", "country_code": "CN"},
                    {"name": "zh_hk", "language": "zh-HK", "country_code": "HK"},
                ],
            })

    if not query_plan:
        print("[harvest][rss] empty query_plan")
        return []

    if max_queries is not None:
        query_plan = query_plan[:max_queries]

    if anchor_date is None:
        anchor_date = datetime.date.today()

    effective_end = anchor_date
    effective_start = anchor_date - relativedelta(months=lookback_months)

    if stop_after_valid is None:
        stop_after_valid = max(target_recent, 24)

    if type_targets is None:
        base = max(target_recent, 24)
        third = base // 3
        type_targets = {
            "macro": third,
            "spillover": third,
            "index_direct": base - 2 * third,
        }

    if locale_soft_targets is None:
        # soft caps; not strict stopping conditions
        locale_soft_targets = {
            "en_hk": max(8, target_recent // 2),
            "zh_cn": max(6, target_recent // 3),
            "zh_hk": max(6, target_recent // 3),
        }

    def _type_target_met(type_counts: Dict[str, int], type_targets: Dict[str, int]) -> bool:
        return all(type_counts.get(k, 0) >= v for k, v in type_targets.items())

    def _is_locale_soft_full(locale_name: str, locale_counts: Dict[str, int]) -> bool:
        if not locale_name or locale_name not in locale_soft_targets:
            return False
        return locale_counts.get(locale_name, 0) >= locale_soft_targets[locale_name]

    if verbose:
        print("\n" + "=" * 90)
        print("[harvest][rss] START harvest_news")
        print(f"[harvest][rss] query plans        : {len(query_plan)}")
        print(f"[harvest][rss] target_recent      : {target_recent}")
        print(f"[harvest][rss] stop_after_valid   : {stop_after_valid}")
        print(f"[harvest][rss] effective window   : [{effective_start} ~ {effective_end}]")
        print(f"[harvest][rss] type_targets       : {type_targets}")
        print(f"[harvest][rss] locale_soft_targets: {locale_soft_targets}")
        print("=" * 90)

    harvested: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {k: 0 for k in type_targets}
    locale_counts: Dict[str, int] = {k: 0 for k in locale_soft_targets}

    def _add_date_range_to_query(
        q: str,
        start_date: Optional[datetime.date],
        end_date: Optional[datetime.date],
    ) -> str:
        q2 = q.strip()
        if start_date:
            q2 += f" after:{start_date.isoformat()}"
        if end_date:
            q2 += f" before:{end_date.isoformat()}"
        return q2

    # phase 1: respect locale soft caps
    # phase 2: relax locale soft caps if still not enough by type
    for phase in [1, 2]:
        if verbose:
            print(f"\n[harvest][rss] ===== phase {phase} =====")

        for i, plan in enumerate(query_plan, start=1):
            q = str(plan.get("query", "")).strip()
            news_type = plan.get("news_type")
            query_lang = plan.get("query_lang")
            locales = plan.get("locales") or []

            if not q:
                continue

            # hard type quota: skip this query entirely if this type is already full
            if news_type in type_targets and type_counts.get(news_type, 0) >= type_targets[news_type]:
                if verbose:
                    print(
                        f"\n[harvest][rss] skip query {i}/{len(query_plan)} "
                        f"| type={news_type} already full | q={q}"
                    )
                continue

            if verbose:
                print(f"\n[harvest][rss] query {i}/{len(query_plan)}")
                print(f"[harvest][rss] q = {q}")
                print(f"[harvest][rss] news_type={news_type} | query_lang={query_lang}")

            q2 = _add_date_range_to_query(q, effective_start, effective_end)

            for loc in locales:
                loc_name = loc.get("name", "")
                loc_lang = loc.get("language")
                loc_cc = loc.get("country_code", "")

                # phase 1: obey locale soft caps
                if phase == 1 and _is_locale_soft_full(loc_name, locale_counts):
                    if verbose:
                        print(
                            f"[harvest][rss]   locale={loc_name} skipped in phase1 "
                            f"(soft cap reached)"
                        )
                    continue

                try:
                    if verbose:
                        print(
                            f"[harvest][rss]   locale={loc_name} "
                            f"lang={loc_lang} country={loc_cc}"
                        )

                    batch = google_news_rss(
                        q=q2,
                        country_code=loc_cc,
                        language=loc_lang,
                        query_label=q,
                    ) or []

                    if verbose:
                        print(f"[harvest][rss]   -> raw batch size: {len(batch)}")

                    kept_this_locale = 0

                    for j, it in enumerate(batch, start=1):
                        # enrich metadata before filtering
                        it["query"] = q
                        it["news_type"] = news_type
                        it["query_lang"] = query_lang
                        it["search_language"] = loc_lang
                        it["search_country_code"] = loc_cc
                        it["search_locale"] = loc_name

                        # hard type quota again at item level
                        if news_type in type_targets and type_counts.get(news_type, 0) >= type_targets[news_type]:
                            if verbose:
                                print(
                                    f"    [=] skip #{j}: type quota full "
                                    f"| type={news_type} | locale={loc_name} "
                                    f"| title={(it.get('title') or '')[:120]}"
                                )
                            continue

                        # phase 1 locale soft cap at item level
                        if phase == 1 and _is_locale_soft_full(loc_name, locale_counts):
                            if verbose:
                                print(
                                    f"    [=] skip #{j}: locale soft cap full "
                                    f"| locale={loc_name} | title={(it.get('title') or '')[:120]}"
                                )
                            continue

                        old_len = len(harvested)
                        _append_if_in_range(it, harvested, effective_start, effective_end)

                        if len(harvested) > old_len:
                            kept_this_locale += 1

                            if news_type in type_targets:
                                type_counts[news_type] = type_counts.get(news_type, 0) + 1
                            if loc_name in locale_soft_targets:
                                locale_counts[loc_name] = locale_counts.get(loc_name, 0) + 1

                            if verbose:
                                print(
                                    f"    [+] kept #{j}: date={it.get('approx_date')} "
                                    f"| type={news_type} | locale={loc_name} "
                                    f"| type_counts={type_counts} | locale_counts={locale_counts} "
                                    f"| title={(it.get('title') or '')[:120]}"
                                )
                        else:
                            if verbose:
                                print(
                                    f"    [-] drop #{j}: date={it.get('approx_date')} "
                                    f"| type={news_type} | locale={loc_name} "
                                    f"| title={(it.get('title') or '')[:120]}"
                                )

                    harvested = dedup_items(harvested)

                    recent = [
                        it for it in harvested
                        if in_range(it.get("approx_date"), effective_start, effective_end)
                    ]

                    if verbose:
                        print(
                            f"[harvest][rss]   locale done | kept_this_locale={kept_this_locale} | "
                            f"dedup_total={len(harvested)} | recent_in_range={len(recent)}"
                        )

                    # preferred stop: balanced type targets met
                    if _type_target_met(type_counts, type_targets):
                        if verbose:
                            print(
                                f"[harvest][rss] reached balanced type targets: {type_counts}"
                            )
                        break

                    # hard global stop
                    if len(recent) >= stop_after_valid:
                        if verbose:
                            print(
                                f"[harvest][rss] reached hard stop_after_valid={stop_after_valid}"
                            )
                        break

                except Exception as e:
                    print(f"[harvest][rss] err {e} | q={q} | locale={loc_name}")

            recent = [
                it for it in harvested
                if in_range(it.get("approx_date"), effective_start, effective_end)
            ]

            if _type_target_met(type_counts, type_targets):
                break

            if len(recent) >= stop_after_valid:
                break

        recent = [
            it for it in harvested
            if in_range(it.get("approx_date"), effective_start, effective_end)
        ]

        if _type_target_met(type_counts, type_targets):
            break

        if len(recent) >= stop_after_valid:
            break

    recent = [
        it for it in harvested
        if in_range(it.get("approx_date"), effective_start, effective_end)
    ]

    if verbose:
        print("\n" + "-" * 90)
        print(f"[harvest][rss] final recent count : {len(recent)}")
        print(f"[harvest][rss] final type_counts  : {type_counts}")
        print(f"[harvest][rss] final locale_counts: {locale_counts}")
        print("-" * 90)

    return recent[:target_recent]



def build_fulltext_jobs(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience helper:
    prepare items for fetch_fulltext.read(...)

    Output:
    [
      {
        "title": ...,
        "origin_url": ...,
        "approx_date": ...,
        ...
      }
    ]
    """
    jobs = []
    for it in news_items or []:
        origin_url = it.get("origin_url")
        if not origin_url:
            continue
        jobs.append(it)
    return jobs