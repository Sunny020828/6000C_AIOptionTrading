from db.operations import fetch_news
import asyncio
from typing import List, Dict, Any, Optional
from get_news.fetch_fulltext import read
import httpx

async def enrich_news_with_fulltext(
    news_list: List[Dict[str, Any]],
    max_chars: int = 300,
    max_concurrency: int = 1,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(item):
        url = (item.get("url") or "").strip()

        try:
            fulltext = await asyncio.to_thread(read, url) if url else ""
        except httpx.HTTPStatusError as e:
            status = getattr(e.response, "status_code", None)
            print(f"[news-context][http-error] status={status} url={url}")
            fulltext = ""
        except Exception as e:
            print(f"[news-context][fulltext-failed] url={url} err={e}")
            fulltext = ""

        if not fulltext:
            fulltext = (
                (item.get("summary") or "").strip()
                or (item.get("content") or "").strip()
                or (item.get("title") or "").strip()
            )

        item["fulltext_300"] = fulltext[:300]
        return item

    tasks = [asyncio.create_task(_one(x)) for x in news_list]
    return await asyncio.gather(*tasks)

def build_news_context_block(
    news_list: List[Dict[str, Any]],
    *,
    max_items: Optional[int] = None,
) -> str:
    items = news_list[:max_items] if max_items is not None else news_list

    lines = []
    lines.append("## News Context")
    lines.append("The following articles were pre-filtered as relevant market news.")
    lines.append("Use them only as supplementary context for interpreting metrics, regime, and decision rationale.")
    lines.append("")

    for _, item in enumerate(items, start=1):
        title = (item.get("title") or "").strip()
        source = (item.get("source") or "").strip()
        published_at = item.get("published_at") or item.get("approx_date") or ""
        fulltext_300 = (item.get("fulltext_300") or "").strip()
        url = (item.get("url") or "").strip()

        lines.append(f"Title: {title or '(none)'}")
        lines.append(f"Date: {published_at or '(none)'}")
        lines.append(f"Source: {source or '(none)'}")
        lines.append(f"Excerpt: {fulltext_300 or '(none)'}")
        if url:
            lines.append(f"URL: {url}")
        lines.append("")

    return "\n".join(lines).strip()

async def build_news_context_from_db(
    start_date,
    end_date,
    *,
    keyword: Optional[str] = None,
    limit: int = 10,
    fulltext_chars: int = 300,
    max_concurrency: int = 5,
) -> str:
    news_list = fetch_news(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        keyword=keyword,
    )
    print(len(news_list))
    if not news_list:
        return "## News Context\nNo relevant news articles were found in the database."

    enriched = await enrich_news_with_fulltext(
        news_list,
        max_chars=fulltext_chars,
        max_concurrency=max_concurrency,
    )

    return build_news_context_block(enriched)

