
import re
import json
import os
import html
import urllib.parse
from urllib.parse import quote

import requests
import httpx
from lxml import etree

try:
    import trafilatura
    HAS_TRA = True
except Exception:
    HAS_TRA = False


# ============================================================
# Proxy config
# ============================================================
LOCAL_PROXY = os.getenv("GOOGLE_NEWS_PROXY", "").strip() or None
USE_PROXY = bool(LOCAL_PROXY)

PROXIES = {
    "http": LOCAL_PROXY,
    "https": LOCAL_PROXY,
} if USE_PROXY else None


def _requests_kwargs() -> dict:
    kwargs = {
        "timeout": 30,
    }
    if USE_PROXY:
        kwargs["proxies"] = PROXIES
    return kwargs


def _httpx_client_kwargs() -> dict:
    kwargs = {
        "timeout": 30,
        "trust_env": False,
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://news.google.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        "follow_redirects": True,
    }
    if USE_PROXY:
        # httpx new style
        kwargs["proxy"] = LOCAL_PROXY
    return kwargs


# ============================================================
# Google News redirect decode
# ============================================================

def get_google_params(url: str):
    with requests.Session() as s:
        s.trust_env = False
        response = s.get(url, **_requests_kwargs())
        response.raise_for_status()

    tree = etree.HTML(response.text)
    if tree is None:
        raise ValueError("Failed to parse Google News HTML")

    sign = tree.xpath('//*[@data-n-a-sg]/@data-n-a-sg')
    ts = tree.xpath('//*[@data-n-a-ts]/@data-n-a-ts')
    source = tree.xpath('//*[@data-n-a-id]/@data-n-a-id')

    if not sign or not ts or not source:
        raise ValueError("Failed to extract source/sign/ts from Google News page")

    return source[0], sign[0], ts[0]


def get_origin_url(source: str, sign: str, ts: str) -> str | None:
    """
    Construct a request using the extracted parameters and resolve
    the original article URL from Google News redirect.
    """
    url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"

    req_data = [[[
        "Fbv4je",
        f"[\"garturlreq\",[[\"zh-HK\",\"HK\",[\"FINANCE_TOP_INDICES\",\"WEB_TEST_1_0_0\"],null,null,1,1,\"HK:zh-Hant\",null,480,null,null,null,null,null,0,5],\"zh-HK\",\"HK\",1,[2,4,8],1,1,null,0,0,null,0],\"{source}\",{ts},\"{sign}\"]",
        None,
        "generic"
    ]]]

    payload = f"f.req={quote(json.dumps(req_data))}"

    headers = {
        "Host": "news.google.com",
        "X-Same-Domain": "1",
        "Accept-Language": "zh-CN",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Accept": "*/*",
        "Origin": "https://news.google.com",
        "Referer": "https://news.google.com/",
        "Accept-Encoding": "gzip, deflate, br",
    }

    response = requests.post(
        url,
        headers=headers,
        data=payload,
        **_requests_kwargs(),
    )
    response.raise_for_status()

    match = re.search(r'https?://[^\s",\\]+', response.text)
    return match.group() if match else None


# ============================================================
# Full text extraction
# ============================================================

HTML_TAG = re.compile(r"<[^>]+>")
SS = re.compile(r"(?is)<(script|style|noscript|template).*?>.*?</\1>")
WS = re.compile(r"\s+")


def clean(txt: str) -> str:
    if not txt:
        return ""
    txt = SS.sub(" ", txt)
    txt = HTML_TAG.sub(" ", txt)
    txt = WS.sub(" ", txt)
    return html.unescape(txt).strip()


def pick_amp(html_str: str, base: str) -> str | None:
    m = re.search(r'rel=["\\\']amphtml["\\\']\\s+href=["\\\']([^"\\\']+)["\\\']', html_str, re.I)
    return urllib.parse.urljoin(base, m.group(1)) if m else None


def read(url: str) -> str:
    with httpx.Client(**_httpx_client_kwargs()) as c:
        r = c.get(url)
        r.raise_for_status()

        html1 = r.text or ""
        base = str(r.url)

        # 1) Try AMP first
        amp = pick_amp(html1, base)
        if amp:
            try:
                ra = c.get(amp)
                ra.raise_for_status()
                html2 = ra.text or ""

                if HAS_TRA:
                    x = trafilatura.extract(
                        html2,
                        include_comments=False,
                        include_formatting=False,
                    ) or ""
                    if x:
                        return x.strip()

                ps = re.findall(r"(?is)<p[^>]*>(.*?)</p>", html2)
                if ps:
                    return clean(" ".join(ps))
            except Exception:
                pass

        # 2) Use trafilatura on original page
        if HAS_TRA:
            try:
                x = trafilatura.extract(
                    html1,
                    include_comments=False,
                    include_formatting=False,
                ) or ""
                if x and len(x.strip()) > 400:
                    return x.strip()
            except Exception:
                pass

        # 3) Backup: collect paragraph-like blocks
        blocks = re.findall(r"(?is)<(h1|h2|h3|p|li)[^>]*>(.*?)</\1>", html1)
        if blocks:
            text = " ".join(clean(b[1]) for b in blocks if b and b[1].strip())
            if len(text) > 300:
                return text.strip()

        # 4) Try main/article region
        m = (
            re.search(r"(?is)<main[^>]*>(.*?)</main>", html1)
            or re.search(r"(?is)<article[^>]*>(.*?)</article>", html1)
        )
        if m:
            text = clean(m.group(1))
            if len(text) > 300:
                return text

        # 5) Fallback: full page cleanup
        return clean(html1)


if __name__ == "__main__":
    # rss_url = "https://news.google.com/rss/articles/CBMixAFBVV95cUxNTzdkampZMFdfbUU2alQzQlhTeVhNM000LTFFZVZoc3Bxal92cnlyWlJpUk5DS3BGMU9aN09uUFBBY0hkMlF0aXN0UGhjUUpzMThJOURRTWhQVkhOaWhuci1RUllNS2ljYWZvQ2xCYURWNkdlQ3BjVnhRYmpJZGV6MTdLTXhXclpxZ2hLZ1lHS0dfLXA2bUVqM01TeXc5RF91TDlZTExLcGJUSmk3eWRpSFBoemY0aGM3WnYxeWpmME5IYm1r?oc=5"
    rss_url='https://www.bloomberg.com/news/articles/2024-04-29/hong-kong-s-hang-seng-index-jumps-20-from-january-low-heads-for-bull-market'
    print(f"USE_PROXY={USE_PROXY}, LOCAL_PROXY={LOCAL_PROXY}")

    source, sign, ts = get_google_params(rss_url)
    url = get_origin_url(source, sign, ts)
    print("origin_url:", url)

    if url:
        print(read(url))