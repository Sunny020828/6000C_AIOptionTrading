# File: src/loansys_llm_cro/urls.py
from get_news.fetch_fulltext import get_google_params, get_origin_url
import urllib
import re
import datetime
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# =============================================
# URL & date helpers
# =============================================
_NOISE_QPARAM_PREFIXES = ("utm_",)
_NOISE_QPARAM_KEYS = {
    "gclid","fbclid","mc_eid","mc_cid","igshid","ref","ref_","src","spm",
    "yclid","adid","campaign_id","track","tracking_id","cmpid","cmp","ncid",
}
MONTH_ABBR = {m: i for i, m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], start=1
)}


BLACKLISTED_DOMAINS = {
    "lokadarshan.news", "enterprise.news", "ies.news","shopify.com"
}
_JUNK_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"(?:^|[\/\.\-\?_])(shop|deal|coupon|discount|buy|sale|promo)(?:[\/\.\-\?_]|$)",
        r"(?:^|[&?])(track|ref|affid|utm_[a-z0-9_]*|clickid|tag)=",   #Scan only query keys
        r"(?:^|[\/\.\-\?_])(product|item|cart|add-to-cart)(?:[\/\.\-\?_]|$)",
        r"aliexpress|ebay|amazon|walmart|bestbuy|flipkart|pinduoduo|pricegrabber|slickdeals|gearbest|shein|temu",
    ]
]

def resolve_real_url(url: str) -> str:
    """还原Google News源站"""
    if not url:
        return url
    if "news.google.com/rss/articles/" in url:
        try:
            source, sign, ts = get_google_params(url)
            real_url = get_origin_url(source, sign, ts)
            if real_url:
                return real_url
        except Exception:
            pass
    return url


    
def canonicalize_url(u: str) -> str:
    """
    Single-point normalization: Completed in one go
    1) Common jumps during unpacking(Google)
    2) If it's a first-level Google News RSS feed, use get_google_params/get_origin_url to restore the origin server.
    3) Light normalization (https, remove utm_*, remove fragment, remove AMP variants, remove trailing slash)
    """
    if not u:
        return ""
    try:
        # --- First unpack common redirects ---
        pu = urllib.parse.urlsplit(u)
        netloc = (pu.netloc or "").lower()
        path   = pu.path or ""

        # Google redirect: https://www.google.com/url?url=<orig> or ?q=<orig>
        if netloc.endswith("google.com") and path.startswith("/url"):
            qs = dict(urllib.parse.parse_qsl(pu.query))
            u = qs.get("q") or qs.get("url") or u
            pu = urllib.parse.urlsplit(u)
            netloc = (pu.netloc or "").lower()
            path   = pu.path or ""

        # --- Restoring the Google News RSS feed ---
        if "news.google.com/rss/articles/" in u:
            try:
                s_id, s_sign, s_ts = get_google_params(u)
                origin = get_origin_url(s_id, s_sign, s_ts)
                if origin:
                    u = origin
                    pu = urllib.parse.urlsplit(u)
                    netloc = (pu.netloc or "").lower()
                    path   = pu.path or ""
            except Exception:
                pass

        # --- Light normalization ---
        scheme = (pu.scheme or "https").lower()
        # Remove AMP subdomains/path suffixes
        if netloc.startswith("amp."):
            netloc = netloc[4:]
        if path.endswith("/amp") or path.endswith("/amp/"):
            path = path[:-4] if path.endswith("/amp") else path[:-5]
        path = re.sub(r"/+$", "", path)

        # Remove utm_* and other noise parameters, keep others
        q = [(k, v) for (k, v) in urllib.parse.parse_qsl(pu.query, keep_blank_values=True)
             if not k.lower().startswith("utm_")]
        query = urllib.parse.urlencode(q)

        return urllib.parse.urlunsplit((scheme, netloc, path, query, "")) or u.strip()
    except Exception:
        return u.strip()
    
    
def url_hostname(u: str) -> str:
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:
        return ""

def is_blacklisted_url(canon_url: str) -> bool:
    host = url_hostname(canon_url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in BLACKLISTED_DOMAINS)
    
    
def is_junk_url(canon_url: str) -> bool:
    return any(p.search(canon_url.lower()) for p in _JUNK_PATTERNS)


def publisher_from_url(u: str) -> str:
    try:
        netloc = urllib.parse.urlsplit(u).netloc.lower()
        for p in ("www.", "amp.", "m.", "news."):
            if netloc.startswith(p):
                netloc = netloc[len(p):]
        return netloc
    except Exception:
        return ""




def norm_title(t: str) -> str:
    return (t or "").strip().lower()


    
def url_hash(u: str) -> str:
    canon = canonicalize_url(u)
    return hashlib.sha1(canon.encode("utf-8", errors="ignore")).hexdigest()


def infer_date_from_url_or_text(url: str, text: str = "") -> Optional[str]:
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", url)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime.date(y, mo, d).isoformat()
        except Exception:
            pass
    m2 = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s*(20\d{2})", text or url, flags=re.I)
    if m2:
        mo = MONTH_ABBR[m2.group(1)[:3].lower()]
        d  = int(m2.group(2))
        y  = int(m2.group(3))
        try:
            return datetime.date(y, mo, d).isoformat()
        except Exception:
            pass
    m3 = re.search(r"(20\d{2})", url)
    if m3:
        y = int(m3.group(1))
        try:
            return datetime.date(y, 1, 1).isoformat()
        except Exception:
            pass
    return None



def coerce_to_date(x) -> Optional[datetime.date]:
    """Unify multiple inputs into a date; return None on failure."""
    if x is None:
        return None
    if isinstance(x, datetime.date) and not isinstance(x, datetime.datetime):
        return x
    if isinstance(x, datetime.datetime):
        # Directly extract UTC date (ignore time)
        return x.date()
    s = str(x).strip()
    # 1) First capture YYYY-MM-DD prefix (compatible with 2025-11-02T20:09:43Z / +08:00 etc.)
    m = re.match(r'^(\d{4}-\d{2}-\d{2})', s)
    if m:
        try:
            return datetime.date.fromisoformat(m.group(1))
        except Exception:
            return None
    # 2) Fallback failure
    return None


def in_range(dt_val: Optional[object],
             sd: Optional[datetime.date] = None,
             ed: Optional[datetime.date] = None) -> bool:
    d = coerce_to_date(dt_val)
    if not d:
        return False
    if sd and d < sd:
        return False
    if ed and d > ed:
        return False
    return True


def dedup_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Ensure news is not duplicated within a single round
    seen = set()
    out = []
    for it in items:
        title = norm_title(it.get("title",""))
        url   = it.get("url","")
        src   = (it.get("source","") or "").lower()
        key = (title, url_hash(url))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def date_sort_key(it: Dict[str, Any]) -> Tuple[int, str]:
    """
    Sort by date (recent first). Items without valid date go to the tail.
    Secondary key keeps order stable across runs.
    """
    iso = it.get("approx_date")
    ts = 0
    if iso:
        try:
            ts = datetime.date.fromisoformat(iso).toordinal()
        except Exception:
            ts = 0
    return (ts, url_hash(it.get("url","")))


