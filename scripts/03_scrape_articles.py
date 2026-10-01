# scripts/03_scrape_articles.py
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

try:
    from trafilatura import extract as trafi_extract
    from trafilatura.metadata import extract_metadata as trafi_extract_metadata
except Exception as e:
    print("ERROR: trafilatura is not installed or failed to import.")
    print("Install: pip install trafilatura")
    raise

try:
    from dateutil import parser as dateparser  # optional but very useful
except Exception:
    dateparser = None

try:
    from tqdm import tqdm  # optional progress bar
except Exception:
    tqdm = None


GDEFLATE_HEADERS = {
    "User-Agent": "nlp-portal-project/1.0 (student project)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en,en-US;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


@dataclass
class ArticleRecord:
    portal: str
    topic: str
    url: str

    fetched_at_utc: str
    http_status: Optional[int]
    final_url: Optional[str]
    content_type: Optional[str]

    title: Optional[str]
    date_published: Optional[str]     # ISO string if parsed
    author: Optional[str]
    sitename: Optional[str]

    text: Optional[str]
    n_chars: int
    n_words: int
    text_sha1: Optional[str]

    error: Optional[str]              # filled only if failed or empty extraction


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def parse_date_to_iso(dt_str: Optional[str]) -> Optional[str]:
    if not dt_str:
        return None
    dt_str = dt_str.strip()
    if not dt_str:
        return None
    # best effort parsing
    try:
        if dateparser is not None:
            dt = dateparser.parse(dt_str)
        else:
            # fallback: try ISO-like
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def load_done_urls(out_jsonl: Path) -> set[str]:
    done = set()
    if not out_jsonl.exists():
        return done
    with out_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                u = obj.get("url")
                if u:
                    done.add(u)
            except Exception:
                # ignore malformed line
                continue
    return done


def save_html(raw_html_dir: Path, portal: str, topic: str, url: str, html: str) -> Path:
    raw_html_dir.mkdir(parents=True, exist_ok=True)
    h = sha1_text(url)
    p = raw_html_dir / f"{portal}__{topic}__{h}.html.gz"
    with gzip.open(p, "wt", encoding="utf-8") as gz:
        gz.write(html)
    return p


def fetch_with_retries(
    session: requests.Session,
    url: str,
    headers: dict,
    timeout: int,
    max_retries: int,
    backoff_base: float,
) -> tuple[Optional[requests.Response], Optional[str]]:
    """
    Returns (response, error_str)
    """
    backoff = backoff_base
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)

            # Handle hard rate limit / transient server errors with backoff
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(120.0, backoff) + random.uniform(0, 2.0)
                time.sleep(wait)
                backoff *= 2.0
                continue

            # For other codes, just return (even 403/404 etc.)
            return r, None

        except requests.RequestException as e:
            wait = min(120.0, backoff) + random.uniform(0, 2.0)
            time.sleep(wait)
            backoff *= 2.0
            last = f"{type(e).__name__}: {e}"
            err = f"request_error attempt={attempt}/{max_retries}: {last}"

    return None, err


def extract_article(html: str, url: str) -> tuple[Optional[str], dict]:
    """
    Returns (text, metadata_dict)
    """
    # main text
    text = trafi_extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    md = trafi_extract_metadata(html, default_url=url)
    meta = {}
    if md is not None:
        # md fields are attributes
        meta = {
            "title": getattr(md, "title", None),
            "date": getattr(md, "date", None),
            "author": getattr(md, "author", None),
            "sitename": getattr(md, "sitename", None),
        }

    return text, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/url_candidates_disjoint.csv")
    ap.add_argument("--out", default="data/articles_raw.jsonl")
    ap.add_argument("--raw-html-dir", default="data/raw_html")
    ap.add_argument("--save-html", choices=["none", "fail", "all"], default="fail",
                    help="Save raw HTML to data/raw_html. 'fail' saves only for errors/empty extraction.")
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--sleep-min", type=float, default=0.25)
    ap.add_argument("--sleep-max", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--portal", default=None, help="Optional filter: only one portal (bbc/guardian/aljazeera)")
    ap.add_argument("--topic", default=None, help="Optional filter: only one topic (topicA/topicB)")
    args = ap.parse_args()

    in_csv = Path(args.input)
    out_jsonl = Path(args.out)
    raw_html_dir = Path(args.raw_html_dir)

    if not in_csv.exists():
        print(f"ERROR: input CSV not found: {in_csv}")
        sys.exit(1)

    df = pd.read_csv(in_csv)
    # basic sanity
    for col in ("portal", "topic", "url"):
        if col not in df.columns:
            print(f"ERROR: input CSV missing column: {col}")
            sys.exit(1)

    if args.portal:
        df = df[df["portal"] == args.portal]
    if args.topic:
        df = df[df["topic"] == args.topic]

    df = df.drop_duplicates(subset=["portal","url"]).reset_index(drop=True)

    done = load_done_urls(out_jsonl)
    todo = df[~df["url"].isin(done)].copy().reset_index(drop=True)

    if args.limit and args.limit > 0:
        todo = todo.iloc[: args.limit].copy()

    print(f"Input rows: {len(df)}")
    print(f"Already scraped: {len(done)}")
    print(f"To scrape now: {len(todo)}")
    print(f"Output: {out_jsonl}")

    # one session (sequential, stable, low friction)
    session = requests.Session()

    # open once, append
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    f_out = out_jsonl.open("a", encoding="utf-8")

    iterable = todo.itertuples(index=False)
    if tqdm is not None:
        iterable = tqdm(list(iterable), total=len(todo), desc="scraping")

    ok, fail, empty = 0, 0, 0

    try:
        for row in iterable:
            portal = getattr(row, "portal")
            topic = getattr(row, "topic")
            url = getattr(row, "url")

            fetched_at = datetime.now(timezone.utc).isoformat()

            r, req_err = fetch_with_retries(
                session=session,
                url=url,
                headers=GDEFLATE_HEADERS,
                timeout=args.timeout,
                max_retries=args.max_retries,
                backoff_base=5.0,
            )

            http_status = None
            final_url = None
            ctype = None
            html = None

            if r is not None:
                http_status = r.status_code
                final_url = str(r.url) if r.url else None
                ctype = r.headers.get("Content-Type", None)
                # keep only html-ish
                if ctype and ("text/html" not in ctype.lower()) and ("application/xhtml" not in ctype.lower()):
                    rec = ArticleRecord(
                        portal=portal, topic=topic, url=url,
                        fetched_at_utc=fetched_at,
                        http_status=http_status,
                        final_url=final_url,
                        content_type=ctype,
                        title=None, date_published=None, author=None, sitename=None,
                        text=None, n_chars=0, n_words=0, text_sha1=None,
                        error=f"non_html_content_type: {ctype}",
                    )
                    f_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    f_out.flush()
                    fail += 1
                    time.sleep(random.uniform(args.sleep_min, args.sleep_max))
                    continue

                # requests will decode gzip/brotli automatically in most cases
                html = (r.text or "").strip()

            if r is None:
                rec = ArticleRecord(
                    portal=portal, topic=topic, url=url,
                    fetched_at_utc=fetched_at,
                    http_status=None,
                    final_url=None,
                    content_type=None,
                    title=None, date_published=None, author=None, sitename=None,
                    text=None, n_chars=0, n_words=0, text_sha1=None,
                    error=req_err or "request_failed_unknown",
                )
                f_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                f_out.flush()
                fail += 1
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))
                continue

            # non-200 is allowed but usually not useful
            if http_status is not None and http_status >= 400:
                if args.save_html in ("all", "fail") and html:
                    save_html(raw_html_dir, portal, topic, url, html)
                rec = ArticleRecord(
                    portal=portal, topic=topic, url=url,
                    fetched_at_utc=fetched_at,
                    http_status=http_status,
                    final_url=final_url,
                    content_type=ctype,
                    title=None, date_published=None, author=None, sitename=None,
                    text=None, n_chars=0, n_words=0, text_sha1=None,
                    error=f"http_error_{http_status}",
                )
                f_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                f_out.flush()
                fail += 1
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))
                continue

            # extract
            text, meta = extract_article(html, final_url or url)
            text = normalize_ws(text) if text else None

            title = meta.get("title")
            author = meta.get("author")
            sitename = meta.get("sitename")
            date_raw = meta.get("date")
            date_iso = parse_date_to_iso(date_raw)

            if args.save_html == "all" and html:
                save_html(raw_html_dir, portal, topic, url, html)

            if not text or len(text) < 50:
                if args.save_html in ("all", "fail") and html:
                    save_html(raw_html_dir, portal, topic, url, html)
                rec = ArticleRecord(
                    portal=portal, topic=topic, url=url,
                    fetched_at_utc=fetched_at,
                    http_status=http_status,
                    final_url=final_url,
                    content_type=ctype,
                    title=title, date_published=date_iso, author=author, sitename=sitename,
                    text=None, n_chars=0, n_words=0, text_sha1=None,
                    error="empty_or_too_short_extraction",
                )
                f_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                f_out.flush()
                empty += 1
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))
                continue

            n_chars = len(text)
            n_words = len(text.split())
            t_sha1 = sha1_text(text)

            rec = ArticleRecord(
                portal=portal, topic=topic, url=url,
                fetched_at_utc=fetched_at,
                http_status=http_status,
                final_url=final_url,
                content_type=ctype,
                title=title, date_published=date_iso, author=author, sitename=sitename,
                text=text,
                n_chars=n_chars,
                n_words=n_words,
                text_sha1=t_sha1,
                error=None,
            )
            f_out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            f_out.flush()
            ok += 1

            # polite sleep
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))

    finally:
        f_out.close()

    print("\n=== DONE ===")
    print(f"ok={ok} fail={fail} empty={empty}")
    print(f"saved: {out_jsonl}")


if __name__ == "__main__":
    main()
