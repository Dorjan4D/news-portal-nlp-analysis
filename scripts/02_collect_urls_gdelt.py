# scripts/02_collect_urls_gdelt.py
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

HEADERS = {"User-Agent": "nlp-portal-project/1.0 (student project)"}

# Conservative pacing to avoid quota spikes / overload behavior
SUCCESS_SLEEP_SEC = 8.0
MAX_RETRIES = 10

MAXRECORDS = 250               # hard cap in artlist responses
DEFAULT_WINDOW_DAYS = 14       # initial window, will adaptively shrink if capped
MIN_WINDOW_DAYS = 1            # do not split smaller than this

CKPT_DIR = Path("data/url_ckpt")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = Path("data/url_candidates.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

PORTALS = {
    "bbc": ["bbc.com", "bbc.co.uk"],
    "guardian": ["theguardian.com"],
    "aljazeera": ["aljazeera.com"],
}

TOPICS = {
    "topicA": ["israel", "gaza", "hamas", "palestine", "west bank", "ceasefire", "idf", "netanyahu"],
    "topicB": ["inflation", "economy", "markets", "oil", "gdp", "recession", "interest rates", "central bank"],
}


def build_query(domains: list[str], keywords: list[str]) -> str:
    def or_group(items, render):
        if len(items) == 1:
            return render(items[0])  # NO parentheses
        return "(" + " OR ".join(render(x) for x in items) + ")"  # parentheses only for OR

    kw_part = or_group(keywords, lambda k: f"\"{k}\"" if " " in k else k)
    dom_part = or_group(domains, lambda d: f"domainis:{d}")
    return f"{kw_part} {dom_part} sourcelang:english"


def load_ckpt(portal: str, topic: str) -> set[str]:
    p = CKPT_DIR / f"{portal}__{topic}.txt"
    if not p.exists():
        return set()
    return set(u.strip() for u in p.read_text(encoding="utf-8").splitlines() if u.strip())


def save_ckpt(portal: str, topic: str, urls: set[str]) -> None:
    p = CKPT_DIR / f"{portal}__{topic}.txt"
    p.write_text("\n".join(sorted(urls)), encoding="utf-8")


def gdelt_get_json(params: dict) -> dict | None:
    backoff = 10.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(GDELT_ENDPOINT, params=params, headers=HEADERS, timeout=30)

            if r.status_code == 429:
                wait = min(600.0, backoff) + random.uniform(0, 3)
                print(f"  ! 429 rate limit (attempt {attempt}/{MAX_RETRIES}) -> sleep {wait:.1f}s")
                time.sleep(wait)
                backoff = min(600.0, backoff * 2)
                continue

            if r.status_code >= 500:
                wait = min(600.0, backoff) + random.uniform(0, 3)
                print(f"  ! {r.status_code} server error (attempt {attempt}/{MAX_RETRIES}) -> sleep {wait:.1f}s")
                time.sleep(wait)
                backoff = min(600.0, backoff * 2)
                continue

            r.raise_for_status()

            text = (r.text or "").strip()
            if "Parentheses may only be used" in text:
                raise RuntimeError(f"GDELT query syntax error: {text[:200]}")
            ct = (r.headers.get("Content-Type") or "").lower()

            if (not text) or (("application/json" not in ct) and (not text.startswith("{"))):
                snippet = text[:120].replace("\n", " ")
                wait = min(600.0, backoff) + random.uniform(0, 3)
                print(
                    f"  ! non-JSON/empty response (attempt {attempt}/{MAX_RETRIES}) "
                    f"ct={ct} snip='{snippet}' -> sleep {wait:.1f}s"
                )
                time.sleep(wait)
                backoff = min(600.0, backoff * 2)
                continue

            try:
                data = r.json()
            except Exception:
                wait = min(600.0, backoff) + random.uniform(0, 3)
                print(f"  ! JSON parse failed (attempt {attempt}/{MAX_RETRIES}) -> sleep {wait:.1f}s")
                time.sleep(wait)
                backoff = min(600.0, backoff * 2)
                continue

            time.sleep(SUCCESS_SLEEP_SEC)
            return data

        except requests.RequestException as e:
            wait = min(600.0, backoff) + random.uniform(0, 3)
            print(f"  ! request error (attempt {attempt}/{MAX_RETRIES}): {type(e).__name__} -> sleep {wait:.1f}s")
            time.sleep(wait)
            backoff = min(600.0, backoff * 2)

    return None


def fetch_window_urls(query: str, start_dt: datetime, end_dt: datetime) -> list[str]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sort": "datedesc",
        "maxrecords": MAXRECORDS,  # up to 250 in artlist
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }
    data = gdelt_get_json(params)
    if not data:
        return []
    arts = data.get("articles", [])
    urls = []
    for a in arts:
        u = a.get("url")
        if u:
            urls.append(u)
    return urls


def _safe_midpoint(a: datetime, b: datetime) -> datetime:
    """Return a midpoint strictly between a and b (when possible)."""
    delta = b - a
    mid = a + delta / 2
    # ensure progress in case of weird rounding
    if mid <= a:
        mid = a + timedelta(seconds=1)
    if mid >= b:
        mid = b - timedelta(seconds=1)
    return mid


def fetch_urls_adaptive(query: str, start_dt: datetime, end_dt: datetime, min_window: timedelta, depth: int = 0) -> list[str]:
    """
    Fetch URLs for [start_dt, end_dt] adaptively:
      - if we hit MAXRECORDS, split the window into halves and retry
      - stop splitting once window <= min_window
    """
    got = fetch_window_urls(query, start_dt, end_dt)

    window_len = end_dt - start_dt
    indent = "  " * min(depth, 6)

    if len(got) >= MAXRECORDS and window_len > min_window:
        mid = _safe_midpoint(start_dt, end_dt)
        print(f"{indent}! hit cap ({len(got)} == {MAXRECORDS}) -> split {start_dt.date()}–{end_dt.date()} at {mid.date()}")
        left = fetch_urls_adaptive(query, start_dt, mid, min_window, depth + 1)
        right = fetch_urls_adaptive(query, mid, end_dt, min_window, depth + 1)
        return left + right

    if len(got) >= MAXRECORDS and window_len <= min_window:
        # This can still be truncated if a single day has >250 matches.
        print(f"{indent}! WARNING: cap hit even at min window ({min_window}). {start_dt.date()}–{end_dt.date()} may be truncated.")

    return got


def collect_combo(
    portal: str,
    topic: str,
    domains: list[str],
    keywords: list[str],
    target: int,
    start_date: date,
    end_date: date,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_window_days: int = MIN_WINDOW_DAYS,
) -> None:
    query = build_query(domains, keywords)
    collect_all = (target <= 0)

    print(f"\n[{portal} | {topic}] target={'ALL' if collect_all else target}")
    print(f"  query: {query}")
    print(f"  range: {start_date} -> {end_date} (window_days={window_days}, min_window_days={min_window_days})")

    urls = load_ckpt(portal, topic)
    print(f"  ckpt already has: {len(urls)}")

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)

    cur = start_dt
    step = timedelta(days=window_days)
    min_window = timedelta(days=min_window_days)

    while cur < end_dt and (collect_all or len(urls) < target):
        nxt = min(cur + step, end_dt)

        got = fetch_urls_adaptive(query, cur, nxt, min_window=min_window)
        before = len(urls)
        urls.update(got)
        after = len(urls)

        save_ckpt(portal, topic, urls)
        print(f"  window {cur.date()}–{nxt.date()} -> +{len(got)} (unique {before}->{after}) [saved ckpt]")

        cur = nxt


def export_combined_csv(selected_portals: list[str] | None = None, selected_topics: list[str] | None = None) -> None:
    rows = []
    for p in CKPT_DIR.glob("*.txt"):
        name = p.stem  # portal__topic
        if "__" not in name:
            continue

        portal, topic = name.split("__", 1)

        if selected_portals and portal not in selected_portals:
            continue
        if selected_topics and topic not in selected_topics:
            continue

        urls = [u.strip() for u in p.read_text(encoding="utf-8").splitlines() if u.strip()]
        for u in urls:
            rows.append({"portal": portal, "topic": topic, "url": u})

    df = pd.DataFrame(rows, columns=["portal", "topic", "url"]).drop_duplicates()
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV} (rows={len(df)})")
    if not df.empty:
        print(df.groupby(["portal", "topic"]).size())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portal", default=None, help="Run only one portal (bbc/guardian/aljazeera)")
    ap.add_argument("--topic", default=None, help="Run only one topic (topicA/topicB)")
    ap.add_argument("--target", type=int, default=1200, help="Target unique URLs per (portal, topic). Use 0 for ALL.")
    ap.add_argument("--days-back", type=int, default=548, help="How many days back to scan (e.g., 365, 548).")
    ap.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="Initial window size in days (adaptive will split).")
    ap.add_argument("--min-window-days", type=int, default=MIN_WINDOW_DAYS, help="Minimum window size in days for adaptive splitting.")
    args = ap.parse_args()

    if args.portal and args.portal not in PORTALS:
        raise SystemExit(f"Unknown portal: {args.portal}. Options: {list(PORTALS.keys())}")
    if args.topic and args.topic not in TOPICS:
        raise SystemExit(f"Unknown topic: {args.topic}. Options: {list(TOPICS.keys())}")
    if args.window_days < 1:
        raise SystemExit("--window-days must be >= 1")
    if args.min_window_days < 1:
        raise SystemExit("--min-window-days must be >= 1")
    if args.min_window_days > args.window_days:
        raise SystemExit("--min-window-days must be <= --window-days")

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days_back)

    portals_to_run = PORTALS if not args.portal else {args.portal: PORTALS[args.portal]}
    topics_to_run = TOPICS if not args.topic else {args.topic: TOPICS[args.topic]}

    # Collect
    for portal, domains in portals_to_run.items():
        for topic, keywords in topics_to_run.items():
            collect_combo(
                portal=portal,
                topic=topic,
                domains=domains,
                keywords=keywords,
                target=args.target,
                start_date=start_date,
                end_date=end_date,
                window_days=args.window_days,
                min_window_days=args.min_window_days,
            )

    # Export from ckpts
    export_combined_csv(
        selected_portals=list(portals_to_run.keys()) if args.portal else None,
        selected_topics=list(topics_to_run.keys()) if args.topic else None,
    )


if __name__ == "__main__":
    main()
