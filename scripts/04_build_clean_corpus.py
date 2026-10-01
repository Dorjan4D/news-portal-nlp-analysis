#!/usr/bin/env python3
"""
Build a high-precision, model-safe candidate corpus from articles_raw.jsonl.

This stage intentionally does NOT create train/val/test splits.
It performs:
  * successful-fetch/date/page-type eligibility
  * label-blind removal of obvious web/template artifacts before topic scoring
  * high-precision topic relevance revalidation from title + article lead
  * global (label-blind) text cleaning
  * exact deduplication after cleaning
  * detailed audit/removal reports

Important methodological property:
  clean_text_global() never receives the portal label. Target outlet names,
  domains, author names, and template artifacts are removed using one global
  rule set from every article.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
import unicodedata
import statistics
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Configuration: corpus definition
# ---------------------------------------------------------------------------

DEFAULT_START = "2024-07-28"
DEFAULT_END = "2026-01-27"
DEFAULT_MIN_BODY_WORDS = 150
TOPIC_LEAD_WORDS = 180

# Guardian sections that are ordinary written news/reporting and are comparable
# to BBC News and Al Jazeera News/Economy. Explicit opinion, culture, food,
# lifestyle, sports and commercial verticals are intentionally not included.
GUARDIAN_ALLOWED_SECTIONS = {
    "world",
    "us-news",
    "australia-news",
    "politics",
    "uk-news",
    "business",
    "environment",
    "society",
    "technology",
    "global-development",
    "media",
    "law",
    "education",
    "science",
    "news",
    "money",
}

# Exact outlet/domain clues: one global list, applied to ALL samples.
# Standalone Guardian is case-sensitive because guardian is an ordinary word.
OUTLET_PATTERNS_CASE_INSENSITIVE = [
    r"\bBBC(?:\s+News|\s+Verify|\s+Sport|\s+World\s+Service|\s+Sounds|\s+iPlayer|\s+InDepth)?\b",
    r"\bAl\s*Jazeera(?:\s+English)?\b",
    r"\b(?:www\.)?bbc\.(?:com|co\.uk)\b",
    r"\b(?:www\.)?aljazeera\.com\b",
    r"\b(?:www\.)?theguardian\.com\b",
]
OUTLET_PATTERNS_CASE_SENSITIVE = [
    r"\bThe Guardian\b",
    r"\bGuardian News and Media\b",
    r"\bGuardian Australia\b",
    r"\bGuardian\b",
]

# Wire/source identity can become another shortcut, especially when bylines are
# retained differently by outlets. Remove these names globally as source credits.
WIRE_PATTERNS = [
    r"\bReuters\b",
    r"\bAssociated Press\b",
    r"\bAgence France-Presse\b",
    r"\bAgence France Presse\b",
    r"\bAP\b",
    r"\bAFP\b",
]

# Clearly non-semantic web artifacts. These patterns are deliberately specific.
LEADING_ARTIFACT_PATTERNS = [
    re.compile(r"^\s*-\s*Published\s+", re.IGNORECASE),
    re.compile(r"^\s*Published\s+", re.IGNORECASE),
]

# Tail-only markers. If they occur near the end, the marker and everything after
# it is removed. This is safer than deleting similar language in article prose.
TAIL_CUTOFF_PATTERNS = [
    re.compile(r"\bBBC InDepth is the home on the website and app\b", re.IGNORECASE),
    re.compile(r"\bYou can now sign up for notifications that will alert you whenever\b", re.IGNORECASE),
    re.compile(r"\bWant more\?\s*Visit BBC Food\b", re.IGNORECASE),
    re.compile(r"\bDon['’]t forget To follow us on\b", re.IGNORECASE),
    re.compile(r"\bSign up to the [^.]{0,100} newsletter\b", re.IGNORECASE),
    re.compile(r"\bSign up for the [^.]{0,100} newsletter\b", re.IGNORECASE),
    re.compile(r"\bGet in Touch If you have any questions or comments about any of our newsletters\b", re.IGNORECASE),
    re.compile(r"\bExplore more on these topics\b", re.IGNORECASE),
]

# Meta/correction material near the end. We remove only the matching sentence,
# not the rest of the article.
TAIL_SENTENCE_PATTERNS = [
    re.compile(r"\bThis article was (?:amended|corrected)[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"\bOriginally published\s+[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"\bFirst published\s+[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"©\s*\d{4}[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
    re.compile(r"\bAll rights reserved\b[^.?!]*(?:[.?!]|$)", re.IGNORECASE),
]

# High-specificity promotional/newsletter blocks that can occur before the very
# end of an extracted article. These rules are GLOBAL and label-blind: they are
# applied to every article, regardless of portal. They require unmistakable web
# subscription language so legitimate mentions such as "he wrote in his
# newsletter" or "sign up to a ceasefire" are preserved.
GLOBAL_PROMO_PATTERNS = [
    re.compile(
        r"\bSign up for our Politics Essential newsletter to read top political analysis,?\s*"
        r"gain insight from across the UK and stay up to speed with the big moments\.?\s*"
        r"It(?:'|’)?ll be delivered straight to your inbox every weekday\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bFollow the twists and turns of Trump(?:'|’)?s (?:second )?(?:presidential )?term with "
        r"[^.]{0,220}?newsletter\.?\s*Readers in the UK can sign up here\.?\s*"
        r"Those outside the UK can sign up here\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bNorth America correspondent[^.]{0,220}?newsletter\.?\s*Readers in the UK can sign up here\.?\s*"
        r"Those outside the UK can sign up here\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bSign up\s+First Thing is delivered to thousands of inboxes every weekday\.?\s*"
        r"If you(?:'|’)?re not already signed up, subscribe now\.?",
        re.IGNORECASE,
    ),
    re.compile(r"\bSign up:\s*AU Breaking News email\b", re.IGNORECASE),
    re.compile(
        r"\bSign up\s+If you would like to receive The Crunch to your email inbox[^.]{0,180}?sign up here\.?",
        re.IGNORECASE,
    ),
    re.compile(r"\bSign up here for a weekly roundup of The Upside[^.]{0,220}?(?:[.!?]|$)", re.IGNORECASE),
    re.compile(
        r"\bDon(?:'|’)?t miss important [^.]{0,160}?coverage\.?\s*Get our free app and sign up for [^.]{0,160}?alerts\.?",
        re.IGNORECASE,
    ),
    # Generic but still high-specificity newsletter CTA: the sentence must
    # contain both a newsletter reference and subscription/delivery vocabulary.
    re.compile(
        r"(?<!\w)[^.?!]{0,180}\b(?:sign up|subscribe)\b[^.?!]{0,180}\bnewsletter\b[^.?!]{0,180}(?:[.?!]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\w)[^.?!]{0,180}\bnewsletter\b[^.?!]{0,180}\b(?:sign up|subscribe|inbox|delivered)\b[^.?!]{0,180}(?:[.?!]|$)",
        re.IGNORECASE,
    ),
]


def strip_global_promos(text: str) -> tuple[str, int]:
    total = 0
    for pat in GLOBAL_PROMO_PATTERNS:
        text, n = pat.subn(" ", text)
        total += n
    return normalize_spaces(text), total


# Generic URLs / emails / social handles are web-level artifacts rather than
# journalistic language. Apply globally.
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
SOCIAL_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,30}\b")

# BBC's scraped link annotation appears as ", external,". Remove only this
# structural annotation, not normal uses of the English word "external".
BBC_EXTERNAL_ANNOTATION_RE = re.compile(r",\s*external\b\s*,?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Topic relevance rules
# ---------------------------------------------------------------------------

TOPIC_A_SPECIFIC = [
    r"\bgaza\b",
    r"\bhamas\b",
    r"\bpalestin(?:e|ian|ians)\b",
    r"\bwest bank\b",
    r"\bidf\b",
    r"\bnetanyahu\b",
    r"\bisraeli\s+(?:military|army|forces|troops|airstrikes?|strikes?|offensive)\b",
    r"\bisrael[-–— ]hamas\b",
]
TOPIC_A_ISRAEL = [r"\bisrael\b", r"\bisraeli\b"]
TOPIC_A_CONFLICT = [
    r"\bwar\b", r"\bconflict\b", r"\bceasefire\b", r"\btruce\b",
    r"\bhostages?\b", r"\bcaptives?\b", r"\battack(?:s|ed)?\b",
    r"\bairstrikes?\b", r"\bstrikes?\b", r"\bbomb(?:ing|ed|s)?\b",
    r"\bmilitary\b", r"\barmy\b", r"\boffensive\b", r"\bsiege\b",
    r"\bhumanitarian\b", r"\baid\b", r"\bkilled\b", r"\bdead\b",
    r"\bhezbollah\b", r"\blebanon\b", r"\biran\b",
]

# Strong economic phrases/concepts. One strong concept in the headline is enough;
# in the lead we require evidence from >=2 distinct concept groups unless one of
# the very explicit macro/monetary phrases is present.
ECON_GROUPS = {
    "macro": [
        r"\binflation\b", r"\beconom(?:y|ic|ics|ies)\b", r"\bgdp\b",
        r"\bgross domestic product\b", r"\brecession\b", r"\bdeflation\b",
        r"\bcost[- ]of[- ]living\b", r"\bunemployment\b", r"\bjobless\b",
        r"\beconomic growth\b", r"\bproductivity\b", r"\bwage growth\b",
    ],
    "monetary": [
        r"\binterest rates?\b", r"\bcentral bank\b", r"\bmonetary policy\b",
        r"\bfederal reserve\b", r"\bthe fed\b", r"\becb\b",
        r"\bbank of england\b", r"\brate cut(?:s)?\b", r"\brate hike(?:s)?\b",
    ],
    "markets": [
        r"\bstock market\b", r"\bfinancial markets?\b", r"\bstock prices?\b",
        r"\bshare prices?\b", r"\bbond yields?\b", r"\btreasury yields?\b",
        r"\bwall street\b", r"\bs&p 500\b", r"\bdow jones\b", r"\bnasdaq\b",
    ],
    "fiscal_trade": [
        r"\btariffs?\b", r"\btrade war\b", r"\btrade deficit\b",
        r"\bbudget deficit\b", r"\bfiscal policy\b", r"\bgovernment debt\b",
        r"\bpublic debt\b", r"\bnational debt\b", r"\bgovernment spending\b",
        r"\btax cuts?\b", r"\btax rises?\b", r"\btax increases?\b",
    ],
    "energy_econ": [
        r"\boil prices?\b", r"\bcrude oil\b", r"\bbrent crude\b", r"\bbrent\b",
        r"\bopec\+?\b", r"\bprice of oil\b", r"\boil market\b",
        r"\bbarrels? per day\b", r"\benergy prices?\b", r"\bgas prices?\b",
    ],
    "finance_business": [
        r"\bearnings\b", r"\bprofits?\b", r"\brevenue\b", r"\binvestors?\b",
        r"\bshares? (?:rose|fell|rise|fall|jumped|slid|gained|lost)\b",
        r"\bmarkets? (?:rose|fell|rise|fall|rallied|slid|gained|lost)\b",
        r"\bbankruptcy\b", r"\binsolvency\b",
    ],
}
# Explicit phrases strong enough to qualify in the lead alone.
ECON_EXPLICIT_LEAD = ECON_GROUPS["macro"] + ECON_GROUPS["monetary"] + [
    r"\bstock market\b", r"\bfinancial markets?\b", r"\boil prices?\b",
    r"\bcrude oil\b", r"\bbrent crude\b", r"\bopec\+?\b",
]

# High-precision headline vocabulary for Topic B. General-news articles need one
# of these concepts in the headline; dedicated Economy/Business pages get a
# slightly more permissive lead-based check.
ECON_TITLE_PATTERNS = [
    # Macro / labour-market indicators
    r"\binflation\b", r"\beconom(?:y|ic|ics|ies)\b", r"\bgdp\b", r"\brecession\b",
    r"\bunemployment\b", r"\bjobless\b", r"\bjobs\b",
    r"\bjob (?:cuts?|losses?|market|growth|creation)\b", r"\bwages?\b", r"\bcost[- ]of[- ]living\b",

    # Monetary policy / rates
    r"\binterest rates?\b", r"\bcentral bank\b", r"\bfederal reserve\b", r"\bthe fed\b",
    r"\brate cut(?:s)?\b", r"\brate hike(?:s)?\b",

    # Fiscal policy / trade. Avoid bare 'trade' because 'trade union' is not
    # necessarily an economics article.
    r"\btariffs?\b",
    r"\btrade (?:war|deal|agreement|talks?|tensions?|relations?|policy|deficit|surplus)\b",
    r"\bfree trade\b", r"\bglobal trade\b", r"\binternational trade\b",
    r"\bbudget\b", r"\bdeficit\b", r"\bdebt\b", r"\btax(?:es|ation)?\b",

    # Financial markets. Avoid bare 'market' / singular 'stock' because of
    # Christmas markets, fish stocks, etc.
    r"\bstock markets?\b", r"\bfinancial markets?\b", r"\bstocks\b", r"\bshare prices?\b",
    r"\bshares? (?:rise|rises|rose|fall|falls|fell|jump|jumps|jumped|drop|drops|dropped|slump|slumps|slumped|surge|surges|surged|gain|gains|gained|lose|loses|lost)\b",
    r"\bbond yields?\b",
    r"\b(?:stock|financial|global|world|bond|currency|oil|energy|housing|property|labour|labor) markets?\b",
    r"\bmarkets? (?:rally|rallies|rallied|rise|rises|rose|fall|falls|fell|slump|slumps|slumped|surge|surges|surged|turmoil|sell[- ]?off)\b",

    # Energy economics
    r"\boil prices?\b", r"\bcrude oil\b", r"\bbrent\b", r"\bopec\+?\b",
    r"\benergy prices?\b", r"\bgas prices?\b",

    # Finance / business terms that are specific enough in a headline
    r"\bearnings\b", r"\bprofits?\b", r"\brevenue\b", r"\binvest(?:ment|or|ors)\b",
    r"\bfinance\b", r"\bfinancial\b", r"\bimf\b", r"\bworld bank\b", r"\bloans?\b", r"\bbailout\b",
    r"\bcurrenc(?:y|ies)\b", r"\banti-dumping\b", r"\bsurcharges?\b", r"\bconsumer prices?\b",
    r"\bproperty\b", r"\bhousing\b", r"\bexports?\b", r"\bimports?\b", r"\bpension\b",
    r"\bpay (?:deal|rise|rises|cut|cuts)\b", r"\bborrowing\b", r"\bwealth\b", r"\bpoverty\b",
    r"\bbanking\b", r"\bbanks?\b", r"\bdollar\b", r"\beurozone\b",
    r"\b(?:euro|yen|rupee|yuan) (?:rises?|falls?|gains?|loses?|slides?|surges?|weakens?|strengthens?|value)\b",
    r"\bmortgage\b",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def parse_utc_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        v = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def date_bound(s: str, end: bool = False) -> datetime:
    dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    if end:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return dt


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_punctuation_gaps(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([\(\[“‘])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]”’])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Delete obvious empty attribution holes such as "According to ,".
    text = re.sub(r"\baccording to\s*,", "", text, flags=re.IGNORECASE)
    # V7: do not rewrite generic "told,". In passive constructions this
    # corrupted ordinary prose (e.g. "they were told, after..." -> "were said,").
    return text.strip()


def repair_mojibake(text: str) -> str:
    """Repair/neutralize the mojibake actually present in the raw corpus.

    Crucially, we also normalize typographic punctuation for *all* portals. This
    avoids turning BBC's damaged curly punctuation into a new inverse shortcut.
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)

    # Recover common truncated E2 sequences seen in this scrape.
    text = text.replace("â¦", "...")   # truncated UTF-8 ellipsis
    text = text.replace("â¬", "€")     # truncated euro sign
    text = text.replace("â¢", "•")     # truncated bullet
    text = text.replace("â¯", " ")     # truncated narrow/no-break space

    # Strong contextual recovery for contractions and possessives.
    text = re.sub(r"(?<=\w)â(?=(?:s|t|re|ve|ll|d|m)\b)", "'", text, flags=re.IGNORECASE)
    # Ambiguous orphan before a lowercase continuation may be a closing quote,
    # dash, or possessive mark. Treat it as a quote; all double-quote typography
    # is removed globally below so we do not guess portal-specifically.
    text = re.sub(r"(?<=\w)â(?=\s+[a-z])", '"', text)
    # Quote-like positions.
    text = re.sub(r"(?:(?<=^)|(?<=[\s([{]))â(?=[A-Z0-9])", '"', text)
    text = re.sub(r"(?<=\w)â(?=[.,!?;:]|$)", '"', text)
    # A lone marker surrounded by spaces is most often a lost dash/quote; use a
    # neutral separator rather than leaving a class-specific corruption glyph.
    text = re.sub(r"\sâ\s", " - ", text)

    # Repair valid two-byte mojibake such as Ã© -> é and Â£ -> £.
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in ("Ã", "Â") and i + 1 < len(text):
            pair = text[i:i+2]
            try:
                fixed = pair.encode("latin-1").decode("utf-8")
                out.append(fixed); i += 2; continue
            except Exception:
                i += 1; continue
        if ch in ("Ã", "Â", "â"):
            # Unrecoverable orphan: remove it globally rather than preserve a
            # portal-specific extraction token.
            i += 1; continue
        out.append(ch); i += 1
    text = "".join(out)

    # Standardize typography identically for every portal. We retain punctuation
    # categories, but not site/encoding-specific Unicode variants.
    trans = str.maketrans({
        "’": "'", "‘": "'", "ʼ": "'", "ʻ": "'",
        "“": '"', "”": '"',
        "–": "-", "—": "-", "‑": "-",
        "…": "...",
        "\u00a0": " ",
    })
    text = text.translate(trans)
    # BBC quote bytes are not recoverable reliably. To avoid quote corruption or
    # quote typography becoming a portal shortcut, drop double quotation marks
    # from ALL samples. Quoted words themselves remain unchanged.
    text = text.replace('"', "")
    return text

def word_list(text: str) -> list[str]:
    return re.findall(r"\b\w+(?:['’]\w+)?\b", text or "", flags=re.UNICODE)


def first_n_words(text: str, n: int) -> str:
    words = word_list(text)
    return " ".join(words[:n])


def compile_any(patterns: Iterable[str], flags=re.IGNORECASE) -> re.Pattern:
    return re.compile("(?:" + ")|(?:".join(patterns) + ")", flags)


A_SPEC_RE = compile_any(TOPIC_A_SPECIFIC)
A_ISRAEL_RE = compile_any(TOPIC_A_ISRAEL)
A_CONFLICT_RE = compile_any(TOPIC_A_CONFLICT)
ECON_GROUP_RE = {k: compile_any(v) for k, v in ECON_GROUPS.items()}
ECON_EXPLICIT_RE = compile_any(ECON_EXPLICIT_LEAD)
ECON_TITLE_RE = compile_any(ECON_TITLE_PATTERNS)


# ---------------------------------------------------------------------------
# Global authors and title catalog
# ---------------------------------------------------------------------------

GENERIC_AUTHOR_VALUES = {
    "editorial", "news agencies", "guardian staff", "guardian staff reporter",
    "al jazeera staff", "staff", "agencies", "reuters", "associated press",
}


def author_components(raw: str) -> list[str]:
    if not raw:
        return []
    raw = repair_mojibake(raw)
    chunks = re.split(r"\s*;\s*|\s+and\s+", raw)
    out = []
    for c in chunks:
        c = normalize_spaces(c.strip(" ,|-"))
        if not c or c.casefold() in GENERIC_AUTHOR_VALUES:
            continue
        # Person-like / multiword organization-like full identities only.
        toks = c.split()
        if 2 <= len(toks) <= 8 and 4 <= len(c) <= 100:
            if sum(ch.isalpha() for ch in c) >= max(3, len(c) // 2):
                out.append(c)
    return out


NAME_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’.-][^\W\d_]+)*", re.UNICODE)

def _name_token(s: str) -> str:
    return s.casefold().replace("’", "'")

class AuthorMatcher:
    """Fast token-trie-ish exact full-name remover for thousands of authors."""
    def __init__(self, names: list[str]):
        by_first = defaultdict(list)
        for name in names:
            toks = tuple(_name_token(m.group(0)) for m in NAME_TOKEN_RE.finditer(name))
            if len(toks) >= 2:
                by_first[toks[0]].append(toks)
        for k in by_first:
            by_first[k].sort(key=len, reverse=True)
        self.by_first = dict(by_first)

    def sub(self, text: str) -> tuple[str, int]:
        matches = list(NAME_TOKEN_RE.finditer(text))
        if not matches:
            return text, 0
        toks = [_name_token(m.group(0)) for m in matches]
        spans = []
        i = 0
        while i < len(toks):
            cands = self.by_first.get(toks[i])
            found = None
            if cands:
                for seq in cands:
                    L = len(seq)
                    if i + L <= len(toks) and tuple(toks[i:i+L]) == seq:
                        found = L
                        break
            if found:
                spans.append((matches[i].start(), matches[i+found-1].end()))
                i += found
            else:
                i += 1
        if not spans:
            return text, 0
        out=[]; last=0
        for a,b in spans:
            out.append(text[last:a]); out.append(" "); last=b
        out.append(text[last:])
        return "".join(out), len(spans)

def build_author_matcher(rows: list[dict]) -> tuple[AuthorMatcher, list[str]]:
    names = set()
    for o in rows:
        for a in author_components(o.get("author") or ""):
            names.add(a)
    names = sorted(names, key=lambda s: (-len(s), s.casefold()))
    return AuthorMatcher(names), names

def build_title_lookup(rows: list[dict]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Index raw titles by first 20 and 10 casefolded chars."""
    d20: dict[str, list[str]] = defaultdict(list)
    d10: dict[str, list[str]] = defaultdict(list)
    for o in rows:
        t = normalize_spaces(repair_mojibake(o.get("title") or ""))
        if len(t) >= 8:
            cf=t.casefold(); d20[cf[:20]].append(t); d10[cf[:10]].append(t)
    for d in (d20,d10):
        for k in d: d[k].sort(key=len, reverse=True)
    return dict(d20), dict(d10)


# ---------------------------------------------------------------------------
# Template cleanup
# ---------------------------------------------------------------------------

REC_START_RE = re.compile(r"Recommended Stories\s+list of\s+(\d+)\s+items?", re.IGNORECASE)
REC_MARKER_TMPL = r"-\s*list\s+{n}\s+of\s+{n}"


def strip_recommended_stories(text: str, title_lookup: tuple[dict[str, list[str]], dict[str, list[str]]]) -> tuple[str, int]:
    """Remove AJ-style 'Recommended Stories list of N items...' blocks globally.

    The flattened extraction has no HTML boundaries. We locate the final list marker,
    then identify the last recommended title using the global raw-title catalog.
    A conservative punctuation fallback is used if the title is not in the catalog.
    """
    removed = 0
    while True:
        m = REC_START_RE.search(text)
        if not m:
            break
        try:
            n = int(m.group(1))
        except Exception:
            break
        if not (1 <= n <= 12):
            break
        # Some pages flatten an empty recommendation widget as
        # "Recommended Stories list of 1 itemend of list".
        if n == 1:
            one = re.match(r"\s*end of list", text[m.end():], flags=re.IGNORECASE)
            if one:
                text = text[:m.start()] + " " + text[m.end()+one.end():]
                removed += 1
                continue
        last_marker = re.compile(REC_MARKER_TMPL.format(n=n), re.IGNORECASE)
        lm = last_marker.search(text, m.end())
        if not lm:
            # Remove only the unmistakable marker itself rather than risk prose.
            text = text[:m.start()] + " " + text[m.end():]
            removed += 1
            continue

        after = text[lm.end():]
        after_l = after.casefold()
        lookup20, lookup10 = title_lookup
        key = after_l[:20]
        end_rel = None
        for title in lookup20.get(key, []):
            if after_l.startswith(title.casefold()):
                end_rel = len(title)
                break

        if end_rel is None:
            # Broader first-10-char title index.
            prefix10 = after_l[:10]
            for title in lookup10.get(prefix10, []):
                if after_l.startswith(title.casefold()):
                    end_rel = len(title)
                    break

        if end_rel is None:
            # Estimate the last recommendation-title length from the preceding
            # list items, which ARE delimited by list markers. This is much safer
            # than deleting an arbitrary fixed number of characters from the
            # beginning of the real article body.
            markers = []
            for j in range(1, n + 1):
                jm = re.compile(rf"-\s*list\s+{j}\s+of\s+{n}", re.IGNORECASE).search(text, m.end())
                if jm:
                    markers.append(jm)
            prev_lengths = []
            if len(markers) == n and n >= 2:
                for j in range(n - 1):
                    item = normalize_spaces(text[markers[j].end():markers[j+1].start()])
                    if 8 <= len(item) <= 240:
                        prev_lengths.append(len(item))

            target_len = int(statistics.median(prev_lengths)) if prev_lengths else 85
            lo = max(15, target_len - 40)
            hi = min(len(after), target_len + 55, 210)

            # Prefer a boundary close to the expected headline length where the
            # following prose starts with a capital, digit, or opening quote.
            candidates = []
            for sm in re.finditer(r"\s+", after[lo:hi]):
                pos = lo + sm.end()
                if pos >= len(after):
                    continue
                nxt = after[pos:pos+1]
                if nxt and (nxt.isupper() or nxt.isdigit() or nxt in "'\"“‘("):
                    candidates.append(pos)
            if candidates:
                end_rel = min(candidates, key=lambda x: abs(x - target_len))
            else:
                # Final conservative fallback: end on a word boundary and remove
                # only roughly one normal headline, never an arbitrary mid-word
                # 180-character slice.
                cap = min(len(after), 125)
                ws = [mm.end() for mm in re.finditer(r"\s+", after[:cap])]
                end_rel = ws[-1] if ws else cap

        text = text[:m.start()] + " " + after[end_rel:]
        removed += 1
    return normalize_spaces(text), removed


def strip_tail_boilerplate(text: str, tail_chars: int = 1800) -> tuple[str, int]:
    if not text:
        return text, 0
    cut_start = max(0, len(text) - tail_chars)
    tail = text[cut_start:]
    earliest = None
    for pat in TAIL_CUTOFF_PATTERNS:
        m = pat.search(tail)
        if m:
            pos = cut_start + m.start()
            if earliest is None or pos < earliest:
                earliest = pos
    count = 0
    if earliest is not None:
        text = text[:earliest]
        count += 1

    cut_start = max(0, len(text) - tail_chars)
    head, tail = text[:cut_start], text[cut_start:]
    for pat in TAIL_SENTENCE_PATTERNS:
        tail, n = pat.subn(" ", tail)
        count += n
    return normalize_spaces(head + tail), count


def normalize_target_attributions(text: str) -> tuple[str, int]:
    """Neutralize source-attribution constructions involving target outlets.

    Applied identically to all samples before deleting outlet names. This avoids
    removal scars such as "told the that" and "'s reporter said".
    """
    total = 0
    targets = [
        (r"BBC(?:\s+News|\s+Verify)?", re.IGNORECASE),
        (r"Al\s*Jazeera(?:\s+English)?", re.IGNORECASE),
        (r"The Guardian|Guardian", 0),
    ]
    verbs = r"said|reported|noted|added|explained|wrote|found|stated"
    for target, flags in targets:
        pat = re.compile(rf"\bReporting from\s+[^,.!?]{{1,80}},\s+{target}(?:['’]s)?\s+[^,.!?]{{0,100}}?\s+({verbs})\b", flags)
        text, n = pat.subn(lambda m: f"The report {m.group(1)}", text); total += n
        pat = re.compile(rf"\b{target}['’]s\s+[^,.!?]{{1,100}}?,\s*reporting from\s+[^,.!?]{{1,80}},\s*({verbs})\b", flags)
        text, n = pat.subn(lambda m: f"The report {m.group(1)}", text); total += n
        pat = re.compile(rf"\b{target}['’]s\s+[^,.!?]{{1,80}}?\s+({verbs})\b", flags)
        text, n = pat.subn(lambda m: f"The report {m.group(1)}", text); total += n
        pat = re.compile(rf"\btold\s+(?:the\s+)?{target}\s+that\b", flags)
        text, n = pat.subn("said that", text); total += n
        pat = re.compile(rf"\btold\s+(?:the\s+)?{target}\b", flags)
        text, n = pat.subn("said", text); total += n
        pat = re.compile(rf"\baccording to\s+(?:the\s+)?{target}\b", flags)
        text, n = pat.subn("according to reports", text); total += n
        pat = re.compile(rf"\bspeaking to\s+(?:the\s+)?{target}\b", flags)
        text, n = pat.subn("speaking to reporters", text); total += n
    return text, total


def preclean_for_relevance(text: str, title_lookup: dict[str, list[str]]) -> str:
    text = repair_mojibake(text)
    text, _ = normalize_target_attributions(text)
    text, _ = strip_recommended_stories(text, title_lookup)
    for pat in LEADING_ARTIFACT_PATTERNS:
        text = pat.sub("", text, count=1)
    text = BBC_EXTERNAL_ANNOTATION_RE.sub(", ", text)
    text, _ = strip_tail_boilerplate(text)
    text, _ = strip_global_promos(text)
    return normalize_spaces(text)


# ---------------------------------------------------------------------------
# Page type and topic relevance
# ---------------------------------------------------------------------------


def eligible_page_type(portal: str, url: str) -> tuple[bool, str]:
    path = urlparse(url or "").path.lower()

    if portal == "bbc":
        if re.match(r"^/news/articles/[a-z0-9]+/?$", path):
            return True, "bbc_standard_news_article"
        return False, "bbc_nonstandard_page_type"

    if portal == "aljazeera":
        # Keep dated written reporting, including /features/. Al Jazeera exposes
        # long-form reporting in a separate URL category while BBC/Guardian do
        # not provide an equivalent clean split; excluding it solely for AJ would
        # itself create a portal-dependent genre rule. Liveblogs, video, opinions,
        # programs and section roots remain excluded.
        if re.match(r"^/(?:news|economy|features)/20\d{2}/\d{1,2}/\d{1,2}/[^/]+/?$", path):
            return True, "aj_standard_dated_written_article"
        return False, "aj_nonstandard_page_type"

    if portal == "guardian":
        parts = [p for p in path.split("/") if p]
        if not parts or parts[0] not in GUARDIAN_ALLOWED_SECTIONS:
            return False, "guardian_non_news_or_disallowed_section"
        # Legacy: /section/2025/jan/01/slug
        # Newer:  /section/article/2024/aug/01/slug
        if len(parts) >= 5 and re.fullmatch(r"20\d{2}", parts[1]):
            return True, "guardian_standard_dated_article"
        if len(parts) >= 6 and parts[1] == "article" and re.fullmatch(r"20\d{2}", parts[2]):
            return True, "guardian_standard_dated_article"
        return False, "guardian_nonstandard_page_type"

    return False, "unknown_portal"


def eligible_genre(portal: str, raw_title: str) -> tuple[bool, str]:
    """Reject obvious non-comparable editorial formats not encoded in URL alone.

    Rules that concern title content are applied to every portal first; portal-
    specific rules below only identify site-specific genre conventions.
    """
    t = normalize_spaces(repair_mojibake(raw_title or ""))

    # A target outlet named in the headline makes cleaning unusually destructive
    # and usually indicates a media-about-media story. Exclude these globally
    # rather than leave a direct or inverse identity cue.
    target_title_re = re.compile(
        r"\bBBC(?:\s+News|\s+Verify)?\b|\bAl\s*Jazeera(?:\s+English)?\b|\bThe\s+Guardian\b|\bGuardian\b",
        re.IGNORECASE,
    )
    if target_title_re.search(t):
        return False, "target_outlet_named_in_headline"

    # Explicit opinion/analysis labels are a different genre from ordinary news
    # reporting. The same rule applies to all portals.
    if re.match(r"^(?:Analysis|Opinion|Commentary)\s*:", t, re.IGNORECASE):
        return False, "explicit_analysis_or_opinion_title"

    if portal == "guardian":
        if re.search(r"\|\s*Letters?\s*$", t, re.IGNORECASE):
            return False, "guardian_letters"
        if re.search(r"\bobituary\b", t, re.IGNORECASE):
            return False, "guardian_obituary"
        if re.match(r"^(?:Morning Mail|Afternoon Update|First Thing)\s*:", t, re.IGNORECASE):
            return False, "guardian_newsletter_or_digest"
        if re.match(r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) briefing\s*:", t, re.IGNORECASE):
            return False, "guardian_newsletter_or_digest"
        if re.search(r"\|\s*First Thing\s*$", t, re.IGNORECASE):
            return False, "guardian_newsletter_or_digest"
        if re.match(r"^(?:The Crunch|Five Great Reads)\s*:", t, re.IGNORECASE):
            return False, "guardian_newsletter_or_digest"
        if re.search(r"\binside the .*Weekly\b", t, re.IGNORECASE):
            return False, "guardian_weekly_digest"
        # Guardian headline suffixes after a vertical bar overwhelmingly denote
        # columns/commentary/newsletters (e.g. "| First Thing", "| Author").
        # Remove the format rather than let genre distribution identify portal.
        if " | " in t:
            return False, "guardian_column_or_digest_pipe_title"
        if re.search(r"\bwe would like to hear from you\b", t, re.IGNORECASE):
            return False, "guardian_reader_callout"
        if re.search(r"\bin pictures\s*$", t, re.IGNORECASE):
            return False, "guardian_picture_gallery"
    return True, "ordinary_written_reporting"


def eligible_genre(portal: str, raw_title: str) -> tuple[bool,str]:
    t=normalize_spaces(repair_mojibake(raw_title or ''))
    if re.search(r"\bBBC(?:\s+News|\s+Verify)?\b|\bAl\s*Jazeera(?:\s+English)?\b|\bThe\s+Guardian\b",t,re.I):
        return False,'target_outlet_named_in_headline'
    if re.match(r"^(?:Analysis|Opinion|Commentary)\s*:",t,re.I):
        return False,'explicit_analysis_or_opinion_title'
    if re.match(r"^(?:Fact[- ]?check|Newspaper headlines?|The Papers|What the papers say)\b",t,re.I):
        return False,'factcheck_or_newspaper_digest_format'
    if re.search(r"\b(?:administration briefing|news at a glance)\b",t,re.I):
        return False,'news_briefing_or_digest_format'
    if portal=='guardian':
        if re.search(r"\|\s*Letters?\s*$",t,re.I): return False,'guardian_letters'
        if re.search(r"\bobituary\b",t,re.I): return False,'guardian_obituary'
        if re.match(r"^(?:Morning Mail|Afternoon Update|First Thing)\s*:",t,re.I): return False,'guardian_newsletter_or_digest'
        if re.match(r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) briefing\s*:",t,re.I): return False,'guardian_newsletter_or_digest'
        if re.search(r"\|\s*First Thing\s*$",t,re.I): return False,'guardian_newsletter_or_digest'
        if re.match(r"^(?:The Crunch|Five Great Reads)\s*:",t,re.I): return False,'guardian_newsletter_or_digest'
        if re.search(r"\binside the .*Weekly\b",t,re.I): return False,'guardian_weekly_digest'
        if ' | ' in t: return False,'guardian_column_or_digest_pipe_title'
        if re.search(r"\bwe would like to hear from you\b",t,re.I): return False,'guardian_reader_callout'
        if re.search(r"\bin pictures\s*$",t,re.I): return False,'guardian_picture_gallery'
    return True,'ordinary_written_reporting'


def topic_a_relevant(title: str, body: str) -> tuple[bool, str]:
    """High-precision Topic A with headline-level anchoring."""
    lead = first_n_words(body, TOPIC_LEAD_WORDS)
    direct = re.search(
        r"\b(?:gaza|hamas|palestin(?:e|ian|ians)|west bank|idf|netanyahu|israel|israeli|haniyeh|sinwar|hezbollah)\b",
        title, re.IGNORECASE,
    )
    if direct:
        return True, "A_conflict_anchor_in_title"
    secondary = re.search(r"\b(?:lebanon|iran|houthi|yemen|ceasefire|truce|hostages?|captives?|middle east)\b", title, re.IGNORECASE)
    lead_anchor = re.search(r"\b(?:gaza|hamas|palestin(?:e|ian|ians)|west bank|israel|israeli|hezbollah)\b", lead, re.IGNORECASE)
    if secondary and lead_anchor:
        return True, "A_regional_title_plus_conflict_anchor_in_lead"
    return False, "A_headline_not_sufficiently_anchored_to_target_conflict"


def topic_b_relevant(title: str, body: str, portal: str, url: str) -> tuple[bool, str]:
    """High-precision economic/business coverage validation.

    The original discovery query is only a candidate label. We revalidate using
    the headline and the first 180 content words. This deliberately fixes cases
    such as cooking-oil pages or long political stories that mention an economic
    keyword only much later.
    """
    if ECON_TITLE_RE.search(title):
        return True, "B_economic_or_business_concept_in_title"

    lead = first_n_words(body, TOPIC_LEAD_WORDS)
    group_hits = sum(1 for p in ECON_GROUP_RE.values() if p.search(lead))

    # If the headline itself is indirect, require evidence from at least three
    # independent economic concept groups in the lead. Manual audit showed that
    # two groups can still occur incidentally in broad political/geopolitical
    # stories; three is a safer high-precision fallback.
    if group_hits >= 3:
        return True, "B_multiple_economic_concepts_in_lead"

    path = urlparse(url or "").path.lower()
    dedicated = (
        (portal == "aljazeera" and path.startswith("/economy/")) or
        (portal == "guardian" and (path.startswith("/business/") or path.startswith("/money/")))
    )
    if dedicated and group_hits >= 1:
        return True, "B_dedicated_economy_section_with_economic_lead"

    return False, "B_not_economic_in_headline_or_lead"


def topic_relevant(topic: str, title: str, body: str, portal: str, url: str) -> tuple[bool, str]:
    if topic == "topicA":
        return topic_a_relevant(title, body)
    if topic == "topicB":
        return topic_b_relevant(title, body, portal, url)
    return False, "unknown_topic"



# ---------------------------------------------------------------------------
# FINAL POLICY OVERRIDES (2026-08-27)
# ---------------------------------------------------------------------------
# These rules supersede the earlier draft rules above. They are kept together so
# the final corpus policy is auditable. Page-eligibility may use portal-specific
# URL taxonomies; all text sanitation below is global/label-blind.

GUARDIAN_ALLOWED_SECTIONS = {
    "world", "us-news", "australia-news", "politics", "uk-news", "business",
    "environment", "society", "technology", "global-development", "media",
    "law", "education", "science", "news",
}

# Target brands and owned services/programmes. Longer/specific phrases are
# neutralized before parent brand names are considered. Ambiguous bare programme
# names are touched only in media/source context.
TARGET_OUTLET_RE = re.compile(
    r"\b(?:BBC(?:\s+News)?|Al\s*Jazeera(?:\s+English)?|The\s+Guardian)\b",
    re.IGNORECASE,
)
TARGET_DOMAIN_RE = re.compile(
    r"\b(?:www\.)?(?:bbc\.(?:com|co\.uk)|aljazeera\.com|theguardian\.com)\b",
    re.IGNORECASE,
)
BARE_GUARDIAN_MEDIA_RE = re.compile(
    r"\b(?:according to|told|speaking to|interview(?:ed)? (?:by|with)|reported by|writing for|"
    r"article (?:in|for)|newspaper)\s+(?:the\s+)?Guardian\b|"
    r"\bGuardian\s+(?:newspaper|reporter|correspondent|journalist|reported|said|wrote)\b"
)

TARGET_SERVICE_EXPLICIT_PATTERNS = [
    # BBC
    r"\b(?:the\s+)?BBC\s+Radio\s+(?:4|Four)(?:['’]s)?\s*Today\s+(?:programme|program)\b",
    r"\b(?:the\s+)?BBC(?:['’]s)?\s+Today\s+(?:programme|program)\b",
    r"\b(?:the\s+)?BBC\s+Arabic(?:['’]s)?\s+Gaza\s+Today\s+(?:programme|program)\b",
    r"\b(?:the\s+)?BBC(?:['’]s)?\s+Business\s+Today\s+(?:programme|program)\b",
    r"\bRadio\s+(?:4|Four)(?:['’]s)?\s+Today\s+(?:programme|program)\b",
    r"\bBBC\s+Radio\s+4\b", r"\bBBC\s+Radio\s+5\s+Live\b",
    r"\bBBC\s+World\s+Service\b", r"\bBBC\s+Verify\b", r"\bBBC(?:['’]s)?\s+Newsnight\b",
    r"\bBBC\s+Panorama\b", r"\bBBC\s+Eye\b", r"\bBBC\s+Arabic\b",
    r"\bBBC\s+iPlayer\b", r"\bBBC\s+Sounds\b", r"\bBBC\s+Breakfast\b",
    r"\bHARDtalk\b",
    # Al Jazeera
    r"\bAl\s*Jazeera\s+Arabic\b", r"\bAl\s*Jazeera\s+Mubasher\b",
    r"\bAl\s*Jazeera(?:['’]s)?\s+UpFront\b", r"\bAl\s*Jazeera(?:['’]s)?\s+The\s+Take\b",
    r"\bAl\s*Jazeera(?:['’]s)?\s+Fault\s+Lines\b", r"\bAl\s*Jazeera(?:['’]s)?\s+Inside\s+Story\b",
    r"\bAl\s*Jazeera(?:['’]s)?\s+Counting\s+the\s+Cost\b",
    r"\bAl\s*Jazeera(?:['’]s)?\s+The\s+Listening\s+Post\b",
    r"\bAl\s*Jazeera(?:['’]s)?\s+The\s+Stream\b",
    r"\bAl\s*Jazeera(?:['’]s)?\s+People\s*&\s*Power\b", r"\bAl\s*Jazeera(?:['’]s)?\s+101\s+East\b",
    # Guardian
    r"\bThe\s+Guardian(?:['’]s)?\s+Today\s+in\s+Focus\b",
    r"\bThe\s+Guardian(?:['’]s)?\s+First\s+Edition\b",
    r"\bThe\s+Guardian(?:['’]s)?\s+First\s+Thing\b",
    r"\bThe\s+Guardian(?:['’]s)?\s+Politics\s+Weekly\b",
    r"\bThe\s+Guardian(?:['’]s)?\s+Full\s+Story\b",
    r"\bThe\s+Guardian(?:['’]s)?\s+Long\s+Read\b",
]
TARGET_SERVICE_EXPLICIT_RE = compile_any(TARGET_SERVICE_EXPLICIT_PATTERNS)
TARGET_SERVICE_BARE_NAMES = [
    r"Newsnight", r"Panorama", r"UpFront", r"The\s+Take", r"Fault\s+Lines",
    r"Inside\s+Story", r"Counting\s+the\s+Cost", r"The\s+Listening\s+Post",
    r"The\s+Stream", r"People\s*&\s*Power", r"101\s+East", r"Today\s+in\s+Focus",
    r"First\s+Edition", r"Politics\s+Weekly", r"Full\s+Story", r"The\s+Long\s+Read",
]
TARGET_SERVICE_BARE_ALT = r"(?:" + "|".join(TARGET_SERVICE_BARE_NAMES) + r")"
TARGET_SERVICE_MEDIA_CONTEXT_RE = re.compile(
    rf"(?:\b(?:told|speaking\s+(?:to|on)|appeared\s+on|according\s+to|interview(?:ed)?\s+(?:by|with)|"
    rf"reported\s+by|heard\s+on|on)\s+(?:the\s+)?{TARGET_SERVICE_BARE_ALT}\b)|"
    rf"(?:\b{TARGET_SERVICE_BARE_ALT}\s+(?:programme|program|podcast|show|reported|said|investigation|interview)\b)",
    re.IGNORECASE,
)

# Sanad is Al Jazeera's verification/fact-checking unit, but the bare token can
# also occur in the name of an unrelated Sanad News Agency. Treat it as target
# identity only in unambiguous verification/analysis grammar.
SANAD_TARGET_CONTEXT_RE = re.compile(
    r"(?:\b(?:according to|obtained by|reviewed by|analysed by|analyzed by|verified by|authenticated by)\s+Sanad\b)|"
    r"(?:\bSanad\s+(?:also\s+)?(?:has\s+)?(?:found|verified|analysed|analyzed|documented|authenticated|shows?|concluded|estimates?)\b)|"
    r"(?:\bSanad['’]s\s+(?:analysis|investigation|verification)\b)|"
    r"(?:\b(?:fact-checking|verification)\s+(?:agency|unit|team)\s+Sanad\b)",
    re.I,
)

WIRE_FULL_ALT = r"(?:Reuters|Associated\s+Press|Agence\s+France[- ]Presse)"
WIRE_ABBREV_ALT = r"(?:AP|AFP)"
WIRE_FULL_RE = re.compile(rf"\b{WIRE_FULL_ALT}\b", re.IGNORECASE)
WIRE_NEWS_AGENCY_FORM_RE = re.compile(
    rf"\b(?:{WIRE_FULL_ALT})(?:\s+news\s+agency)?\b", re.IGNORECASE
)

# Topic A: distinguish high-specificity conflict anchors from the broader word
# Palestine/Palestinian, which produced domestic protest/legal false positives.
A_HIGH_RE = re.compile(
    r"\b(?:gaza|hamas|west bank|idf|netanyahu|israel|israeli|sinwar|haniyeh)\b",
    re.IGNORECASE,
)
A_PALESTINE_RE = re.compile(r"\bpalestin(?:e|ian|ians)\b", re.IGNORECASE)
A_REGIONAL_RE = re.compile(r"\b(?:hezbollah|lebanon|iran|houthis?|yemen)\b", re.IGNORECASE)
A_LEBANON_HEZ_RE = re.compile(r"\bhezbollah\b", re.IGNORECASE)
A_CONFLICT_GROUPS_FINAL = {
    "military": compile_any([
        r"\bwar\b", r"\bconflict\b", r"\battack(?:s|ed|ing)?\b", r"\bairstrikes?\b",
        r"\bstrikes?\b", r"\bbomb(?:ing|ed|s)?\b", r"\boffensive\b", r"\bmilitary\b",
        r"\barmy\b", r"\btroops?\b", r"\bfighting\b", r"\bkilled\b", r"\bdead\b",
    ]),
    "ceasefire": compile_any([r"\bceasefire\b", r"\btruce\b", r"\bnegotiat(?:e|es|ed|ions|ing)\b", r"\bpeace talks?\b", r"\bdeal\b"]),
    "hostages": compile_any([r"\bhostages?\b", r"\bcaptives?\b", r"\bprisoner exchange\b", r"\brelease(?:d|s)?\b"]),
    "humanitarian": compile_any([r"\bhumanitarian\b", r"\baid\b", r"\bfamine\b", r"\bhunger\b", r"\bdisplaced\b", r"\bcivilians?\b", r"\bhospital\b"]),
    "occupation_legal_political": compile_any([
        r"\boccupation\b", r"\bsettlements?\b", r"\bannex(?:ation|ed)?\b", r"\bstatehood\b",
        r"\bPalestinian state\b", r"\btwo-state\b", r"\bICC\b", r"\bICJ\b", r"\bUNRWA\b",
        r"\bNakba\b", r"\brecognis(?:e|es|ed|ing|ation)\b", r"\brecogniz(?:e|es|ed|ing|ation)\b",
    ]),
}
A_EXPLICIT_TITLE_RE = compile_any([
    r"\bIsrael[- ]Hamas\b", r"\bGaza\s+(?:war|ceasefire|truce|attack|airstrike|offensive|aid|hostage)\b",
    r"\bIsraeli\s+(?:attack|airstrike|strike|offensive|military|army|troops?)\b",
    r"\bHamas\s+(?:hostage|ceasefire|attack|deal|war)\b", r"\bWest Bank\s+(?:attack|raid|settlement|violence)\b",
])
A_DIPLOMACY_TITLE_RE = compile_any([
    r"\brecogn(?:ise|ize|ition|ised|ized|ising|izing)\s+(?:a\s+)?Palestinian state\b",
    r"\bPalestinian\s+state(?:hood)?\b", r"\btwo-state solution\b", r"\bUNRWA\b",
    r"\bNakba\b", r"\b(?:ICC|ICJ)\b.*\bIsrael\b", r"\bIsrael\b.*\b(?:ICC|ICJ)\b",
])
A_SECONDARY_DOMAIN_RE = compile_any([
    r"\b(?:Emmys?|Oscars?|awards?|actor|actress|film|movie|television|TV|music|singer|football|soccer|match|recipe|food|university|campus)\b",
])
A_REGIONAL_TITLE_CONTEXT_RE = compile_any([
    r"\b(?:attack|strike|airstrike|bomb|war|conflict|ceasefire|truce|fighting|killed|dead|rocket|missile|pager|evacuat(?:e|es|ed|ion)|Hezbollah|Houthi)\w*\b",
])

# Topic B: strong headline concepts plus context-sensitive ambiguous concepts.
ECON_GROUPS_FINAL = {
    "macro": compile_any([
        r"\binflation\b", r"\bdeflation\b", r"\bGDP\b", r"\bgross domestic product\b",
        r"\brecession\b", r"\beconomic growth\b", r"\beconom(?:y|ic)\b", r"\bcost[- ]of[- ]living\b",
        r"\bconsumer prices?\b", r"\bprices? (?:rise|rises|rose|rising|fall|falls|fell|falling|increase|increases|increased|decrease|decreases|decreased|higher|lower)\b", r"\bproductivity\b",
    ]),
    "monetary": compile_any([
        r"\binterest rates?\b", r"\bcentral bank\b", r"\bmonetary policy\b", r"\bfederal reserve\b",
        r"\bthe fed\b", r"\bECB\b", r"\bbank of england\b", r"\brate cuts?\b", r"\brate hikes?\b",
    ]),
    "labour": compile_any([
        r"\bunemployment\b", r"\bjobless\b", r"\bjobs? (?:market|growth|creation|cuts?|losses?)\b",
        r"\blabou?r market\b", r"\bwage growth\b", r"\bminimum wage\b", r"\bliving wage\b",
        r"\b(?:rising|falling|real) wages\b", r"\bpension reform\b", r"\bretirement age\b",
    ]),
    "markets": compile_any([
        r"\bstock markets?\b", r"\bfinancial markets?\b", r"\bbond yields?\b", r"\btreasury yields?\b",
        r"\bWall Street\b", r"\bS&P 500\b", r"\bDow Jones\b", r"\bNasdaq\b", r"\bshare prices?\b",
        r"\bmarkets? (?:rally|rallies|rallied|rise|rises|rose|fall|falls|fell|slump|surge|turmoil|sell[- ]?off)\b",
    ]),
    "fiscal_trade": compile_any([
        r"\btariffs?\b", r"\btrade (?:war|deal|agreement|talks?|tensions?|policy|deficit|surplus)\b",
        r"\bfree trade\b", r"\binternational trade\b", r"\bfiscal policy\b", r"\bbudget deficit\b",
        r"\bpublic debt\b", r"\bnational debt\b", r"\bgovernment debt\b", r"\bsovereign debt\b",
        r"\bgovernment spending\b", r"\bborrowing costs?\b", r"\btax (?:cuts?|rises?|increases?|policy)\b",
        r"\blocal government debt\b", r"\bhidden debt\b",
    ]),
    "energy_econ": compile_any([
        r"\boil prices?\b", r"\bprice of oil\b", r"\bcrude oil\b", r"\bBrent\b", r"\bOPEC\+?\b",
        r"\boil exports?\b", r"\bbarrels? per day\b", r"\benergy prices?\b", r"\bgas prices?\b",
    ]),
    "business_finance": compile_any([
        r"\bearnings\b", r"\bprofits?\b", r"\bcompany revenue\b", r"\bquarterly revenue\b",
        r"\binvestors?\b", r"\binvestment\b", r"\bbankrupt(?:cy)?\b", r"\binsolvency\b",
        r"\bretail sales?\b", r"\bcrypto(?:currency|currencies)?\b", r"\bbitcoin\b", r"\bwealth (?:inequality|divide|gap|disparity)\b", r"\bincome inequality\b",
    ]),
    "currency_banking": compile_any([
        r"\bcurrenc(?:y|ies)\b", r"\bsterling\b", r"\b(?:US|U\.S\.|Australian|Canadian) dollar\b",
        r"\b(?:dollar|euro|yen|yuan|rupee|pound) (?:rises?|falls?|rose|fell|sinks?|tumbles?|slides?|surges?|weakens?|strengthens?)\b",
        r"\bbanking\b", r"\bbank lending\b", r"\bmortgage rates?\b", r"\bmortgage costs?\b",
        r"\bIMF\b", r"\bWorld Bank\b",
    ]),
}
ECON_STRONG_TITLE_RE = compile_any([
    r"\binflation\b", r"\bdeflation\b", r"\bGDP\b", r"\brecession\b", r"\beconomic growth\b",
    r"\bunemployment\b", r"\bjobless\b", r"\bcost[- ]of[- ]living\b", r"\binterest rates?\b",
    r"\bmonetary policy\b", r"\brate cuts?\b", r"\brate hikes?\b", r"\btariffs?\b",
    r"\btrade (?:war|deficit|surplus|deal|agreement|policy)\b", r"\bfiscal policy\b", r"\bbudget deficit\b",
    r"\bpublic debt\b", r"\bnational debt\b", r"\bgovernment debt\b", r"\bsovereign debt\b",
    r"\bstock markets?\b", r"\bfinancial markets?\b", r"\bbond yields?\b", r"\boil prices?\b",
    r"\bcrude oil\b", r"\bOPEC\+?\b", r"\bearnings\b", r"\bprofits?\b", r"\bIMF\b", r"\bWorld Bank\b",
    r"\bretail sales?\b", r"\bwealth (?:inequality|divide|gap|disparity)\b", r"\bincome inequality\b",
    r"\beconomy\b", r"\beconomic (?:growth|crisis|outlook|recovery|downturn|slowdown|activity|performance|policy|conditions|woes|uncertainty|damage|impact|forecast|prospects)\b", r"\bjobs\b", r"\bjob (?:cuts?|losses?|market|creation)\b", r"\blayoffs?\b", r"\bbusiness confidence\b",
    r"\bconsumer confidence\b", r"\bbailout\b", r"\bcurrency swap\b", r"\bfreeports?\b",
    r"\bglobali[sz]ation\b", r"\bcrypto(?:currency|currencies)?\b", r"\bbitcoin\b",
    r"\bstagflation\b", r"\bconsumer prices?\b", r"\banti[- ]dumping\b", r"\boil exports?\b",
    r"\b(?:public|state|government|sovereign) pension(?:s| reform)?\b", r"\b(?:US|U\.S\.|UK|British|Indian|Chinese|Australian|Canadian) debt\b",
    r"\bbudget\b", r"\bmanufacturing\b",
])
ECON_CONTEXT_TITLE_RE = compile_any([
    r"\boil (?:market|exports?)\b", r"\bBrent\b", r"\bmortgage (?:rates?|costs?)\b",
    r"\b(?:stock|financial|housing|property|currency|oil|energy|labou?r) markets?\b",
    r"\bmarkets? (?:rally|rallies|rallied|rise|rises|rose|fall|falls|fell|slump|surge|turmoil|sell[- ]?off)\b",
    r"\bcompany revenue\b", r"\brevenue (?:rose|fell|rises|falls|up|down|growth|decline)\b",
    r"\b(?:US|U\.S\.|Australian|Canadian) dollar\b", r"\bsterling\b",
    r"\b(?:dollar|euro|yen|yuan|rupee|pound) (?:rises?|falls?|rose|fell|sinks?|tumbles?|slides?|surges?|weakens?|strengthens?)\b",
    r"\bgold (?:prices?|rises?|falls?|rose|fell|surges?|slumps?|record|market)\b",
    r"\bprice of gold\b", r"\bbullion\b", r"\bgold and silver\b",
    r"\bwage growth\b", r"\bminimum wage\b", r"\bliving wage\b", r"\b(?:rising|falling|real) wages\b", r"\bpension reform\b", r"\bretirement age\b",
    r"\bjobs? (?:market|growth|creation|cuts?|losses?)\b", r"\btrade (?:talks?|tensions?|relations?)\b",
    r"\bhousing\b", r"\bproperty\b", r"\bborrowing costs?\b",
    r"\bstocks? (?:tumble|tumbles|tumbled|rise|rises|rose|fall|falls|fell|rally|rallies|rallied|surge|surges|surged|slump|slumps|slumped)\b",
    r"\bshares? (?:rise|rises|rose|fall|falls|fell|jump|jumps|jumped|drop|drops|dropped|slump|surge)\b",
    r"\b(?:silver|gold) (?:price|prices|record|rally|rises?|falls?|rose|fell|surges?|slumps?)\b",
    r"\bgrowth figures?\b", r"\boil (?:imports?|purchases?|exports?)\b",
    r"\bhome purchases?\b.*\binstitutional investors?\b",
])
ECON_WEAK_TITLE_RE = compile_any([
    r"\binvestments?\b", r"\b(?:company|companies|business(?:es)?|industry|industries)\b",
    r"\b(?:exports?|imports?)\b", r"\b(?:wages?|salar(?:y|ies)|workers?)\b", r"\blabou?r (?:market|force|costs?|shortages?|codes?)\b",
    r"\b(?:prices?|consumer spending|tax(?:es|ation)?|debt|deficit|bonds?|shares?|stocks?)\b",
    r"\b(?:loans?|credit|rent|housing|property)\b", r"\b(?:factory|factories)\b",
    r"\beconomic (?:crisis|woes|outlook|recovery|downturn|slowdown|growth|activity|performance|uncertainty|damage|impact|forecast|prospects)\b",
])
ECON_NEGATIVE_MASK_RE = compile_any([
    r"\bfuel economy\b", r"\beconomy class\b", r"\bolive oil\b", r"\bcooking oil\b", r"\bseed oils?\b",
    r"\bIrish Revenue\b", r"\bRevenue Commissioners?\b", r"\bmortgage fraud\b", r"\btrade union\b",
    r"\bChristmas market\b", r"\bflea market\b", r"\bfood market\b", r"\bfish stocks?\b", r"\bstock images?\b",
    r"\bdefen[cs]e budget\b", r"\bmilitary budget\b", r"\bdefen[cs]e spending\b", r"\bmilitary spending\b",
    r"\bblack tax\b", r"\bgold(?:en)? visa\b", r"\bwages war\b",
])
LEGAL_CRIMINAL_TITLE_RE = compile_any([
    r"\bfraud\b", r"\bcriminal investigation\b", r"\bcriminal probe\b", r"\blawsuit\b", r"\bcourt\b",
    r"\bindict(?:ed|ment)?\b", r"\barrest(?:ed|s)?\b", r"\bjail(?:ed)?\b", r"\bprison\b", r"\bcharged with\b",
    r"\bcharges?\b", r"\bprobe\b", r"\binvestigation\b", r"\bscam\b",
])
INDEPENDENT_ECON_POLICY_RE = compile_any([
    r"\btariffs?\b", r"\btrade (?:war|deficit|surplus|deal|agreement|policy)\b", r"\binflation\b", r"\bGDP\b",
    r"\beconomic growth\b", r"\brecession\b", r"\binterest rates?\b", r"\bmonetary policy\b",
    r"\bstock markets?\b", r"\bfinancial markets?\b", r"\bpublic debt\b", r"\bnational debt\b",
    r"\bgovernment debt\b", r"\bborrowing costs?\b", r"\bconsumer prices?\b",
])


def _count_group_hits(text: str, groups: dict[str, re.Pattern]) -> int:
    return sum(1 for p in groups.values() if p.search(text or ""))


def _target_mentions(text: str) -> int:
    return len(A_HIGH_RE.findall(text or "")) + len(A_PALESTINE_RE.findall(text or ""))


def eligible_page_type(portal: str, url: str) -> tuple[bool, str]:
    path = urlparse(url or "").path.lower()
    if portal == "bbc":
        return (True, "bbc_standard_news_article") if re.match(r"^/news/articles/[a-z0-9]+/?$", path) else (False, "bbc_nonstandard_page_type")
    if portal == "aljazeera":
        if re.match(r"^/(?:news|economy|features)/20\d{2}/\d{1,2}/\d{1,2}/[^/]+/?$", path):
            return True, "aj_standard_dated_written_article"
        return False, "aj_nonstandard_page_type"
    if portal == "guardian":
        parts=[x for x in path.split('/') if x]
        if not parts or parts[0] not in GUARDIAN_ALLOWED_SECTIONS:
            return False, "guardian_non_news_or_disallowed_section"
        if len(parts)>=5 and re.fullmatch(r"20\d{2}",parts[1]): return True,"guardian_standard_dated_article"
        if len(parts)>=6 and parts[1]=='article' and re.fullmatch(r"20\d{2}",parts[2]): return True,"guardian_standard_dated_article"
        return False,"guardian_nonstandard_page_type"
    return False,"unknown_portal"


def topic_a_relevant(title: str, body: str) -> tuple[bool, str]:
    title=normalize_spaces(title); lead=first_n_words(body,TOPIC_LEAD_WORDS)
    high_title=bool(A_HIGH_RE.search(title)); pal_title=bool(A_PALESTINE_RE.search(title))
    lead_has_target=bool(A_HIGH_RE.search(lead) or A_PALESTINE_RE.search(lead))
    groups=_count_group_hits(lead,A_CONFLICT_GROUPS_FINAL)

    if re.search(r"\bPalestine Action\b",title,re.I):
        return False,"A_palestine_action_domestic_story"
    domestic = (pal_title and not high_title and re.search(
        r"\b(?:arrest|court|ban|proscription|trial|plead|charge|police|rally|protest|demonstration|flag|activist|solidarity|hunger strike|hunger striker|campus|university)\w*\b",title,re.I
    ))
    if domestic:
        return False,"A_domestic_pro_palestine_protest_or_legal_story"
    if re.search(r"\bSomaliland\b",title,re.I):
        return False,"A_israel_somaliland_recognition_story"
    if re.search(r"\b(?:football|soccer|match|fans?)\b",title,re.I) and not re.search(r"\b(?:Gaza|Hamas|West Bank|Palestin(?:e|ian)|Lebanon|Hezbollah|Iran|Yemen|Syria)\b",title,re.I):
        return False,"A_external_sports_or_fan_incident"
    if re.search(r"\bUkraine\b",title,re.I):
        return False,"A_competing_ukraine_story"
    if re.search(r"\bNetanyahu\b",title,re.I) and re.search(r"\b(?:corruption|bribery|fraud|pardon|corruption trial)\b",title,re.I) and not re.search(r"\b(?:ICC|ICJ|war crimes?|Gaza|Hamas|ceasefire)\b",title,re.I):
        return False,"A_netanyahu_personal_corruption_or_pardon_story"
    if re.search(r"\bambassador to Israel\b",title,re.I) and not re.search(r"\b(?:Gaza|Palestin(?:e|ian)|West Bank|settlements?|ceasefire|war)\b",title,re.I):
        return False,"A_diplomatic_appointment_without_conflict_anchor"
    if re.search(r"\bHamas\b",title,re.I) and re.search(r"\b(?:member|suspect|arrest|plot)\w*\b",title,re.I) and not re.search(r"\b(?:Gaza|Israel|Israeli|West Bank|Palestin(?:e|ian))\b",title,re.I):
        return False,"A_external_hamas_security_story"
    if re.search(r"\bUkraine\b",title,re.I) and not high_title:
        return False,"A_competing_ukraine_story"
    if re.search(r"\b(?:Assad|Syrian opposition|Syrian rebels?|opposition forces)\b",title,re.I) and not high_title:
        return False,"A_competing_syria_story"

    if A_SECONDARY_DOMAIN_RE.search(title):
        if not (high_title and _target_mentions(lead)>=3 and groups>=2):
            return False,"A_secondary_domain_incidental_conflict_mention"

    if A_DIPLOMACY_TITLE_RE.search(title) and lead_has_target and groups>=1:
        return True,"A_diplomacy_legal_statehood_title_plus_lead"
    if high_title:
        if not lead_has_target:
            return False,"A_target_title_without_lead_confirmation"
        if A_EXPLICIT_TITLE_RE.search(title):
            return True,"A_explicit_conflict_title_plus_lead_confirmation"
        if groups>=2:
            return True,"A_target_title_plus_two_conflict_groups_in_lead"
        return False,"A_target_title_insufficient_conflict_centrality"
    if pal_title:
        if _target_mentions(lead)>=2 and groups>=2:
            return True,"A_palestine_title_strong_lead_confirmation"
        return False,"A_bare_palestine_title_insufficient_centrality"
    if A_LEBANON_HEZ_RE.search(title) and A_REGIONAL_TITLE_CONTEXT_RE.search(title) and A_HIGH_RE.search(lead) and groups>=2:
        return True,"A_regional_title_direct_israel_gaza_lead"
    return False,"A_no_sufficient_target_conflict_anchor"


def topic_b_relevant(title: str, body: str, portal: str, url: str) -> tuple[bool, str]:
    title=normalize_spaces(title); lead=first_n_words(body,TOPIC_LEAD_WORDS)
    # Mask known ambiguous non-economic senses before positive matching.
    masked=ECON_NEGATIVE_MASK_RE.sub(" ",title)
    groups=_count_group_hits(lead,ECON_GROUPS_FINAL)

    if re.search(r"\b(?:defen[cs]e|military) (?:budget|spending)\b",title,re.I):
        return False,"B_defence_or_military_budget_story"
    # Legal/criminal stories need an independent economic policy/market concept;
    # mere money/investment/bank vocabulary is not enough.
    if LEGAL_CRIMINAL_TITLE_RE.search(title) and not INDEPENDENT_ECON_POLICY_RE.search(title):
        return False,"B_legal_criminal_story_without_independent_economic_signal"
    # Personnel/scandal stories around a central-bank official are not monetary
    # policy merely because the institution is named.
    if re.search(r"\b(?:Fed|Federal Reserve|central bank|Bank of England)\b",title,re.I) and LEGAL_CRIMINAL_TITLE_RE.search(title) and not re.search(r"\b(?:interest rates?|monetary policy|rate cuts?|rate hikes?|inflation)\b",title,re.I):
        return False,"B_central_bank_personnel_legal_story"

    if re.search(r"\b(?:Federal Reserve|Fed|central bank|Bank of England)\b",title,re.I) and re.search(r"\b(?:fire|fired|remove|removal|takeover|take over|fight|chair|chief)\b",title,re.I) and not re.search(r"\b(?:interest rates?|monetary policy|rate cuts?|rate hikes?|inflation|econom(?:y|ic))\b",title,re.I):
        return False,"B_central_bank_personnel_or_independence_story_without_policy_anchor"
    if re.search(r"\b(?:Federal Reserve|Fed|central bank|Bank of England)\b",title,re.I) and re.search(r"\b(?:governor|chair|chief|property|mortgage|home|files?|ethics|scandal|accus\w*|fraud|investigation|probe)\b",title,re.I) and not re.search(r"\b(?:interest rates?|monetary policy|rate cuts?|rate hikes?|inflation|econom(?:y|ic)|jobs?|unemployment)\b",title,re.I):
        return False,"B_central_bank_personal_or_scandal_story_without_policy_anchor"
    if re.search(r"\b(?:IMF|World Bank)\b",title,re.I) and re.search(r"\b(?:leav(?:e|es|ing)|depart(?:s|ing|ure)?|resign(?:s|ed|ing|ation)?|appoint(?:ed|ment)?|named|chief|head|deputy)\b",title,re.I) and not re.search(r"\b(?:economy|economic|inflation|growth|GDP|debt|loan|funding|bailout|forecast|policy)\b",title,re.I):
        return False,"B_international_financial_institution_personnel_story"
    if re.search(r"\b(?:crypto(?:currency)?|bitcoin|Polymarket)\b",title,re.I) and re.search(r"\b(?:gambler|gambling|bet|betting|wager|mission to get back|wallet worth|hacks?|heists?|kidnapp?\w*|theft|stolen|steal|scam|fraud)\b",title,re.I):
        return False,"B_crypto_gambling_or_human_interest_story"
    if ECON_STRONG_TITLE_RE.search(masked) and groups>=1:
        return True,"B_strong_economic_title_plus_lead_confirmation"
    if ECON_CONTEXT_TITLE_RE.search(masked) and groups>=2:
        return True,"B_contextual_economic_title_plus_two_lead_groups"
    if ECON_WEAK_TITLE_RE.search(masked) and groups>=2:
        return True,"B_weak_economic_title_plus_two_lead_groups"
    return False,"B_no_clear_economic_headline_anchor"


def topic_relevant(topic: str, title: str, body: str, portal: str, url: str) -> tuple[bool,str]:
    if topic=='topicA': return topic_a_relevant(title,body)
    if topic=='topicB': return topic_b_relevant(title,body,portal,url)
    return False,'unknown_topic'


def _source_alt() -> str:
    outlet=r"(?:the\s+)?(?:BBC(?:\s+News|\s+Verify)?|Al\s*Jazeera(?:\s+English)?|The\s+Guardian)"
    return rf"(?:{TARGET_SERVICE_EXPLICIT_RE.pattern}|{TARGET_SERVICE_BARE_ALT}|{outlet})"
TARGET_SOURCE_ALT=_source_alt()
TARGET_OUTLET_ONLY_ALT=r"(?:BBC(?:\s+News|\s+Verify)?|Al\s*Jazeera(?:\s+English)?|The\s+Guardian)"
PERSON_NAME_RE_SRC=r"(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,4})"
# V9: strict proper-name patterns used in source-attribution rewrites. Unlike
# the legacy pattern, they cannot cross a sentence boundary by consuming a
# trailing full stop plus the next capitalized words (e.g. 'Resul Serdar. For
# Israel'). A small set of common lowercase name particles is supported.
_PERSON_TOKEN_STRICT_V9 = r"(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|(?:al|el)-[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|(?:al|el|bin|ibn|de|del|van|von|da|di|la|le))"
PERSON_NAME_RE_STRICT_V9 = rf"(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+(?:\s+{_PERSON_TOKEN_STRICT_V9}){{1,4}})"
PERSON_SOURCE_NAME_RE_STRICT_V9 = rf"(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+(?:\s+{_PERSON_TOKEN_STRICT_V9}){{0,4}})"


def normalize_target_attributions(text: str) -> tuple[str,int]:
    total=0; source=TARGET_SOURCE_ALT

    # V9: handle possessive outlet attributions BEFORE the generic source rule.
    # Otherwise "according to Al Jazeera's X" becomes the malformed
    # "according to reports's X".  Match reporter names case-sensitively while
    # matching the outlet/role grammar case-insensitively; using re.I on the
    # whole person regex would turn ordinary lowercase words into "names".
    outlet_ci = rf"(?i:(?:the\s+)?{TARGET_OUTLET_ONLY_ALT})"
    person2 = PERSON_NAME_RE_STRICT_V9
    role_ci = (r"(?:(?i:senior|chief|international|foreign|diplomatic|political|economic|business|Middle\s+East)\s+)*"
               r"(?i:correspondent|reporter|journalist)")
    relative_reporting = (r"(?:,\s*(?:(?i:who\s+is\s+)?(?i:reporting|speaking)\s+from\s+[^,.!?]{1,80}))?")

    special = [
        (rf"\b(?i:according to)\s+{outlet_ci}['’]s\s+(?i:correspondents?)\s+(?i:on the ground)\b",
         "according to reports from the ground"),
        (rf"\b(?i:according to)\s+{outlet_ci}['’]s\s+(?i:tracking of)\b",
         "according to a tally of"),
        (rf"\b(?i:according to)\s+{outlet_ci}\s+(?i:teams?)\s+(?i:on the ground)\b",
         "according to teams reporting on the ground"),
        (rf"\b(?i:according to)\s+{outlet_ci}['’]s\s+{role_ci}\s+{person2}\b{relative_reporting}",
         "according to reports"),
        (rf"\b(?i:according to)\s+{outlet_ci}['’]s\s+{person2}\b{relative_reporting}",
         "according to reports"),
    ]
    for pat,repl in special:
        text,n=re.subn(pat,repl,text); total+=n

    rules=[
        (rf"\btold\s+{source}\s+that\b","said that"), (rf"\btold\s+{source}\b","said"),
        (rf"\baccording to\s+{source}\b","according to reports"),
        (rf"\bspeaking to\s+{source}\b","speaking to reporters"),
        (rf"\bspeaking on\s+{source}\b","speaking in an interview"),
        (rf"\bin an interview with\s+{source}\b","in an interview"),
        (rf"\binterviewed by\s+{source}\b","interviewed by reporters"),
        (rf"\bappeared on\s+{source}\b","appeared in an interview"),
    ]
    for pat,repl in rules:
        text,n=re.subn(pat,repl,text,flags=re.I); total+=n
    # Outlet-owned reporter constructions. Keep outlet/role/verb matching
    # case-insensitive, but proper names case-sensitive and sentence-bounded.
    outlet_only_ci = rf"(?i:{TARGET_OUTLET_ONLY_ALT})"
    reporter_verb_ci = r"(?i:said|reported|wrote|added|noted|explained|found)"
    pat=re.compile(
        rf"\b{outlet_only_ci}['’]s\s+{PERSON_SOURCE_NAME_RE_STRICT_V9}"
        rf"(?:,\s*(?i:reporting|speaking)\s+from\s+[^,.!?]{{1,80}},?)?\s+({reporter_verb_ci})\b"
    )
    text,n=pat.subn(lambda m:'__TARGET_REPORTER_ATTR__',text); total+=n
    pat=re.compile(
        rf"\b{outlet_only_ci}\s+(?i:correspondent|reporter|journalist)\s+"
        rf"{PERSON_SOURCE_NAME_RE_STRICT_V9}\s+({reporter_verb_ci})\b"
    )
    text,n=pat.subn(lambda m:'__TARGET_REPORTER_ATTR__',text); total+=n
    extras=[
        (rf"\bconfirmed to\s+{source}\s+that\b","confirmed that"), (rf"\bconfirmed to\s+{source}\b","confirmed"),
        (rf"\banalysis by\s+{source}\b","analysis"),
        (rf"\bin an? (?:exclusive )?article in\s+{TARGET_OUTLET_ONLY_ALT}\b","in an article"),
        (rf"\bfootage broadcast on\s+{source}\b","broadcast footage"),
    ]
    for pat,repl in extras:
        text,n=re.subn(pat,repl,text,flags=re.I); total+=n
    # Target-owned programme/service used adjectivally with a media object, e.g.
    # "a BBC Panorama investigation". Neutralize the whole branded modifier
    # rather than leave a distinctive programme name or reject an otherwise
    # useful article.
    branded_obj = r"(?:investigation|analysis|report|documentary|interview|survey|poll|programme|program|podcast)"
    text,n=re.subn(
        rf"\b(?:a|an|the)\s+{source}\s+({branded_obj[3:-1]})\b",
        lambda m: ("an " if m.group(1).lower()[0] in "aeiou" else "a ") + m.group(1).lower(),
        text,flags=re.I,
    ); total+=n
    # Source as subject.
    verbs=r"reported|reports|is reporting|said|found|wrote|noted|added|stated|confirmed|revealed|understands|learned|investigated"
    pat=re.compile(rf"\b{source}\s+({verbs})\b",re.I)
    def subj(m):
        v=m.group(1).lower()
        if v in {'reported','reports','is reporting','said','wrote','noted','added','stated','confirmed','revealed'}: return 'reports indicated'
        if v=='found': return 'an investigation found'
        if v=='investigated': return 'an investigation examined'
        return 'reports indicate'
    text,n=pat.subn(subj,text); total+=n
    pat=re.compile(rf"\b{source}['’]s\s+(investigation|analysis|report|documentary|interview|survey|poll)\b",re.I)
    text,n=pat.subn(lambda m:('an ' if m.group(1).lower()[0] in 'aeiou' else 'a ')+m.group(1).lower(),text); total+=n
    return clean_punctuation_gaps(normalize_spaces(text)),total


def normalize_wire_attributions(text: str) -> tuple[str,int]:
    total=0
    full=rf"(?:the\s+)?{WIRE_FULL_ALT}(?:\s+news\s+agency)?"
    abbr=rf"(?:the\s+)?{WIRE_ABBREV_ALT}\s+(?:news\s+agency|newswire)|(?:the\s+)?(?:news\s+agency|newswire)\s+{WIRE_ABBREV_ALT}"
    source=rf"(?:{full}|{abbr})"
    multi=rf"{source}(?:\s*(?:,|and|&)\s*{source})*"

    # V9: preserve the actual eyewitness attribution while removing the wire
    # identity.  Generic "WIRE said" -> "reports indicated" would otherwise
    # leave a dangling pronoun: "Reports indicated its correspondent ...".
    text,n=re.subn(
        rf"\b{source}\s+said\s+its\s+(?:correspondent|reporter|journalist|photographer)\s+"
        rf"(reported|said|wrote|noted|described)\b",
        lambda m: "A correspondent " + m.group(1).lower(),
        text, flags=re.I,
    ); total+=n

    # Bare AP/AFP are considered wire identities only in unmistakable source
    # attribution grammar; they are not replaced as free-standing abbreviations.
    bare_ctx=rf"(?:{WIRE_ABBREV_ALT})"
    for pat,repl in [
        (rf"\btold\s+{multi}\s+that\b","said that"),(rf"\btold\s+{multi}\b","said"),
        (rf"\baccording to\s+{multi}\b","according to reports"),(rf"\bspeaking to\s+{multi}\b","speaking to reporters"),
        (rf"\bin an interview with\s+{multi}\b","in an interview"),(rf"\binterviewed by\s+{multi}\b","interviewed by reporters"),
        (rf"\basked by\s+{multi}\b","asked by reporters"),
        (rf"\btold\s+{bare_ctx}\s+that\b","said that"),(rf"\btold\s+{bare_ctx}\b","said"),
        (rf"\baccording to\s+{bare_ctx}\b","according to reports"),(rf"\bspeaking to\s+{bare_ctx}\b","speaking to reporters"),
        (rf"\bin an interview with\s+{bare_ctx}\b","in an interview"),(rf"\binterviewed by\s+{bare_ctx}\b","interviewed by reporters"),
    ]:
        text,n=re.subn(pat,repl,text,flags=re.I);total+=n
    pat_bare=re.compile(rf"\b{bare_ctx}\s+(reported|reports|said|found|wrote|noted|confirmed|revealed|added|stated|estimated|calculated)\b",re.I)
    def ws_bare(m):
        v=m.group(1).lower()
        if v in {'reported','reports','said','wrote','added','stated'}: return 'reports indicated'
        if v=='found': return 'an analysis found'
        return 'reports '+v
    text,n=pat_bare.subn(ws_bare,text);total+=n
    pat=re.compile(rf"\b{multi}\s+(reported|reports|said|found|wrote|noted|confirmed|revealed|added|stated|estimated|calculated)\b",re.I)
    def ws(m):
        v=m.group(1).lower()
        if v in {'reported','reports','said','wrote','added','stated'}: return 'reports indicated'
        if v=='found': return 'an analysis found'
        return 'reports '+v
    text,n=pat.subn(ws,text);total+=n
    # Full names are unambiguous even when unusual syntax remains: neutralize as
    # a generic agency rather than leave an identity shortcut. AP/AFP are NOT
    # replaced globally because they are ambiguous abbreviations.
    text,n=WIRE_FULL_RE.subn('the news agency',text);total+=n
    # Absorb common generic agency constructions to reduce placeholder frequency.
    for pat,repl in [
        (r"\btold\s+the news agency\s+that\b","said that"),(r"\btold\s+the news agency\b","said"),
        (r"\baccording to\s+the news agency\b","according to reports"),
        (r"\bthe news agency\s+(?:reported|reports|said|says)\b","reports indicated"),
        (r"\bthe news agency['’]s\s+(?:footage|video|images?|photographs?)\b",lambda m:m.group(0).split()[-1]),
    ]:
        text,n=re.subn(pat,repl,text,flags=re.I);total+=n
    return clean_punctuation_gaps(normalize_spaces(text)),total


def remove_structural_author(text: str, author_raw: str) -> tuple[str,int]:
    """Remove current-article author only in unmistakable byline/credit contexts."""
    total=0
    names=author_components(author_raw or '')
    for name in names:
        esc=re.escape(name).replace(r'\ ',r'\s+')
        pats=[
            rf"^\s*(?:By|Reporting by|Written by)\s+{esc}\b\s*[-–—,:|]*\s*",
            rf"^\s*{esc}\s*[-–—,:|]+\s*(?=(?:Reporter|Correspondent|Editor|News|Business|World|Politics)\b)",
            rf"\b(?:additional reporting by|reporting by|written by)\s+{esc}\b",
        ]
        for pat in pats:
            text,n=re.subn(pat,' ',text,flags=re.I);total+=n
    return normalize_spaces(text),total


def strip_residual_target_source_sentences(text: str) -> tuple[str,int]:
    parts=re.split(r"(?<=[.!?])\s+",text); kept=[]; removed=0
    meta=re.compile(r"\b(?:request for comment|reached out|contacted .* for comment|respond(?:ed)? to|analysis by|independently verified|according to|reported by|speaking to|interview with|correspondent|reporter|journalist|programme|program|podcast|sign up|newsletter)\b",re.I)
    for sent in parts:
        has=bool(TARGET_OUTLET_RE.search(sent) or TARGET_DOMAIN_RE.search(sent) or BARE_GUARDIAN_MEDIA_RE.search(sent) or TARGET_SERVICE_EXPLICIT_RE.search(sent) or TARGET_SERVICE_MEDIA_CONTEXT_RE.search(sent) or SANAD_TARGET_CONTEXT_RE.search(sent))
        if has and meta.search(sent): removed+=1
        else: kept.append(sent)
    return normalize_spaces(' '.join(kept)),removed


def unresolved_target_identity(text: str) -> str|None:
    if TARGET_DOMAIN_RE.search(text or ''): return 'target_domain'
    if TARGET_OUTLET_RE.search(text or ''): return 'target_outlet_name'
    if BARE_GUARDIAN_MEDIA_RE.search(text or ''): return 'guardian_media_identity'
    if TARGET_SERVICE_EXPLICIT_RE.search(text or ''): return 'target_owned_service'
    if TARGET_SERVICE_MEDIA_CONTEXT_RE.search(text or ''): return 'target_owned_service_context'
    if SANAD_TARGET_CONTEXT_RE.search(text or ''): return 'target_owned_service_context'
    # Highly distinctive standalone services that should never survive.
    if re.search(r"\b(?:HARDtalk|iPlayer|BBC Sounds)\b",text or '',re.I): return 'target_owned_distinctive_service'
    return None


def preclean_for_relevance(text: str, title_lookup) -> str:
    text=repair_mojibake(text or '')
    text,_=strip_recommended_stories(text,title_lookup)
    for pat in LEADING_ARTIFACT_PATTERNS: text=pat.sub('',text,count=1)
    text=BBC_EXTERNAL_ANNOTATION_RE.sub(', ',text)
    text,_=strip_tail_boilerplate(text); text,_=strip_global_promos(text)
    return normalize_spaces(text)


def clean_text_global(text: str, title_lookup, author_matcher=None, article_author: str='') -> tuple[str,dict]:
    """Final identical, label-blind cleaner. It never receives portal/topic."""
    stats=Counter(); original_len=len(text or ''); text=repair_mojibake(text or '')
    target_reporter_aliases=extract_target_reporter_aliases(text)
    text,n=strip_recommended_stories(text,title_lookup);stats['recommended_blocks']+=n
    for pat in LEADING_ARTIFACT_PATTERNS:
        text,n=pat.subn('',text,count=1);stats['leading_publish_artifacts']+=n
    text,n=BBC_EXTERNAL_ANNOTATION_RE.subn(', ',text);stats['external_annotations']+=n
    text,n=strip_tail_boilerplate(text);stats['tail_boilerplate']+=n
    text,n=strip_global_promos(text);stats['global_promotional_blocks']+=n
    text,n=normalize_target_attributions(text);stats['target_attributions_normalized']+=n
    text,n=normalize_wire_attributions(text);stats['wire_attributions_normalized']+=n
    text,n=strip_synthetic_target_reporter_attribution(text);stats['synthetic_target_reporter_attributions_removed']+=n
    text,n=strip_target_reporter_alias_attributions(text,target_reporter_aliases);stats['target_reporter_alias_attributions_removed']+=n
    text,n=remove_structural_author(text,article_author);stats['structural_author_removed']+=n
    text,n=strip_residual_target_source_sentences(text);stats['target_source_meta_sentences_removed']+=n
    # Catch target-program attribution variants that can otherwise leave a
    # possessive scar after a parent brand was normalized.
    prog_today=r"(?:Gaza\s+|Business\s+)?Today\s+(?:programme|program)"
    for pat,repl in [
        (rf"\b(?:told|said(?:['’]s)?|speaking in an interview(?:['’]s)?)\s+(?:the\s+)?{prog_today}\b","said"),
        (rf"\b(?:told|said(?:['’]s)?)\s+(?:the\s+)?Newsnight\b","said"),
        (rf"\b(?:Radio\s+(?:4|Four)|Radio)['’]s\s+{prog_today}\b",""),
        (rf"\bAsked on\s+(?:the\s+)?{prog_today}\b","Asked in an interview"),
        (rf"\bsaid\s+(?:[A-Z][A-Za-z'’ -]{{1,55}}['’]s\s+)?(?:[A-Z][A-Za-z'’ -]{{1,35}}\s+)?{prog_today}\b","said"),
        (rf"\btold\s+(?:the\s+)?{prog_today}\b","said"),
    ]:
        text,k=re.subn(pat,repl,text,flags=re.I);stats['target_program_residue_repaired']+=k
    # Source-credit metadata is not article content. Remove the whole local
    # reporting-credit clause rather than leave names or malformed fragments.
    text,k=re.subn(r"[^.!?]{0,140}\bcontributed (?:reporting|to this report)\b[^.!?]*(?:[.!?]|$)",' ',text,flags=re.I);stats['reporting_credit_sentences_removed']+=k
    text,k=re.subn(r"\bwith (?:additional )?reporting (?:from|by) [^.!?]{1,140}(?:[.!?]|$)",' ',text,flags=re.I);stats['reporting_credit_sentences_removed']+=k
    text,n=URL_RE.subn(' ',text);stats['urls']+=n
    text,n=EMAIL_RE.subn(' ',text);stats['emails']+=n
    text,n=SOCIAL_HANDLE_RE.subn(' ',text);stats['social_handles']+=n
    # Generic flattened byline/header remnants only; no global person-name deletion.
    text,n=re.subn(r"^\s*(?:By|Reporting by|Written by)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){1,4}\s*[-–—,:|]*\s*",'',text);stats['generic_structural_bylines']+=n
    # Mechanical scars only, not paraphrasing.
    text,n=re.subn(r"(?<!\w)['’]s(?=\s+(?:said|reported|noted|added|explained|wrote)\b)",' ',text,flags=re.I);stats['orphan_possessive_scars']+=n
    # V6 disabled broad told-scar rewrite
    n=0
    text=clean_punctuation_gaps(normalize_spaces(text))
    stats['chars_removed_total']=max(0,original_len-len(text))
    return text,dict(stats)

# ---------------------------------------------------------------------------
# Final global cleaner
# ---------------------------------------------------------------------------


def remove_repeated_title(body: str, title: str) -> tuple[str, bool]:
    if not body or not title:
        return body, False
    b = normalize_spaces(body)
    t = normalize_spaces(title)
    # Exact normalized repetition at body start.
    if b.casefold().startswith(t.casefold()):
        rest = b[len(t):].lstrip(" -–—:|.\t")
        return rest, True
    # Sometimes title gets a trailing site suffix that cleaning removed; compare
    # a long prefix only if it is substantial.
    if len(t) >= 50:
        prefix = t[:50].casefold()
        if b[:50].casefold() == prefix:
            # Do not fuzzy-delete; exact prefix alone is insufficient.
            pass
    return b, False


# V7 extraction-integrity gate -------------------------------------------------
# In this scrape, BBC .com article extraction prepends the visible page H1 to
# the extracted article text. BBC .co.uk does not. Because the primary models
# are deliberately BODY ONLY, a surviving H1 would give one source a headline
# shortcut. We therefore remove it only when its boundary can be identified at
# high precision; otherwise that .com row is rejected rather than guessed at.
#
# This is deliberately separate from clean_text_global(): the linguistic cleaner
# remains label-blind and never receives portal/topic. This gate corrects a known
# source-extraction structure, analogous to source-specific page-type eligibility.
_H1_WORD_RE = re.compile(r"\b[\w£$€%]+(?:['’\-][\w£$€%]+)*\b", re.UNICODE)


def _h1_word_matches(text: str):
    return list(_H1_WORD_RE.finditer(text or ""))


# V9 Al Jazeera extraction-integrity gate --------------------------------------
# A small subset of Al Jazeera pages in this scrape have a related video/card
# headline flattened immediately in front of the actual article lead.  Because
# the main classifier is BODY ONLY, that headline-like text must not survive.
# The detector below is intentionally high precision: it only flags a short,
# punctuation-free, headline-like first clause followed by an independently
# sentence-like second clause.  Flagged rows are rejected rather than boundary-
# stripped, so a mistaken boundary can never delete genuine body prose.
_AJ_LEADING_HEADLINE_ACTION_RE = re.compile(
    r"\b(?:kills?|dies?|blows?|destroys?|collapses?|pounds?|waits?|suggests?|reopens?|"
    r"restricts?|grows?|worr(?:y|ies)|deteriorates?|slams?|attacks?|backs?|rejects?|"
    r"approves?|halts?|blocks?|allows?|returns?|warns?|urges?|calls?|faces?|sparks?|"
    r"shakes?|hits?|cuts?|raises?|falls?|rises?)\b", re.I,
)
_AJ_BODY_START_RE = re.compile(
    r"^(?:A|An|The|At\s+least|Israel|Iran(?:['’]s)?|United\s+States|US|Fourteen|Strong|"
    r"Hundreds|Thousands|Palestinian|Israeli|Qatar|Lebanon|Syria|Gaza)\b"
)
_AJ_BODY_FINITE_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|would|can|could|may|might|must|should|"
    r"says?|said|claimed|claims|appealed|appeals|launched|launches|killed|kills|"
    r"condemned|condemns|warned|warns|called|calls|reported|reports|showed|shows|"
    r"began|remains|faces|announced|approved|rejected|urged|accused|continues|created|"
    r"released|opened|closed|rose|fell|slumped|surged|moved|pushed|sought|took|made|"
    r"found|adding|adds?)\b", re.I,
)
_AJ_BAD_PREFIX_FINAL_WORDS = {
    "in","on","at","to","from","for","of","with","by","as","into","onto","across",
    "over","under","amid","among","against","after","before","during","through","around",
    "near","via","and","or","but","nor","yet","so","that","which","who","whose","whom",
    "a","an","the","this","these","those","its","their","his","her","our","your","my",
    "one","more","less","than","about","despite","without","within","between","since",
    "central","northern","southern","eastern","western","occupied","besieged","ongoing",
    "following","including","excluding","toward","towards",
}


def _aljazeera_leading_embedded_heading(body: str) -> str:
    """Return the suspicious leading heading, or empty string if no hard match."""
    b = normalize_spaces(body or "")
    words = list(re.finditer(r"\S+", b))
    for n in range(5, min(16, len(words))):
        end = words[n-1].end()
        prefix = b[:end]
        rest = b[end:].lstrip()
        # A genuine extraction heading in this failure mode is not sentence-terminated.
        if re.search(r"[.!?;:]", prefix):
            continue
        last_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-z]+)?", prefix)
        if not last_tokens:
            continue
        last = last_tokens[-1].casefold().replace("’", "'")
        if last in _AJ_BAD_PREFIX_FINAL_WORDS or last.endswith("ing"):
            continue
        if not _AJ_BODY_START_RE.match(rest):
            continue
        first_clause = re.split(r"[.!?;:]", rest, maxsplit=1)[0]
        first_words = " ".join(first_clause.split()[:14])
        if not _AJ_BODY_FINITE_RE.search(first_words):
            continue
        prefix_is_headline = (
            re.match(r"^(?:How|Why|What)\b", prefix, re.I)
            or _AJ_LEADING_HEADLINE_ACTION_RE.search(prefix)
            or re.match(r"^(?:Israel|Iran|United\s+States|US|China|Russia)\s+has\s+\w+(?:ed|en)\b", prefix, re.I)
        )
        if prefix_is_headline:
            return prefix
    return ""


def remove_known_leading_extraction_heading(
    body: str, title: str, portal: str, url: str
) -> tuple[str | None, str, str]:
    """Return (body_without_heading, method, removed_heading).

    V8 intentionally makes the BBC.com extraction rule more conservative than
    V7. The raw BBC.com scrape flattens the visible H1 immediately in front of
    the article body, with no reliable separator. V7 attempted to infer
    alternate-H1 boundaries from fuzzy title similarity; end-to-end QA showed
    that this can either leave a headline suffix or consume the first body word.

    Therefore a BBC.com row is retained only when the raw body starts with the
    metadata title exactly and the text immediately after that title has a
    plausible sentence start. A lowercase continuation indicates that the
    metadata title is only a prefix of a longer visible H1, so the row is
    rejected rather than guessed at. Non-exact/alternate H1s are also rejected.
    BBC.co.uk rows are unaffected and are preferred at canonical deduplication.
    """
    b = normalize_spaces(body)
    t = normalize_spaces(title)
    host = (urlparse(url or "").hostname or "").casefold()

    if portal == "aljazeera" and host in {"aljazeera.com", "www.aljazeera.com"}:
        rogue_heading = _aljazeera_leading_embedded_heading(b)
        if rogue_heading:
            return None, "aljazeera_leading_embedded_media_heading", rogue_heading
        return b, "not_applicable", ""

    if portal != "bbc" or host not in {"bbc.com", "www.bbc.com"}:
        return b, "not_applicable", ""
    if not b or not t:
        return None, "bbc_com_unresolved_h1", ""

    if not b.casefold().startswith(t.casefold()):
        return None, "bbc_com_unresolved_alt_h1", ""

    # Do not blindly cut the metadata-title prefix: it can be a strict prefix
    # of the visible H1 (e.g. "failure" -> "failures" or a trailing clause).
    rest = b[len(t):].lstrip(" -–—:|.\t")
    probe = rest.lstrip("\"'“”‘’([{")
    if not probe:
        return None, "bbc_com_unresolved_h1", ""
    first = probe[0]
    if first.isalpha() and not first.isupper():
        return None, "bbc_com_title_is_prefix_of_longer_h1", ""
    if not (first.isupper() or first.isdigit()):
        return None, "bbc_com_unresolved_h1", ""

    return rest, "bbc_com_exact_metadata_h1_validated", t


def clean_text_global(
    text: str,
    title_lookup: dict[str, list[str]],
    author_matcher: AuthorMatcher,
) -> tuple[str, dict]:
    """One identical, label-blind cleaning function for every article."""
    stats = Counter()
    original_len = len(text or "")

    text = repair_mojibake(text or "")

    text, n = normalize_target_attributions(text)
    stats["target_attributions_normalized"] += n

    text, n = strip_recommended_stories(text, title_lookup)
    stats["recommended_blocks"] += n

    for pat in LEADING_ARTIFACT_PATTERNS:
        text, n = pat.subn("", text, count=1)
        stats["leading_publish_artifacts"] += n

    text, n = BBC_EXTERNAL_ANNOTATION_RE.subn(", ", text)
    stats["external_annotations"] += n

    text, n = strip_tail_boilerplate(text)
    stats["tail_boilerplate"] += n

    text, n = strip_global_promos(text)
    stats["global_promotional_blocks"] += n

    # Remove target outlet identities globally.
    for p in OUTLET_PATTERNS_CASE_INSENSITIVE:
        text, n = re.subn(p, " ", text, flags=re.IGNORECASE)
        stats["outlet_identifiers"] += n
    for p in OUTLET_PATTERNS_CASE_SENSITIVE:
        text, n = re.subn(p, " ", text)
        stats["outlet_identifiers"] += n

    # Remove known source agencies globally.
    for p in WIRE_PATTERNS:
        text, n = re.subn(p, " ", text, flags=re.IGNORECASE)
        stats["wire_identifiers"] += n

    # Remove EVERY known author identity from EVERY sample.
    text, n = author_matcher.sub(text)
    stats["author_identifiers"] += n

    # Source-credit remnants are non-content metadata. Remove globally.
    text, n = re.subn(r"(?:,\s*)?(?:and\s+)?contributed\s+(?:reporting|to this report)\.?", " ", text, flags=re.IGNORECASE)
    stats["source_credit_scars"] += n
    text, n = re.subn(r"\b(?:with\s+)?additional reporting by\b", " ", text, flags=re.IGNORECASE)
    stats["source_credit_scars"] += n
    text, n = re.subn(r"(?<!\w)['’]s(?=\s+(?:said|reported|noted|added|explained|wrote)\b)", " ", text, flags=re.IGNORECASE)
    stats["orphan_possessive_scars"] += n

    # Repair the small, deterministic grammatical scar that can be left when an
    # outlet name is removed from constructs such as "X told The Guardian that".
    # This rule is global and only fires on syntactically impossible remnants.
    text, n = re.subn(r"\btold\s+the\s+(that|it|he|she|they|we|there)\b", r"said \1", text, flags=re.IGNORECASE)
    stats["told_the_scars_repaired"] += n

    # Removing a target outlet/source name can leave syntactically impossible
    # fragments such as "the said" or "the has". Repair ONLY these impossible
    # structures, identically in every article. Normal phrases such as
    # "the report" or "the newspaper" are left untouched.
    text, n = re.subn(
        r"\bthe\s+(said|reported|added|noted|explained|wrote|stated|confirmed|announced|has|have|had|is|was|were|will|would|can|could)\b",
        lambda m: "it " + m.group(1).lower(), text, flags=re.IGNORECASE,
    )
    stats["impossible_the_verb_scars_repaired"] += n
    text, n = re.subn(r"\btold\s+the\s+said\s+(?:newspaper|outlet|broadcaster)\b", "told reporters", text, flags=re.IGNORECASE)
    stats["source_object_scars_repaired"] += n
    text, n = re.subn(r"\bnews agency\s+The\s+(?=(?:said|reported|cited)\b)", "a news agency ", text, flags=re.IGNORECASE)
    stats["source_agency_scars_repaired"] += n
    text, n = re.subn(r"\bThe\s*\(\s*\)\s*", " ", text)
    stats["empty_source_parentheses_repaired"] += n

    # After names/outlets are blanked, clean generic beginning-of-text byline scars.
    text, n = re.subn(
        r"^\s*(?:By|Reporting by|Written by)\s+(?:in\s+[A-Z][A-Za-z .'-]{1,60})?\s*[-–—,:|]*\s*",
        "", text, flags=re.IGNORECASE,
    )
    stats["generic_byline_scars"] += n

    # BBC-style flattened author/role metadata, only when it is structurally clear.
    text, n = re.subn(
        r"^\s*Author,\s*[^-]{0,120}?\s*-\s*Role,\s*(?:News|Correspondent|Reporter|Editor|Senior correspondent|Business reporter)?\s*[-–—,:|]*\s*",
        "", text, flags=re.IGNORECASE,
    )
    stats["author_role_artifacts"] += n

    text, n = URL_RE.subn(" ", text)
    stats["urls"] += n
    text, n = EMAIL_RE.subn(" ", text)
    stats["emails"] += n
    text, n = SOCIAL_HANDLE_RE.subn(" ", text)
    stats["social_handles"] += n

    # URL/email/handle deletion itself can expose the same impossible "the +
    # verb" scar (for example "the @WSJ would" -> "the would"). Run the
    # conservative repair once more after those web identifiers are removed.
    text, n = re.subn(
        r"\bthe\s+(said|reported|added|noted|explained|wrote|stated|confirmed|announced|has|have|had|is|was|were|will|would|can|could)\b",
        lambda m: "it " + m.group(1).lower(), text, flags=re.IGNORECASE,
    )
    stats["post_web_identifier_scars_repaired"] += n

    text = clean_punctuation_gaps(normalize_spaces(text))
    stats["chars_removed_total"] = max(0, original_len - len(text))
    return text, dict(stats)



# Re-declare the final cleaner here, after the legacy draft cleaner above, so
# this definition is the one used by main().
def clean_text_global(text: str, title_lookup, author_matcher=None, article_author: str='') -> tuple[str,dict]:
    """Final identical, label-blind cleaner. It never receives portal/topic."""
    stats=Counter(); original_len=len(text or ''); text=repair_mojibake(text or '')
    target_reporter_aliases=extract_target_reporter_aliases(text)
    text,n=strip_recommended_stories(text,title_lookup);stats['recommended_blocks']+=n
    for pat in LEADING_ARTIFACT_PATTERNS:
        text,n=pat.subn('',text,count=1);stats['leading_publish_artifacts']+=n
    text,n=BBC_EXTERNAL_ANNOTATION_RE.subn(', ',text);stats['external_annotations']+=n
    text,n=strip_tail_boilerplate(text);stats['tail_boilerplate']+=n
    text,n=strip_global_promos(text);stats['global_promotional_blocks']+=n
    text,n=normalize_target_attributions(text);stats['target_attributions_normalized']+=n
    text,n=normalize_wire_attributions(text);stats['wire_attributions_normalized']+=n
    text,n=strip_synthetic_target_reporter_attribution(text);stats['synthetic_target_reporter_attributions_removed']+=n
    text,n=strip_target_reporter_alias_attributions(text,target_reporter_aliases);stats['target_reporter_alias_attributions_removed']+=n
    text,n=remove_structural_author(text,article_author);stats['structural_author_removed']+=n
    text,n=strip_residual_target_source_sentences(text);stats['target_source_meta_sentences_removed']+=n
    # Catch target-program attribution variants that can otherwise leave a
    # possessive scar after a parent brand was normalized.
    prog_today=r"(?:Gaza\s+|Business\s+)?Today\s+(?:programme|program)"
    for pat,repl in [
        (rf"\b(?:told|said(?:['’]s)?|speaking in an interview(?:['’]s)?)\s+(?:the\s+)?{prog_today}\b","said"),
        (rf"\b(?:told|said(?:['’]s)?)\s+(?:the\s+)?Newsnight\b","said"),
        (rf"\b(?:Radio\s+(?:4|Four)|Radio)['’]s\s+{prog_today}\b",""),
        (rf"\bAsked on\s+(?:the\s+)?{prog_today}\b","Asked in an interview"),
        (rf"\bsaid\s+(?:[A-Z][A-Za-z'’ -]{{1,55}}['’]s\s+)?(?:[A-Z][A-Za-z'’ -]{{1,35}}\s+)?{prog_today}\b","said"),
        (rf"\btold\s+(?:the\s+)?{prog_today}\b","said"),
    ]:
        text,k=re.subn(pat,repl,text,flags=re.I);stats['target_program_residue_repaired']+=k
    # Source-credit metadata is not article content. Remove the whole local
    # reporting-credit clause rather than leave names or malformed fragments.
    text,k=re.subn(r"[^.!?]{0,140}\bcontributed (?:reporting|to this report)\b[^.!?]*(?:[.!?]|$)",' ',text,flags=re.I);stats['reporting_credit_sentences_removed']+=k
    text,k=re.subn(r"\bwith (?:additional )?reporting (?:from|by) [^.!?]{1,140}(?:[.!?]|$)",' ',text,flags=re.I);stats['reporting_credit_sentences_removed']+=k
    text,n=URL_RE.subn(' ',text);stats['urls']+=n
    text,n=EMAIL_RE.subn(' ',text);stats['emails']+=n
    text,n=SOCIAL_HANDLE_RE.subn(' ',text);stats['social_handles']+=n
    text,n=re.subn(r"^\s*(?:By|Reporting by|Written by)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){1,4}\s*[-–—,:|]*\s*",'',text);stats['generic_structural_bylines']+=n
    text,n=re.subn(r"(?<!\w)['’]s(?=\s+(?:said|reported|noted|added|explained|wrote)\b)",' ',text,flags=re.I);stats['orphan_possessive_scars']+=n
    # V6: do not globally rewrite 'told that/he/...' -> 'said ...'; this corrupted ordinary passive constructions such as 'were told that'.
    n=0
    # A few grammatical neutralizations deliberately use generic lower-case
    # phrases. Capitalize them only when they begin a sentence; do not otherwise
    # rewrite article prose.
    generated_starts = (
        r"(?:reports indicated|reports cited|according to reports|speaking to reporters|"
        r"speaking in an interview|in an interview|interviewed by reporters|appeared in an interview|"
        r"broadcast footage|an investigation|an analysis|an Ipsos poll|a YouGov poll|"
        r"a reporter|a journalist|a photographer)"
    )
    pat = re.compile(rf"(^|[.!?]\s+)({generated_starts})\b", re.I)
    def _cap_generated(m):
        phrase=m.group(2)
        return m.group(1)+phrase[:1].upper()+phrase[1:]
    text,n=pat.subn(_cap_generated,text);stats['generated_sentence_starts_capitalized']+=n
    text=clean_punctuation_gaps(normalize_spaces(text))
    stats['chars_removed_total']=max(0,original_len-len(text))
    return text,dict(stats)



def extract_target_reporter_aliases(text: str) -> list[str]:
    """Extract reporter names only from explicit target-owned source constructions.

    V9 deliberately keeps proper-name matching case-sensitive and sentence-bounded.
    The older global ``re.I`` pattern could treat ordinary lowercase words as name
    tokens and greedily span a period into the next sentence.
    """
    out = set()
    outlet_ci = rf"(?i:{TARGET_OUTLET_ONLY_ALT})"
    role_ci = r"(?i:reporter|correspondent|journalist)"
    verb_ci = r"(?i:said|reported|wrote|noted|added|explained|found)"
    reporting_ci = r"(?i:reporting|speaking)"
    person = PERSON_NAME_RE_STRICT_V9
    pats = [
        rf"\b{outlet_ci}['’]s\s+({person})(?=\s+{verb_ci}\b|,\s*(?:(?i:who\s+is)\s+)?{reporting_ci}\b)",
        rf"\b{outlet_ci}\s+{role_ci}\s+({person})(?=\s+{verb_ci}\b|,)",
    ]
    for pat in pats:
        for m in re.finditer(pat, text or ""):
            name = normalize_spaces(m.group(1)).strip(" ,.-")
            if len(name.split()) >= 2:
                out.add(name)
                last = name.split()[-1]
                if len(last) >= 4:
                    out.add(last)
    return sorted(out, key=len, reverse=True)


def strip_target_reporter_alias_attributions(text: str, aliases: list[str]) -> tuple[str, int]:
    total = 0
    verbs = r"(?:said|reported|wrote|noted|added|explained|found)"
    for alias in aliases:
        a = re.escape(alias).replace(r"\ ", r"\s+")
        # "..., said Khodr." / "..., reported Zeina Khodr."
        text, n = re.subn(rf",?\s*\b{verbs}\s+{a}\b(?=\s*[,.;:!?]|\s*$)", "", text, flags=re.I); total += n
        # "according to Khodr"
        text, n = re.subn(rf"\baccording to\s+{a}\b", "according to reports", text, flags=re.I); total += n
        # Sentence-boundary source subject: preserve clause and capitalize it.
        pat = re.compile(rf"(^|[.!?]\s+)\b{a}\s+{verbs}(?:\s+that)?\s+([a-z])", re.I)
        text, n = pat.subn(lambda m: m.group(1) + m.group(2).upper(), text); total += n
        # Mid-sentence source subject.
        text, n = re.subn(rf"\b{a}\s+{verbs}(?:\s+that)?\s+", "", text, flags=re.I); total += n
    return clean_punctuation_gaps(normalize_spaces(text)), total

# ---------------------------------------------------------------------------
# Final source-normalization overrides / scar cleanup
# ---------------------------------------------------------------------------

_WIRE_CORE_FULL = r"(?:Reuters|Associated\s+Press|Agence\s+France[- ]Press(?:e)?)"
_WIRE_CORE_WITH_AFP = rf"(?:{_WIRE_CORE_FULL}|AFP)"
_WIRE_ROLE = r"(?:reporter|journalist|photographer|spokesperson|contractor|cameraman)"
_WIRE_VERB = r"(?:reported|reports|said|says|wrote|noted|confirmed|revealed|added|stated|estimated|calculated|found|cited)"


def normalize_wire_attributions(text: str) -> tuple[str, int]:
    """Neutralize wire identities without manufacturing malformed noun phrases.

    Full wire names and AFP are unambiguous. Bare AP is touched only in clear
    news-source grammar, so names/abbreviations such as Welsh 'ap' are preserved.
    """
    total = 0

    # Partnered polls: preserve the non-wire partner rather than create a generic
    # placeholder such as "a the news agency/Ipsos poll".
    text, n = re.subn(
        rf"\b(?:a|an|the)?\s*{_WIRE_CORE_WITH_AFP}\s*/\s*(Ipsos|YouGov)\s+poll\b",
        lambda m: ("an " if m.group(1).lower().startswith("i") else "a ") + m.group(1) + " poll",
        text, flags=re.I,
    ); total += n

    # Agency staff / media objects: drop the agency identity but retain the role.
    text, n = re.subn(
        rf"\b(?:a|an|the)\s+{_WIRE_CORE_WITH_AFP}(?:\s+news\s+agency)?\s+({_WIRE_ROLE[3:-1]})\b",
        lambda m: ("A " if m.group(0)[:1].isupper() else "a ") + m.group(1).lower(), text, flags=re.I,
    ); total += n
    text, n = re.subn(
        rf"\b{_WIRE_CORE_WITH_AFP}(?:\s+news\s+agency)?\s+({_WIRE_ROLE[3:-1]})\b",
        lambda m: ("A " if m.group(0)[:1].isupper() else "a ") + m.group(1).lower(), text, flags=re.I,
    ); total += n

    # Agency-produced objects. Source identity is not needed for the model text.
    obj = r"(?:poll|tally|footage|video|images?|photographs?|analysis|investigation|report)"
    text, n = re.subn(
        rf"\b(?:a|an|the)\s+{_WIRE_CORE_WITH_AFP}(?:\s+news\s+agency)?\s+({obj[3:-1]})\b",
        lambda m: ("an " if m.group(1).lower().startswith(tuple("aeiou")) else "a ") + m.group(1).lower(),
        text, flags=re.I,
    ); total += n
    text, n = re.subn(
        rf"\b{_WIRE_CORE_WITH_AFP}(?:\s+news\s+agency)?\s+({obj[3:-1]})\b",
        lambda m: ("an " if m.group(1).lower().startswith(tuple("aeiou")) else "a ") + m.group(1).lower(),
        text, flags=re.I,
    ); total += n

    source = rf"(?:the\s+)?{_WIRE_CORE_WITH_AFP}(?:\s+news\s+agency)?"
    # Multiple named agencies acting as one grammatical subject.
    multi_subject = re.compile(rf"\b{source}\s*(?:,|and|&)\s*{source}\s+({_WIRE_VERB[3:-1]})\b", re.I)
    text, n = multi_subject.subn("reports indicated", text); total += n
    # Object/source attributions.
    for pat, repl in [
        (rf"\btold\s+{source}\s+that\b", "said that"),
        (rf"\btold\s+{source}\b", "said"),
        (rf"\baccording to\s+{source}\b", "according to reports"),
        (rf"\bspeaking to\s+{source}\b", "speaking to reporters"),
        (rf"\bin an interview with\s+{source}\b", "in an interview"),
        (rf"\binterviewed by\s+{source}\b", "interviewed by reporters"),
        (rf"\basked by\s+{source}\b", "asked by reporters"),
        (rf"\bin a statement (?:sent|given) to\s+{source}\b", "in a statement"),
    ]:
        text, n = re.subn(pat, repl, text, flags=re.I); total += n

    # Agency as grammatical subject.
    pat = re.compile(rf"\b{source}\s+({_WIRE_VERB[3:-1]})\b", re.I)
    def _wire_subject(m):
        v = m.group(1).lower()
        if v == "found":
            return "an analysis found"
        if v == "cited":
            return "reports cited"
        return "reports indicated"
    text, n = pat.subn(_wire_subject, text); total += n

    # Bare AP only when the grammar makes the Associated Press meaning clear.
    ap_source = r"(?:the\s+)?AP"
    for pat, repl in [
        (rf"\btold\s+{ap_source}\s+that\b", "said that"),
        (rf"\btold\s+{ap_source}\b", "said"),
        (rf"\baccording to\s+{ap_source}\b", "according to reports"),
        (rf"\bspeaking to\s+{ap_source}\b", "speaking to reporters"),
        (rf"\bin an interview with\s+{ap_source}\b", "in an interview"),
        (rf"\binterviewed by\s+{ap_source}\b", "interviewed by reporters"),
        (rf"\basked by\s+{ap_source}\b", "asked by reporters"),
        (rf"\bin a statement (?:sent|given) to\s+{ap_source}\b", "in a statement"),
        (rf"\b{ap_source}\s+news\s+agency\b", "the news agency"),
        (rf"\({ap_source}\)\s+news\s+agency\b", "the news agency"),
        (rf"\breport(?:ed)?\s+by\s+{ap_source}\b", "reported in wire reports"),
    ]:
        text, n = re.subn(pat, repl, text, flags=re.I); total += n
    pat = re.compile(rf"\b{ap_source}\s+({_WIRE_VERB[3:-1]})\b")  # case-sensitive AP
    def _ap_subject(m):
        v=m.group(1)
        if v == "found": return "an analysis found"
        if v == "cited": return "reports cited"
        return "reports indicated"
    text, n = pat.subn(_ap_subject, text); total += n
    text, n = re.subn(r"\b(?:a|an|the)\s+(?:the\s+)?AP\s+(reporter|journalist|photographer|spokesperson|contractor|cameraman)\b",
                      lambda m: "a " + m.group(1).lower(), text); total += n

    # Remaining full names / AFP are unambiguous. Keep a generic source concept,
    # then absorb the most common attribution constructions below.
    text, n = re.subn(rf"\b{_WIRE_CORE_FULL}(?:\s+news\s+agency)?\b", "the news agency", text, flags=re.I); total += n
    text, n = re.subn(r"\bAFP(?:\s+news\s+agency)?\b", "the news agency", text); total += n

    # Repair article duplication / source-name duplication deterministically.
    def _fix_article(m):
        return "The news agency" if m.group(0)[:1].isupper() else "the news agency"
    text, n = re.subn(r"\b(?:a|an|the)\s+the\s+news\s+agency(?:\s+news\s+agency)?\b", _fix_article, text, flags=re.I); total += n
    text, n = re.subn(r"\bthe\s+news\s+agency\s+news\s+agency\b", "the news agency", text, flags=re.I); total += n
    text, n = re.subn(r"\bthe\s+news\s+agency\s*\(\s*(?:AFP|AP)\s*\)", "the news agency", text); total += n
    text, n = re.subn(r"\bthe\s+news\s+agency\s*/\s*(Ipsos|YouGov)\s+poll\b", lambda m: "an " + m.group(1) + " poll", text, flags=re.I); total += n

    # Absorb generic agency attribution where possible, minimizing any synthetic
    # placeholder frequency.
    for pat, repl in [
        (r"\btold\s+the news agency\s+that\b", "said that"),
        (r"\btold\s+the news agency\b", "said"),
        (r"\baccording to\s+the news agency\b", "according to reports"),
        (r"\bthe news agency\s+(?:reported|reports|said|says)\b", "reports indicated"),
        (r"\bthe news agency['’]s\s+(footage|video|images?|photographs?)\b", lambda m: m.group(1)),
    ]:
        text, n = re.subn(pat, repl, text, flags=re.I); total += n

    text, n = re.subn(r"(^|[.!?]\s+)The reports indicated\b", lambda m: m.group(1)+"Reports indicated", text); total += n
    text, n = re.subn(r"(^|[.!?]\s+)The reports cited\b", lambda m: m.group(1)+"Reports cited", text); total += n
    return clean_punctuation_gaps(normalize_spaces(text)), total


def strip_synthetic_target_reporter_attribution(text: str) -> tuple[str, int]:
    """Remove only an internal sentinel created from a target-owned reporter attribution.

    The sentinel makes this safe: naturally occurring prose such as "a reporter said"
    is not touched. The sentinel must never survive into output (audited separately).
    """
    total = 0
    marker = r"__TARGET_REPORTER_ATTR__"

    # Structural correspondent lead-in followed by the branded reporter source.
    pat = re.compile(rf"(^|[.!?]\s+)Reporting from\s+[^.!?]{{1,120}}?,\s*{marker}\s+(?:that\s+)?([a-z])", re.I)
    text, n = pat.subn(lambda m: m.group(1) + m.group(2).upper(), text); total += n
    text, n = re.subn(rf"\bReporting from\s+[^.!?]{{1,120}}?,\s*{marker}\s+(?:that\s+)?", "", text, flags=re.I); total += n

    # Sentence boundary: remove source attribution and capitalize surviving clause.
    pat = re.compile(rf"(^|[.!?]\s+){marker}\s+(?:that\s+)?([a-z])", re.I)
    text, n = pat.subn(lambda m: m.group(1) + m.group(2).upper(), text); total += n

    # Mid-sentence source attribution: remove only the internally-generated marker.
    text, n = re.subn(rf"{marker}\s+(?:that\s+)?", "", text); total += n
    # Defensive cleanup if punctuation unexpectedly follows the marker.
    text, n = re.subn(marker, "", text); total += n

    return clean_punctuation_gaps(normalize_spaces(text)), total

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

@dataclass
class Reject:
    article_id: str
    portal: str
    topic: str
    url: str
    title: str
    stage: str
    reason: str



# ==========================================================================
# FINAL V4 OVERRIDES
# Added after full local audit of the previous candidate. These definitions
# supersede earlier draft/patch definitions above.
# ==========================================================================

# External wire agencies are not target labels. Normalize only explicit source
# attribution grammar; preserve remaining genuine Reuters/AP/AFP mentions rather
# than manufacturing a portal-correlated "the news agency" placeholder.
def normalize_wire_attributions(text: str) -> tuple[str, int]:
    total = 0
    full = r"(?:Reuters|Associated\s+Press|Agence\s+France[- ]Presse|AFP)"
    src = rf"(?:the\s+)?{full}(?:\s+news\s+agency)?"
    # Partnered polls: preserve the non-wire polling organisation.
    text,n=re.subn(rf"\b(?:a|an|the)?\s*{src}\s*/\s*(Ipsos|YouGov)\s+poll\b",
                   lambda m:("an " if m.group(1).lower().startswith('i') else "a ")+m.group(1)+" poll",text,flags=re.I);total+=n
    # Preserve the generic reporter role when the wire name modifies it:
    # "told Associated Press reporters" -> "told reporters".
    text,n=re.subn(rf"\btold\s+{src}\s+(reporters?|journalists?|correspondents?)\b",
                   lambda m:"told "+m.group(1).lower(),text,flags=re.I);total+=n
    text,n=re.subn(r"\btold\s+(?:the\s+)?AP\s+(reporters?|journalists?|correspondents?)\b",
                   lambda m:"told "+m.group(1).lower(),text);total+=n
    for pat,repl in [
        (rf"\btold\s+{src}\s+that\b","said that"),(rf"\btold\s+{src}\b","said"),
        (rf"\baccording to\s+{src}\b","according to reports"),(rf"\bspeaking to\s+{src}\b","speaking to reporters"),
        (rf"\bin an interview with\s+{src}\b","in an interview"),(rf"\binterviewed by\s+{src}\b","interviewed by reporters"),
        (rf"\basked by\s+{src}\b","asked by reporters"),
    ]:
        text,n=re.subn(pat,repl,text,flags=re.I);total+=n
    # Bare AP only in unmistakable source grammar (case-sensitive AP).
    for pat,repl in [
        (r"\btold\s+(?:the\s+)?AP\s+that\b","said that"),(r"\btold\s+(?:the\s+)?AP\b","said"),
        (r"\baccording to\s+(?:the\s+)?AP\b","according to reports"),(r"\bspeaking to\s+(?:the\s+)?AP\b","speaking to reporters"),
        (r"\bin an interview with\s+(?:the\s+)?AP\b","in an interview"),(r"\binterviewed by\s+(?:the\s+)?AP\b","interviewed by reporters"),
    ]:
        text,n=re.subn(pat,repl,text);total+=n
    # V9: if the agency is followed by its own reporter role, preserve the
    # eyewitness attribution but remove the agency identity.  This must run
    # before the generic agency-subject rule or it creates "Reports indicated
    # its correspondent ..." with a dangling possessive pronoun.
    text,n=re.subn(
        rf"\b{src}\s+said\s+its\s+(?:correspondent|reporter|journalist|photographer)\s+"
        rf"(reported|said|wrote|noted|described)\b",
        lambda m:"A correspondent "+m.group(1).lower(), text, flags=re.I); total+=n

    # Named agency as reporting subject.
    pat=re.compile(rf"\b{src}\s+(reported|reports|said|says|found|wrote|noted|confirmed|revealed|added|stated|estimated|calculated|cited)\b",re.I)
    def _wire_subj(m):
        v=m.group(1).lower()
        if v=='found': return 'an analysis found'
        if v=='cited': return 'reports cited'
        return 'reports indicated'
    text,n=pat.subn(_wire_subj,text);total+=n
    pat_ap=re.compile(r"\b(?:the\s+)?AP\s+(reported|reports|said|says|found|wrote|noted|confirmed|revealed|added|stated|estimated|calculated|cited)\b")
    text,n=pat_ap.subn(_wire_subj,text);total+=n
    return clean_punctuation_gaps(normalize_spaces(text)),total


# Programme/source tails which the previous candidate could leave after correctly
# removing the target parent brand. Because these patterns only operate after an
# impossible generated attribution such as "said's ...", they are safe to repair.
_BBC_SOURCE_TAIL_V4 = (
    r"(?:Newshour(?:\s+programme)?|Newsday(?:\s+programme)?|World\s+Tonight(?:\s+programme)?|"
    r"(?:The\s+)?World\s+[Aa]t\s+One(?:\s+programme)?|Woman['’]s\s+Hour(?:\s+programme)?|"
    r"Gaza\s+Lifeline(?:\s+programme)?|Gaza\s+Today(?:\s+programme|\s+podcast)?|Gaza\s+radio\s+service|"
    r"World\s+Business\s+Report|Good\s+Morning\s+Scotland(?:\s+programme)?|Scotcast\s+podcast|"
    r"Opening\s+Bell|Breakfast|Radio\s+Scotland\s+Breakfast(?:\s+programme)?|"
    r"20\s+Minutes\s+with\s+Mark\s+Patterson(?:\s+programme)?|Sunday\s+with\s+Laura\s+Kuenssberg(?:\s+programme)?|"
    r"PM\s+programme|Radio\s+4['’]s\s+PM\s+programme)"
)
_AJ_SOURCE_TAIL_V4 = (
    r"(?:UpFront|The\s+Take(?:\s+podcast)?|Fault\s+Lines|Inside\s+Story|Counting\s+the\s+Cost|"
    r"The\s+Listening\s+Post|The\s+Stream|People\s*&\s*Power|101\s+East|Mawazine\s+programme)"
)
_TARGET_SOURCE_TAIL_V4 = rf"(?:{_BBC_SOURCE_TAIL_V4}|{_AJ_SOURCE_TAIL_V4})"
_PERSON_TAIL_V4 = r"(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+(?:(?:al|el|bin|ibn|de|van|von)[- ]?[A-Z]?[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+|[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+)){1,4})"


def repair_target_source_scars_v4(text: str) -> tuple[str,int]:
    total=0
    # Possessive source tails after generated attribution verbs.
    for starter in [r"said",r"reported",r"speaking\s+to\s+reporters",r"speaking\s+in\s+an\s+interview"]:
        text,n=re.subn(rf"\b({starter})['’]s\s+{_TARGET_SOURCE_TAIL_V4}\b",lambda m:m.group(1),text,flags=re.I);total+=n
    # Target-owned reporter/person after a generated possessive: "said's Name".
    text,n=re.subn(rf"\b(said|reported)['’]s\s+{_PERSON_TAIL_V4}\b",lambda m:m.group(1),text);total+=n
    # Target role/partner tails.
    role=r"(?:Gaza\s+correspondent|US\s+(?:news\s+)?partner\s+CBS(?:\s+News)?|correspondent|reporter|journalist)"
    text,n=re.subn(rf"\b(said|reported)['’]s\s+{role}\b",lambda m:m.group(1),text,flags=re.I);total+=n
    # Non-possessive BBC service tails left after a parent source was consumed.
    extra=(r"(?:World\s+Service(?:['’]s\s+(?:Newshour|Newsday)\s+programme|\s+programme\s+(?:Newshour|Newsday))?|"
           r"Radio(?:\s+4)?['’]s\s+Good\s+Morning\s+Scotland\s+programme|"
           r"Radio\s+4['’]s\s+(?:World\s+Tonight|(?:The\s+)?World\s+at\s+One)\s+programme)")
    text,n=re.subn(rf"\b(said|reported)\s+{extra}\b",lambda m:m.group(1),text,flags=re.I);total+=n
    # A few source-context variants observed in the audit.
    text,n=re.subn(r"\btold\s+(?:the\s+)?Sunday\s+with\s+Laura\s+Kuenssberg(?:\s+programme)?\b","said",text,flags=re.I);total+=n
    text,n=re.subn(r"\bspeaking\s+in\s+an\s+interview\s+One['’]s\s+Sunday\s+with\s+Laura\s+Kuenssberg\s+programme\b","speaking in an interview",text,flags=re.I);total+=n
    text,n=re.subn(r"\bsaid\s+One['’]s\s+Sunday\s+with\s+Laura\s+Kuenssberg\s+programme\s+that\b","said that",text,flags=re.I);total+=n
    text,n=re.subn(r"\bsaid\s+One['’]s\s+Sunday\s+with\s+Laura\s+Kuenssberg\s+programme\b","said",text,flags=re.I);total+=n
    text,n=re.subn(r"\bspeaking\s+to\s+reporters['’]s\s+Gaza\s+Today\s+podcast\b","speaking to reporters",text,flags=re.I);total+=n
    text,n=re.subn(r"\bspeaking\s+to\s+reporters['’]s\s+Laura\s+Kuenssberg\b","speaking to reporters",text,flags=re.I);total+=n
    text,n=re.subn(r"\bin\s+an\s+interview['’]s\s+(?:US\s+(?:news\s+)?partner\s+CBS(?:\s+News)?|Laura\s+Kuenssberg)\b","in an interview",text,flags=re.I);total+=n
    text,n=re.subn(r"\bsaid\s+Newshour\b","said",text,flags=re.I);total+=n
    text,n=re.subn(r"\btold\s+Gaza\s+Today\b","said",text,flags=re.I);total+=n
    text,n=re.subn(r"\btold\s+Radio\s+4['’]s\s+(?:World\s+Tonight|(?:The\s+)?World\s+[Aa]t\s+One)\s+programme\b","said",text,flags=re.I);total+=n
    text,n=re.subn(r"\bSpeaking\s+on\s+Radio\s+4['’]s\s+PM\s+programme\b","Speaking in an interview",text,flags=re.I);total+=n
    # Impossible generic possessive verbs must never survive. Do not guess at
    # arbitrary prose; leave a residual for the hard identity gate if unmatched.
    return clean_punctuation_gaps(normalize_spaces(text)),total


# Save the prior final cleaner and wrap it with the V4 scar repair. The prior
# cleaner still handles recommendation widgets, boilerplate, target attributions,
# structural bylines, URLs and formatting globally/label-blindly.
_clean_text_global_before_v4 = clean_text_global
def clean_text_global(text: str, title_lookup, author_matcher=None, article_author: str='') -> tuple[str,dict]:
    cleaned,stats=_clean_text_global_before_v4(text,title_lookup,author_matcher,article_author)
    cleaned,n=repair_target_source_scars_v4(cleaned);stats['target_source_scars_repaired_v4']=n
    cleaned=clean_punctuation_gaps(normalize_spaces(cleaned))
    return cleaned,stats


# Expanded hard gate: a grammatical scar or distinctive target programme in
# source context is a failure, even if the parent outlet name has disappeared.
_TARGET_PROGRAM_CONTEXT_V4=re.compile(
    rf"(?:\b(?:told|said|reported|speaking\s+(?:to|on)|appeared\s+on|according\s+to|interview(?:ed)?\s+(?:by|with)|on)\s+(?:the\s+)?{_TARGET_SOURCE_TAIL_V4}\b)|"
    rf"(?:\b{_TARGET_SOURCE_TAIL_V4}\s+(?:programme|program|podcast|show|radio\s+service)\b)",re.I)
_GRAMMAR_SCAR_V4=re.compile(r"\b(?:said|told|reported|wrote|added|noted|explained|found|stated|confirmed|speaking|interview|reports)['’]s\b|\breporters['’]s\b|\breports\s+teams?\s+on\s+the\s+ground\b|\breports\s+indicated\s+its\s+(?:correspondent|reporter|journalist|photographer)\b|__TARGET_REPORTER_ATTR__",re.I)
_unresolved_before_v4=unresolved_target_identity
def unresolved_target_identity(text: str) -> str|None:
    old=_unresolved_before_v4(text)
    if old: return old
    if _TARGET_PROGRAM_CONTEXT_V4.search(text or ''): return 'target_owned_service_context_v4'
    if _GRAMMAR_SCAR_V4.search(text or ''): return 'cleaning_grammatical_scar'
    return None


# Topic A: broaden the domestic/legal guard for Palestine-identity stories such
# as a journalist detained/deported abroad where the war is advocacy context.
_topic_a_before_v4=topic_a_relevant
def topic_a_relevant(title: str, body: str) -> tuple[bool,str]:
    t=normalize_spaces(title)
    high=bool(A_HIGH_RE.search(t)); pal=bool(A_PALESTINE_RE.search(t))
    if pal and not high and re.search(r"\b(?:arrest|arrested|court|ban|banned|proscription|trial|plead|charge|charged|police|rally|protest|demonstration|flag|activist|solidarity|hunger strike|campus|university|deport|deported|detain|detained|journalist|journalism)\w*\b",t,re.I):
        return False,'A_domestic_pro_palestine_protest_or_legal_story'
    return _topic_a_before_v4(title,body)


# Topic B: reject the observed crypto retrieval/human-interest case and require
# generic budget/jobs headlines to carry genuine economic context.
_topic_b_before_v4=topic_b_relevant
def topic_b_relevant(title: str, body: str, portal: str, url: str) -> tuple[bool,str]:
    t=normalize_spaces(title); lead=first_n_words(body,TOPIC_LEAD_WORDS)
    if re.search(r"\b(?:crypto(?:currency)?|bitcoin|Polymarket)\b",t,re.I) and re.search(r"\b(?:hard drive|landfill|retriev\w*|lost|thrown away|wallet|gambler|gambling|bet|betting|wager|hack\w*|heist\w*|kidnapp?\w*|theft|stolen|scam|fraud)\b",t,re.I):
        return False,'B_crypto_gambling_or_human_interest_story'
    # Generic culture/sector budget stories should not qualify from 'budget' alone.
    if re.search(r"\bbudget\b",t,re.I) and re.search(r"\b(?:arts?|culture|cultural|museum|theatre|opera|gallery|festival)\b",t,re.I) and not re.search(r"\b(?:econom(?:y|ic)|tax|fiscal|deficit|debt|inflation|growth|jobs?|business)\b",t,re.I):
        return False,'B_sector_budget_story_without_broader_economic_anchor'
    # Immigration/enforcement stories containing 'jobs' are not labour-market
    # coverage unless an independent economic/labour-market anchor is present.
    if re.search(r"\b(?:asylum|migrant|immigration|Home Office)\b",t,re.I) and re.search(r"\bjobs?\b",t,re.I) and not re.search(r"\b(?:econom(?:y|ic)|labou?r market|unemployment|wages?|job (?:cuts?|losses?|market|creation))\b",t,re.I):
        return False,'B_immigration_jobs_story_without_labour_market_anchor'
    return _topic_b_before_v4(title,body,portal,url)


def topic_relevant(topic: str,title: str,body: str,portal: str,url: str)->tuple[bool,str]:
    if topic=='topicA': return topic_a_relevant(title,body)
    if topic=='topicB': return topic_b_relevant(title,body,portal,url)
    return False,'unknown_topic'


# ========================================================================== 
# FINAL V5 OVERRIDES
# Added after end-to-end review of the V4 local run supplied by the user.
# V5 is intentionally narrow: remove structural trailing wire-service credits,
# close two observed Topic-B false-positive classes, and exclude live trackers.
# ========================================================================== 

# Structural end credits such as "With Agence France-Presse and Reuters" are
# website/reporting-credit furniture, not article prose. Remove only when they
# occur as a trailing credit; in-text Reuters/AP/AFP sourcing remains genuine
# journalistic content.
_WIRE_CREDIT_NAME_V5 = r"(?:Reuters|Associated\s+Press|Agence\s+France[- ]Presse|AFP)"
_TRAILING_WIRE_CREDIT_V5 = re.compile(
    rf"(?:\s+|^)[-–—]?\s*(?:With|Additional\s+reporting\s+by)\s+{_WIRE_CREDIT_NAME_V5}"
    rf"(?:\s*(?:,|and|&)\s*{_WIRE_CREDIT_NAME_V5})*\s*\.?\s*$",
    re.I,
)

def remove_trailing_wire_credit_v5(text: str) -> tuple[str,int]:
    text,n=_TRAILING_WIRE_CREDIT_V5.subn('',text or '')
    return clean_punctuation_gaps(normalize_spaces(text)),n
_clean_text_global_before_v5 = clean_text_global
def clean_text_global(text: str, title_lookup, author_matcher=None, article_author: str='') -> tuple[str,dict]:
    cleaned,stats=_clean_text_global_before_v5(text,title_lookup,author_matcher,article_author)
    before=len(cleaned)
    cleaned,n=remove_trailing_wire_credit_v5(cleaned)
    stats['trailing_wire_credits_removed_v5']=n
    stats['chars_removed_v5_wire_credit']=max(0,before-len(cleaned))
    cleaned=clean_punctuation_gaps(normalize_spaces(cleaned))
    return cleaned,stats

# Live tracker/data-dashboard pages are not comparable ordinary written-news
# articles even if their URL happens to look like a dated news article.
_eligible_genre_before_v5 = eligible_genre
def eligible_genre(portal: str, raw_title: str) -> tuple[bool,str]:
    t=normalize_spaces(raw_title)
    if re.search(r"\blive\s+(?:tracker|updates?)\b",t,re.I):
        return False,'live_tracker_or_updates_format'
    return _eligible_genre_before_v5(portal,raw_title)

# Close two false-positive senses observed in the 420-example V4 manual audit.
_topic_b_before_v5 = topic_b_relevant
def topic_b_relevant(title: str, body: str, portal: str, url: str) -> tuple[bool,str]:
    t=normalize_spaces(title)
    # "economy" as an airline cabin/fare class is not the economy topic.
    if re.search(r"\beconomy\s+(?:class|passengers?|seats?|cabin|fares?|tickets?)\b",t,re.I) or \
       re.search(r"\b(?:passengers?|seats?|cabin)\b.{0,35}\beconomy\b",t,re.I):
        return False,'B_airline_economy_class_non_economic_sense'
    # Crypto-company criminal/custody/personnel stories are legal/criminal unless
    # the headline independently anchors markets/economic policy.
    if re.search(r"\b(?:crypto(?:currency)?|bitcoin|FTX|Binance)\b",t,re.I) and \
       re.search(r"\b(?:custody|prison|jail|released?|sentenc\w*|convict\w*|guilty|charges?|charged|trial|arrest\w*|fraud|money\s+laundering)\b",t,re.I) and \
       not INDEPENDENT_ECON_POLICY_RE.search(t):
        return False,'B_crypto_legal_criminal_story_without_independent_economic_signal'
    return _topic_b_before_v5(title,body,portal,url)

# ========================================================================== 
# FINAL V6 OVERRIDES
# Final QA fixes: preserve ordinary passive "told" grammar and close the few
# residual off-topic legal/criminal edge cases found by targeted full-corpus audit.
# ========================================================================== 
_topic_a_before_v6 = topic_a_relevant
def topic_a_relevant(title: str, body: str) -> tuple[bool,str]:
    t=normalize_spaces(title)
    high=bool(A_HIGH_RE.search(t)); pal=bool(A_PALESTINE_RE.search(t))
    # Domestic/historical crime cases whose Palestine identity is the hook, but
    # which are not reporting on the target conflict itself.
    if pal and not high and re.search(r"\b(?:murder|murdered|hate crime|convict|convicted|landlord)\b",t,re.I):
        return False,'A_external_palestine_crime_story'
    # External criminal prosecutions of Hezbollah membership are not the current
    # Israel-Lebanon conflict merely because the lead provides group background.
    if re.search(r"\bHezbollah\b",t,re.I) and re.search(r"\b(?:trial|court|charged|prosecution|arrested?)\b",t,re.I) and \
       not re.search(r"\b(?:Israel|Israeli|Lebanon|Lebanese|war|attack|strike|ceasefire|conflict)\b",t,re.I):
        return False,'A_external_hezbollah_legal_story'
    return _topic_a_before_v6(title,body)

_topic_b_before_v6 = topic_b_relevant
def topic_b_relevant(title: str, body: str, portal: str, url: str) -> tuple[bool,str]:
    t=normalize_spaces(title)
    # Extend the crypto criminal guard to ordinary morphology observed in QA
    # (e.g. "crypto criminals ... launder ... drug gangs").
    if re.search(r"\b(?:crypto(?:currency)?|bitcoin|FTX|Binance)\b",t,re.I) and \
       re.search(r"\b(?:criminals?|crime|drug gangs?|launder(?:ing|ed|s)?|dirty money|custody|prison|jail|released?|sentenc\w*|convict\w*|guilty|charges?|charged|trial|arrest\w*|fraud)\b",t,re.I) and \
       not INDEPENDENT_ECON_POLICY_RE.search(t):
        return False,'B_crypto_legal_criminal_story_without_independent_economic_signal'
    return _topic_b_before_v6(title,body,portal,url)

def topic_relevant(topic: str,title: str,body: str,portal: str,url: str)->tuple[bool,str]:
    if topic=='topicA': return topic_a_relevant(title,body)
    if topic=='topicB': return topic_b_relevant(title,body,portal,url)
    return False,'unknown_topic'

# ========================================================================== 
# FINAL V7 OVERRIDES
# Narrow final-QA patch after full V6 review.
_topic_a_before_v7 = topic_a_relevant

def topic_a_relevant(title: str, body: str) -> tuple[bool, str]:
    t = normalize_spaces(title)
    # Domestic US/other legislation aimed at pro-Palestine organisations is not
    # itself reporting on the Israel/Palestine conflict. This closes the observed
    # anti-NGO/tax-exempt-status false positive without broadening the exclusion
    # to ordinary diplomacy, statehood, aid, sanctions, or conflict policy.
    if re.search(r"\bpro[- ]Palestin(?:e|ian)\b", t, re.I) and re.search(
        r"\b(?:bill|legislation|legislative|Congress|House|Senate|NGOs?|nonprofits?|non-profits?|tax[- ]exempt)\b",
        t, re.I,
    ):
        return False, "A_domestic_pro_palestine_legislation_or_NGO_story"
    return _topic_a_before_v7(title, body)


def topic_relevant(topic: str, title: str, body: str, portal: str, url: str) -> tuple[bool, str]:
    if topic == "topicA":
        return topic_a_relevant(title, body)
    if topic == "topicB":
        return topic_b_relevant(title, body, portal, url)
    return False, "unknown_topic"


# V9 final-pass capitalization. Some later scar-repair wrappers (V4/V5) run
# after the core cleaner's capitalization step and can themselves create a
# generic lower-case attribution at a sentence boundary. Re-run this narrowly
# at the very end so synthetic phrases never leave model text malformed.
_clean_text_global_before_v9_final = clean_text_global
_V9_GENERATED_STARTS_FINAL = (
    r"(?:reports indicated|reports cited|according to reports|speaking to reporters|"
    r"speaking in an interview|in an interview|interviewed by reporters|appeared in an interview|"
    r"broadcast footage|an investigation|an analysis|an Ipsos poll|a YouGov poll|"
    r"a reporter|a journalist|a photographer)"
)
_V9_GENERATED_SENTENCE_START_RE = re.compile(
    rf"(^|[.!?]\s+)({_V9_GENERATED_STARTS_FINAL})\b", re.I
)

def clean_text_global(text: str, title_lookup, author_matcher=None, article_author: str='') -> tuple[str,dict]:
    cleaned, stats = _clean_text_global_before_v9_final(text, title_lookup, author_matcher, article_author)
    def _cap(m):
        phrase = m.group(2)
        return m.group(1) + phrase[:1].upper() + phrase[1:]
    cleaned, n = _V9_GENERATED_SENTENCE_START_RE.subn(_cap, cleaned)
    stats['generated_sentence_starts_capitalized_v9_final'] = n
    cleaned = clean_punctuation_gaps(normalize_spaces(cleaned))
    return cleaned, stats



# ==========================================================================
# FINAL V10 OVERRIDES
# Final freeze-candidate fixes found by manual inspection of the V9 output.
# These are deliberately conservative:
#   * reject residual unambiguous Guardian-owned brand/self-reference contexts
#     rather than trying to rewrite them;
#   * reject BBC "More from InDepth" page furniture;
#   * close three clearly external/secondary Topic-A false-positive patterns
#     observed in the fresh 420-row QA sample.
# The cleaner remains label-blind and the main model input remains body-only.
# ==========================================================================

# Bare "guardian" is an ordinary English noun, so do NOT treat it as identity.
# These contexts, however, are unambiguously the target publication/brand.
_GUARDIAN_MEDIA_CONTEXT_V10 = re.compile(
    r"\bGuardian\s+(?:Australia|US|Essential|investigation|interview|readers?|poll|"
    r"newspaper|reporter|correspondent|journalist|columnist|editor)\b|"
    r"\bHarris\s*/\s*Guardian\b",
    re.IGNORECASE,
)
_BBC_INDEPTH_FURNITURE_V10 = re.compile(r"\bMore\s+from\s+InDepth\b", re.IGNORECASE)

_unresolved_target_identity_before_v10 = unresolved_target_identity
def unresolved_target_identity(text: str) -> str|None:
    old = _unresolved_target_identity_before_v10(text)
    if old:
        return old
    if _GUARDIAN_MEDIA_CONTEXT_V10.search(text or ""):
        return "guardian_media_brand_context_v10"
    if _BBC_INDEPTH_FURNITURE_V10.search(text or ""):
        return "bbc_indepth_page_furniture_v10"
    return None


_topic_a_relevant_before_v10 = topic_a_relevant
def topic_a_relevant(title: str, body: str) -> tuple[bool, str]:
    t = normalize_spaces(title)

    # External/local hate-crime or graffiti cases in which Israel is merely the
    # object of a slogan are not reporting on the target conflict itself.
    if re.search(r"\banti[- ]Israel\b", t, re.IGNORECASE) and re.search(
        r"\b(?:hate[- ]crime|graffiti|vandal(?:ism|ised|ized)?|church)\b", t, re.IGNORECASE
    ) and not re.search(
        r"\b(?:Gaza|West Bank|Hamas|Hezbollah|Lebanon|IDF|"
        r"Israeli\s+(?:attack|strike|raid)|war|ceasefire|hostage|settler)\w*\b",
        t, re.IGNORECASE,
    ):
        return False, "A_external_anti_israel_hate_crime_or_graffiti_story"

    # Celebrity/social-media apology/backlash stories are secondary-domain
    # reactions to the conflict rather than conflict reporting.
    if re.search(r"\b(?:sorry|apolog(?:y|ise|ised|ize|ized)|backlash)\b", t, re.IGNORECASE) and \
       re.search(r"\b(?:Gaza|Israel|Palestin(?:e|ian))\b", t, re.IGNORECASE) and \
       re.search(r"\b(?:video|Instagram|TikTok|post|comedian|actor|actress|celebrity)\b", t, re.IGNORECASE):
        return False, "A_external_celebrity_or_social_media_reaction_story"

    # A headline whose event is explicitly a verbal gaffe/slip-up is not itself
    # conflict reporting even if the underlying speech mentions Gaza policy.
    if re.search(r"\b(?:gaffe|slip[- ]?up)\b", t, re.IGNORECASE) and \
       re.search(r"\b(?:Gaza|Israel|Palestin(?:e|ian)|hostages?)\b", t, re.IGNORECASE):
        return False, "A_secondary_political_gaffe_story"

    return _topic_a_relevant_before_v10(title, body)


def topic_relevant(topic: str, title: str, body: str, portal: str, url: str) -> tuple[bool, str]:
    if topic == "topicA":
        return topic_a_relevant(title, body)
    if topic == "topicB":
        return topic_b_relevant(title, body, portal, url)
    return False, "unknown_topic"


# ==========================================================================
# FINAL V11 OVERRIDES
# Freeze-candidate QA fixes found by full inspection of the V10 output.
# These changes remain global/label-blind and are restricted to unmistakable
# non-content metadata, direct target-brand residue, and a few narrow scope
# false-positive classes.
# ==========================================================================

# End-of-article editorial credits / correction notes are page metadata rather
# than article prose. V10 still retained examples such as:
#   "Additional reporting by ...", "Additional research by ...", "Graphics by ...",
#   "Correction 6 March 2025: ...", "Clarification ...:", "Update ...:",
#   "This article is published in collaboration with ...", and "With agencies".
# Cut from the earliest HIGH-PRECISION marker in the final tail. Similar words
# used normally inside article prose are untouched.
_V11_TAIL_CORE_PATTERNS = [
    re.compile(r"\bAdditional\s+reporting\s+(?:by|from)\b", re.I),
    re.compile(r"\bAdditional\s+research\s+by\b", re.I),
    re.compile(r"\bGraphics\s+by\b", re.I),
    re.compile(r"\bVisual\s+journalism\s+by\b", re.I),
    re.compile(r"\b(?:Photo|Photography|Photographs)\s+by\b", re.I),
    re.compile(
        r"\bCorrection(?:\s+(?:\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?|"
        r"[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?))?\s*:", re.I
    ),
    re.compile(
        r"\bClarification(?:\s+(?:\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?|"
        r"[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?))?\s*:", re.I
    ),
    re.compile(
        r"\bUpdate(?:,\s*|\s+)?(?:\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?|"
        r"[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?)?\s*:", re.I
    ),
    re.compile(r"\bFinding\s+\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?\s*:", re.I),
    re.compile(
        r"\bThis\s+(?:article|piece|story)\s+(?:is|was)\s+"
        r"(?:published\s+in\s+collaboration\s+with|co-published\s+with|"
        r"produced\s+for|produced\s+in\s+partnership\s+with)\b", re.I
    ),
    re.compile(r"\bAn\s+earlier\s+version\s+of\s+this\s+article\s+appeared\s+in\b", re.I),
    re.compile(r"\bWith\s+agencies\b", re.I),
]
_V11_AGENCY_CONTRIBUTED_RE = re.compile(
    r"\b(?:Reuters|Associated\s+Press|AP|AFP|Agence\s+France[- ]Presse)\b"
    r"[^.!?]{0,120}\bcontributed\s+to\s+this\s+story\b", re.I
)

def strip_tail_editorial_metadata_v11(text: str, tail_chars: int = 2800) -> tuple[str, int]:
    s = text or ""
    if not s:
        return s, 0
    cut_start = max(0, len(s) - tail_chars)
    tail = s[cut_start:]
    candidates = []

    def sentence_boundary_ok(abs_pos: int) -> bool:
        before = s[:abs_pos].rstrip()
        return (not before) or before[-1] in ".!?"

    for pat in _V11_TAIL_CORE_PATTERNS:
        for m in pat.finditer(tail):
            abs_pos = cut_start + m.start()
            if not sentence_boundary_ok(abs_pos):
                continue
            # "With agencies" is only metadata when it is itself the tail.
            if re.match(r"(?i)With\s+agencies\b", m.group(0)):
                if s[abs_pos:].strip().rstrip(".") != m.group(0).strip().rstrip("."):
                    continue
            candidates.append(abs_pos)
            break

    # Credits like "Jason Burke and the Associated Press contributed to this
    # story" start before the agency token. Backtrack to the sentence boundary.
    m = _V11_AGENCY_CONTRIBUTED_RE.search(tail)
    if m:
        abs_m = cut_start + m.start()
        prefix = s[:abs_m]
        prev = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
        sent_start = 0 if prev < 0 else prev + 1
        # Only accept when the marker is genuinely in the tail.
        if sent_start >= max(0, len(s) - tail_chars):
            candidates.append(sent_start)

    if not candidates:
        return normalize_spaces(s), 0
    earliest = min(candidates)
    return clean_punctuation_gaps(normalize_spaces(s[:earliest])), 1


_clean_text_global_before_v11 = clean_text_global
def clean_text_global(text: str, title_lookup, author_matcher=None, article_author: str='') -> tuple[str,dict]:
    cleaned, stats = _clean_text_global_before_v11(text, title_lookup, author_matcher, article_author)
    before = len(cleaned)
    cleaned, n = strip_tail_editorial_metadata_v11(cleaned)
    stats["tail_editorial_metadata_removed_v11"] = n
    stats["chars_removed_v11_tail_metadata"] = max(0, before - len(cleaned))

    # Two raw Guardian pages contain the literal typo "said said". This is a
    # source typo rather than a modelling signal; collapse only this exact,
    # mechanically duplicated reporting verb.
    def _dedup_said(m):
        return "Said" if m.group(0)[:1].isupper() else "said"
    cleaned, k = re.subn(r"\bsaid\s+said\b", _dedup_said, cleaned, flags=re.I)
    stats["duplicate_said_repaired_v11"] = k

    cleaned = clean_punctuation_gaps(normalize_spaces(cleaned))
    return cleaned, stats


# V10 correctly handles "Al Jazeera" with a space, but one self-attribution
# survived as "told Al-Jazeera". A blanket ban on hyphenated Al-Jazeera would
# incorrectly reject the unrelated Syrian "Al-Jazeera barracks", so only
# unmistakable MEDIA/source contexts are treated as target identity.
_ALJAZEERA_HYPHEN_MEDIA_V11 = re.compile(
    r"(?:\b(?:told|according\s+to|speaking\s+to|interview(?:ed)?\s+(?:by|with)|"
    r"reported\s+by|writing\s+for)\s+(?:the\s+)?Al[-\u2010-\u2015\s]+Jazeera\b)|"
    r"(?:\bAl[-\u2010-\u2015\s]+Jazeera\s+(?:reporters?|journalists?|correspondents?|"
    r"colleague|network|channel|offices?|cameramen?|media)\b)",
    re.I,
)

_unresolved_target_identity_before_v11 = unresolved_target_identity
def unresolved_target_identity(text: str) -> str|None:
    old = _unresolved_target_identity_before_v11(text)
    if old:
        return old
    if _ALJAZEERA_HYPHEN_MEDIA_V11.search(text or ""):
        return "aljazeera_hyphen_media_identity_v11"
    return None


# Topic A: the earlier domestic pro-Palestine protest guard covered protest,
# rally and demonstration but omitted the synonymous "march". Keep explicit
# Gaza/war/conflict marches when a high-specificity conflict anchor is present.
_topic_a_relevant_before_v11 = topic_a_relevant
def topic_a_relevant(title: str, body: str) -> tuple[bool, str]:
    t = normalize_spaces(title)
    high = bool(A_HIGH_RE.search(t))
    pal = bool(A_PALESTINE_RE.search(t))
    if pal and not high and re.search(r"\bmarch(?:es|ed|ing)?\b", t, re.I):
        return False, "A_domestic_pro_palestine_march_story"
    return _topic_a_relevant_before_v11(title, body)


# Topic B: two fresh QA rows were clearly politics/human-interest stories whose
# only economic-looking hook was personal side income or the timing of a party
# leadership contest around a budget. These narrow guards do not affect genuine
# fiscal/economic policy coverage.
_topic_b_relevant_before_v11 = topic_b_relevant
_V11_REAL_ECON_ANCHOR_RE = re.compile(
    r"\b(?:econom(?:y|ic|ics)|tax(?:es|ation)?|fiscal|deficit|debt|inflation|growth|"
    r"unemployment|wages?|jobs?\s+(?:cuts?|losses?|market|creation)|business|trade|"
    r"tariffs?|interest\s+rates?|public\s+finances?|spending|investment|GDP)\b", re.I
)

def topic_b_relevant(title: str, body: str, portal: str, url: str) -> tuple[bool, str]:
    t = normalize_spaces(title)
    if re.search(r"\b(?:MPs?|lawmakers?|politicians?)\b", t, re.I) and \
       re.search(r"\b(?:side\s+jobs?|side\s+hustles?|outside\s+earnings?)\b", t, re.I) and \
       not _V11_REAL_ECON_ANCHOR_RE.search(t):
        return False, "B_political_side_income_story"
    if re.search(r"\b(?:party\s+)?leadership\s+(?:contest|race)\b", t, re.I) and \
       re.search(r"\bbudget\b", t, re.I) and not _V11_REAL_ECON_ANCHOR_RE.search(t):
        return False, "B_party_leadership_budget_timing_story"
    return _topic_b_relevant_before_v11(title, body, portal, url)


def topic_relevant(topic: str, title: str, body: str, portal: str, url: str) -> tuple[bool, str]:
    if topic == "topicA":
        return topic_a_relevant(title, body)
    if topic == "topicB":
        return topic_b_relevant(title, body, portal, url)
    return False, "unknown_topic"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/articles_raw.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--report-dir", default="outputs/corpus_build")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_BODY_WORDS)
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    start_dt = date_bound(args.start)
    end_dt = date_bound(args.end, end=True)

    print(f"Reading {inp} ...")
    rows = []
    with inp.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                o = json.loads(line)
            except Exception as e:
                print(f"WARNING: JSON parse failure at line {line_no}: {e}", file=sys.stderr)
                continue
            o["_line_no"] = line_no
            url = o.get("final_url") or o.get("url") or f"line:{line_no}"
            o["_article_id"] = sha1_text(url)[:16]
            rows.append(o)

    print(f"Loaded {len(rows):,} raw rows")
    author_matcher, global_author_names = build_author_matcher(rows)
    title_lookup = build_title_lookup(rows)
    print(f"Global author identities catalogued for structural auditing: {len(global_author_names):,}")

    (report_dir / "global_author_names.txt").write_text(
        "\n".join(global_author_names) + "\n", encoding="utf-8"
    )

    stage_counts = Counter()
    reject_reason_counts = Counter()
    rejects: list[Reject] = []
    candidates: list[dict] = []
    before_after_pool: dict[tuple[str, str], list[dict]] = defaultdict(list)

    stage_counts["raw"] = len(rows)

    for o in rows:
        aid = o["_article_id"]
        portal = o.get("portal") or ""
        topic = o.get("topic") or ""
        url = o.get("final_url") or o.get("url") or ""
        raw_title = o.get("title") or ""
        raw_text = o.get("text") or ""

        def reject(stage: str, reason: str) -> None:
            rejects.append(Reject(aid, portal, topic, url, raw_title, stage, reason))
            reject_reason_counts[f"{stage}:{reason}"] += 1

        if o.get("error") is not None:
            reject("fetch", str(o.get("error")))
            continue
        stage_counts["after_successful_fetch"] += 1

        dt = parse_utc_date(o.get("date_published"))
        if dt is None or not (start_dt <= dt <= end_dt):
            reject("date", "missing_or_outside_intended_window")
            continue
        stage_counts["after_date"] += 1

        ok_page, page_reason = eligible_page_type(portal, url)
        if not ok_page:
            reject("page_type", page_reason)
            continue
        stage_counts["after_page_type"] += 1

        ok_genre, genre_reason = eligible_genre(portal, raw_title)
        if not ok_genre:
            reject("genre", genre_reason)
            continue
        stage_counts["after_genre"] += 1

        # V7 extraction-integrity gate: BBC.com text in this scrape prepends the
        # visible H1. Remove it only when the boundary is high-confidence; reject
        # unresolved cases rather than accidentally feeding headlines to body-only models.
        body_source, leading_heading_method, leading_heading_removed = remove_known_leading_extraction_heading(
            raw_text, raw_title, portal, url
        )
        if body_source is None:
            reject("extraction_integrity", leading_heading_method)
            continue
        stage_counts["after_extraction_integrity"] += 1

        # Preclean obvious extraction widgets before relevance scoring, otherwise
        # recommendation text can falsely make an unrelated article match a topic.
        title_pre = preclean_for_relevance(raw_title, title_lookup)
        body_pre = preclean_for_relevance(body_source, title_lookup)

        is_rel, rel_reason = topic_relevant(topic, title_pre, body_pre, portal, url)
        if not is_rel:
            reject("topic_relevance", rel_reason)
            continue
        stage_counts["after_topic_relevance"] += 1

        title_clean, title_stats = clean_text_global(raw_title, title_lookup, author_matcher, o.get("author") or "")
        body_clean, body_stats = clean_text_global(body_source, title_lookup, author_matcher, o.get("author") or "")
        body_clean, repeated = remove_repeated_title(body_clean, title_clean)
        body_clean = clean_punctuation_gaps(normalize_spaces(body_clean))
        title_clean = clean_punctuation_gaps(normalize_spaces(title_clean))
        title_clean = re.sub(r"^[\s:|,;\-]+", "", title_clean).strip()
        title_clean = re.sub(r"[\s|,:;\-]+$", "", title_clean).strip()

        # Hard post-clean identity gate: if an unmistakable target brand/service
        # survives safe neutralization, reject the article rather than aggressively
        # rewrite substantive prose. The check is identical for every portal.
        identity_reason = unresolved_target_identity(title_clean) or unresolved_target_identity(body_clean)
        if identity_reason:
            reject("identity_gate", identity_reason)
            continue
        stage_counts["after_identity_gate"] += 1

        body_words = len(word_list(body_clean))
        if body_words < args.min_words:
            reject("length", f"clean_body_under_{args.min_words}_words")
            continue
        stage_counts["after_min_length"] += 1

        text_hash = sha1_text(body_clean)
        # Main classifier input is BODY ONLY. The cleaned headline is retained
        # as metadata for audit/qualitative analysis, but excluding it from
        # model_text avoids the strong portal shortcut found in headline length.
        model_input = body_clean
        model_safe = {
            "article_id": aid,
            "portal": portal,
            "topic": topic,
            "year_month": dt.strftime("%Y-%m"),
            "title_clean": title_clean,
            "text_clean": body_clean,
            "model_text": model_input,
            "text_clean_sha1": text_hash,
        }
        audit = {
            **model_safe,
            "date_published": o.get("date_published"),
            "url": o.get("url"),
            "final_url": o.get("final_url"),
            "author_raw": o.get("author"),
            "sitename_raw": o.get("sitename"),
            "page_type_rule": page_reason,
            "genre_rule": genre_reason,
            "topic_relevance_rule": rel_reason,
            "raw_title": raw_title,
            "raw_text_prefix": raw_text[:1200],
            "body_source_prefix_after_extraction_gate": body_source[:1200],
            "clean_body_words": body_words,
            "clean_title_words": len(word_list(title_clean)),
            "clean_chars": len(body_clean),
            "title_repeated_at_body_start_removed": bool(repeated or leading_heading_removed),
            "leading_extraction_heading_method": leading_heading_method,
            "leading_extraction_heading_removed": leading_heading_removed,
            "title_cleaning_stats": title_stats,
            "body_cleaning_stats": body_stats,
            "body_fraction_chars_removed": (
                max(0.0, 1.0 - (len(body_clean) / max(1, len(raw_text))))
            ),
        }
        candidates.append({"model": model_safe, "audit": audit})

        key = (portal, topic)
        if len(before_after_pool[key]) < 200:
            before_after_pool[key].append({
                "article_id": aid,
                "portal": portal,
                "topic": topic,
                "title_before": raw_title,
                "title_after": title_clean,
                "body_before_prefix": raw_text[:700],
                "body_after_prefix": body_clean[:700],
            })

    # Exact dedup AFTER global cleaning. Identical clean bodies can emerge from
    # differently templated raw pages.
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for x in candidates:
        by_hash[x["model"]["text_clean_sha1"]].append(x)

    final_pairs: list[dict] = []
    exact_dup_groups = []
    for h, group in by_hash.items():
        if len(group) == 1:
            final_pairs.append(group[0])
            continue

        labels = {(g["model"]["portal"], g["model"]["topic"]) for g in group}
        exact_dup_groups.append({
            "text_clean_sha1": h,
            "n": len(group),
            "labels": sorted([f"{p}/{t}" for p, t in labels]),
            "article_ids": [g["model"]["article_id"] for g in group],
            "urls": [g["audit"].get("final_url") or g["audit"].get("url") for g in group],
        })

        if len(labels) > 1:
            # Cross-label identical text is ambiguous and unsafe: remove whole group.
            for g in group:
                rejects.append(Reject(
                    g["model"]["article_id"], g["model"]["portal"], g["model"]["topic"],
                    g["audit"].get("final_url") or g["audit"].get("url") or "",
                    g["audit"].get("raw_title") or "", "exact_dedup",
                    "identical_clean_body_cross_label",
                ))
                reject_reason_counts["exact_dedup:identical_clean_body_cross_label"] += 1
        else:
            # Same label: keep deterministic representative with most complete body,
            # then earliest line/article id as tie-breaker.
            keep = sorted(
                group,
                key=lambda g: (-g["audit"]["clean_chars"], g["audit"].get("date_published") or "9999", g["model"]["article_id"])
            )[0]
            final_pairs.append(keep)
            for g in group:
                if g is keep:
                    continue
                rejects.append(Reject(
                    g["model"]["article_id"], g["model"]["portal"], g["model"]["topic"],
                    g["audit"].get("final_url") or g["audit"].get("url") or "",
                    g["audit"].get("raw_title") or "", "exact_dedup",
                    "duplicate_clean_body_same_label",
                ))
                reject_reason_counts["exact_dedup:duplicate_clean_body_same_label"] += 1

    final_pairs.sort(key=lambda g: (g["model"]["topic"], g["model"]["portal"], g["model"]["year_month"], g["model"]["article_id"]))
    model_rows = [g["model"] for g in final_pairs]
    audit_rows = [g["audit"] for g in final_pairs]
    stage_counts["after_exact_dedup"] = len(model_rows)

    # Counts by portal/topic/month.
    by_cell = Counter((r["portal"], r["topic"]) for r in model_rows)
    by_month = Counter((r["portal"], r["topic"], r["year_month"]) for r in model_rows)

    # Random before/after samples, deterministic.
    rng = random.Random(args.seed)
    samples = []
    for key, pool in sorted(before_after_pool.items()):
        rng.shuffle(pool)
        samples.extend(pool[:12])

    out_model = out_dir / "corpus_clean_candidate.jsonl"
    out_audit = out_dir / "corpus_clean_audit_candidate.jsonl"
    write_jsonl(out_model, model_rows)
    write_jsonl(out_audit, audit_rows)

    write_csv(report_dir / "rejected_articles.csv", [asdict(r) for r in rejects])
    write_csv(report_dir / "before_after_samples.csv", samples)
    write_jsonl(report_dir / "exact_duplicate_groups.jsonl", exact_dup_groups)

    report = {
        "input": str(inp),
        "input_size_bytes": inp.stat().st_size,
        "input_sha256": sha256_file(inp),
        "random_seed": args.seed,
        "policy": {
            "date_start": args.start,
            "date_end": args.end,
            "minimum_clean_body_words": args.min_words,
            "topic_lead_words": TOPIC_LEAD_WORDS,
            "guardian_allowed_sections": sorted(GUARDIAN_ALLOWED_SECTIONS),
            "global_author_identities_catalogued": len(global_author_names),
            "author_policy": "remove article author only in structural byline/credit contexts; no global person-name erasure",
            "main_model_input": "body_only:model_text==text_clean; headline retained as metadata only",
            "bbc_com_extraction_policy": "retain BBC.com only when metadata title exactly matches the leading H1 prefix and the following body has a validated sentence start; reject alternate/ambiguous H1s",
            "aljazeera_extraction_policy_v9": "reject only high-confidence cases where a short embedded media/card headline is flattened immediately before an independently sentence-like article lead; do not guess/strip the boundary",
            "source_attribution_v9_policy": "possessive target-source/reporter forms are normalized before generic source replacement; strict sentence-bounded proper-name matching prevents cross-sentence deletion",
            "topicA_definition": "Israel/Palestine/Gaza and directly connected regional conflict, headline-anchored",
            "topicB_definition": "economy/finance/markets/fiscal-monetary/trade coverage; candidate label revalidated from headline + lead",
            "topicB_indirect_lead_min_distinct_concept_groups": 3,
            "identity_policy": "context-neutralize target source/program attributions; reject unresolved unmistakable target identities",
            "aljazeera_page_types": "dated /news/, /economy/, /features/ written articles; live/video/opinion/program excluded",
            "note": "Text cleaning is label-blind: clean_text_global never receives portal.",
        },
        "stage_counts": dict(stage_counts),
        "rejection_reasons": dict(reject_reason_counts),
        "final_counts_by_portal_topic": {
            f"{p}/{t}": n for (p, t), n in sorted(by_cell.items())
        },
        "final_counts_by_portal_topic_month": {
            f"{p}/{t}/{ym}": n for (p, t, ym), n in sorted(by_month.items())
        },
        "exact_duplicate_groups": len(exact_dup_groups),
        "output_model_safe": str(out_model),
        "output_audit": str(out_audit),
        "candidate_model_sha256": sha256_file(out_model),
        "candidate_audit_sha256": sha256_file(out_audit),
    }
    (report_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== BUILD COMPLETE ===")
    for k, v in stage_counts.items():
        print(f"{k:>28}: {v:,}")
    print("\nFinal counts:")
    for (p, t), n in sorted(by_cell.items()):
        print(f"  {p:10s} {t:7s}: {n:,}")
    print(f"\nModel-safe candidate: {out_model}")
    print(f"Audit candidate:      {out_audit}")
    print(f"Reports:              {report_dir}")
    print("\nIMPORTANT: do not split/train yet. Run the near-duplicate + leakage audit step next.")



# ==========================================================================
# FINAL V12 OVERRIDES
# Narrow freeze-candidate patch after full V11 end-to-end review.
#
# V12 does NOT reopen topic definitions or general content cleaning. It fixes
# only high-confidence mechanical artifacts observed in the actual final model
# text:
#   * one "A Jazeera" typo-variant in a target-source attribution;
#   * parenthetical wire/link abbreviations left after source neutralization,
#     e.g. "told Associated Press (AP)" -> "said (AP)";
#   * punctuation scars from removed "external:" annotations and source
#     attributions (",:", ",,", exact "..");
#   * a few source-attribution remnants such as "Reporting from X,:".
# All rules are label-blind and applied to every article.
# ==========================================================================

# Canonicalize the observed "A Jazeera" typo only in unmistakable source-
# attribution grammar, then let the existing target-source neutralizer handle it.
_normalize_target_attributions_before_v12 = normalize_target_attributions
_A_JAZEERA_ATTR_V12 = re.compile(
    r"\b(told|according\s+to|speaking\s+to|interviewed\s+by|interview\s+with)\s+"
    r"(?:the\s+)?A\s+Jazeera\b",
    re.IGNORECASE,
)
def normalize_target_attributions(text: str) -> tuple[str, int]:
    text, n0 = _A_JAZEERA_ATTR_V12.subn(
        lambda m: m.group(1) + " Al Jazeera", text or ""
    )
    text, n1 = _normalize_target_attributions_before_v12(text)
    return text, n0 + n1


# Parenthetical abbreviations/link markers can sit immediately after a source
# name in the extracted prose. If the source name is neutralized first, the
# parenthetical token becomes a malformed residue such as "said (AP)".
# Remove only these high-confidence parenthetical suffixes before the existing
# wire-attribution normalizer runs.
_normalize_wire_attributions_before_v12 = normalize_wire_attributions
_WIRE_PAREN_SUFFIX_V12 = re.compile(
    r"\b(Associated\s+Press)\s*\(\s*AP\s*\)|"
    r"\b(Agence\s+France[- ]Presse)\s*\(\s*AFP\s*\)",
    re.IGNORECASE,
)
def normalize_wire_attributions(text: str) -> tuple[str, int]:
    def _wire_paren_repl(m):
        return m.group(1) if m.group(1) is not None else m.group(2)
    text, n0 = _WIRE_PAREN_SUFFIX_V12.subn(_wire_paren_repl, text or "")
    text, n1 = _normalize_wire_attributions_before_v12(text)
    return text, n0 + n1


_clean_text_global_before_v12 = clean_text_global

# These exact structures are impossible/invalid ordinary prose and were traced
# to removal of a target reporter/source or extraction link annotation.
_REPORTING_PREFIX_COLON_V12 = re.compile(
    r"\bReporting\s+from\s+[^.!?]{1,120},\s*:\s*", re.IGNORECASE)
_REPORTING_PREFIX_DBL_COMMA_V12 = re.compile(
    r"\bReporting\s+from\s+[^.!?]{1,120},\s*,\s*", re.IGNORECASE)
_DBL_COMMA_REPORTING_TAIL_V12 = re.compile(
    r",\s*,\s*reporting\s+from\s+[^.!?]{1,120}\.", re.IGNORECASE)
_DBL_COMMA_ADDING_V12 = re.compile(
    r",\s*,\s*adding\s+that\b", re.IGNORECASE)
_TOLD_NEWS_AGENCY_SENTENCE_BREAK_V12 = re.compile(
    r"\btold\s+the\s+news\s+agency,\s+(?=[A-Z])", re.IGNORECASE)
_REPORTING_VERB_PAREN_TAG_V12 = re.compile(
    r"\b(said|says|wrote|reported|reports|noted|added|stated|confirmed|found)"
    r"\s+\((?:AP|AFP|PDF|ICRC)\)",
    re.IGNORECASE,
)

def clean_text_global(text: str, title_lookup, author_matcher=None, article_author: str='') -> tuple[str,dict]:
    cleaned, stats = _clean_text_global_before_v12(
        text, title_lookup, author_matcher, article_author
    )

    repair_count = 0

    # Remove dangling reporter-location prefixes only when the punctuation
    # proves the reporter/source itself was already deleted.
    cleaned, n = _REPORTING_PREFIX_COLON_V12.subn("", cleaned); repair_count += n
    cleaned, n = _REPORTING_PREFIX_DBL_COMMA_V12.subn("", cleaned); repair_count += n
    cleaned, n = _DBL_COMMA_REPORTING_TAIL_V12.subn(".", cleaned); repair_count += n

    # Source attribution removed between a quotation and "adding that".
    cleaned, n = _DBL_COMMA_ADDING_V12.subn(
        ". The report added that", cleaned
    ); repair_count += n

    # One malformed raw/source-attribution boundary: preserve the statement and
    # start the next capitalized sentence rather than keep "told the news agency,"
    # as a dangling clause.
    cleaned, n = _TOLD_NEWS_AGENCY_SENTENCE_BREAK_V12.subn(
        "said. ", cleaned
    ); repair_count += n

    # Parenthetical source/link tokens immediately following reporting verbs are
    # extraction/source-annotation residue, not prose.
    cleaned, n = _REPORTING_VERB_PAREN_TAG_V12.subn(
        lambda m: m.group(1), cleaned
    ); repair_count += n

    # "(PDF)" is a flattened hyperlink annotation when it survives elsewhere.
    cleaned, n = re.subn(r"\s*\(\s*PDF\s*\)", "", cleaned, flags=re.IGNORECASE)
    repair_count += n

    # Globally safe punctuation normalization. A double comma, comma-colon, or
    # exactly two periods is never intended as a portal/style feature here.
    cleaned, n = re.subn(r",\s*:", ":", cleaned); repair_count += n
    cleaned, n = re.subn(r",\s*,\s*", ", ", cleaned); repair_count += n
    cleaned, n = re.subn(r"(?<!\.)\.\.(?!\.)", ".", cleaned); repair_count += n

    stats["mechanical_scar_repairs_v12"] = repair_count
    cleaned = clean_punctuation_gaps(normalize_spaces(cleaned))
    return cleaned, stats

if __name__ == "__main__":
    main()
