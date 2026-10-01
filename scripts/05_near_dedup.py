#!/usr/bin/env python3
"""
Conservative near-duplicate detection for the cleaned candidate corpus.

The goal is NOT to merge independent reports of the same event. We only remove
articles that are extremely close textual copies / minor revisions.

Candidate generation uses bottom-k hashes of 5-word shingles. Candidate pairs
are then verified with full-text similarity (RapidFuzz if available, otherwise
SequenceMatcher). Only pairs above --remove-threshold are merged automatically.
A broader --report-threshold is written for manual inspection.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import hashlib
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable

try:
    from rapidfuzz.fuzz import ratio as rf_ratio
except Exception:
    rf_ratio = None
    from difflib import SequenceMatcher

STOP = {
    "the","a","an","and","or","but","of","to","in","on","for","with","as","at","by",
    "from","is","are","was","were","be","been","being","that","this","it","its","their",
    "his","her","they","he","she","we","you","has","have","had","will","would","could",
    "should","not","no","after","before","over","under","into","about","than","more","most",
}
WORD_RE = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def norm_words(text: str, limit: int = 1400) -> list[str]:
    return WORD_RE.findall((text or "").casefold())[:limit]


def sketch(text: str, k: int = 32, shingle_n: int = 5) -> tuple[int, ...]:
    words = norm_words(text)
    if len(words) < shingle_n:
        return ()
    vals = set()
    for i in range(len(words) - shingle_n + 1):
        sh = words[i:i+shingle_n]
        if sum(w not in STOP for w in sh) < 2:
            continue
        b = "\x1f".join(sh).encode("utf-8", errors="ignore")
        vals.add(zlib.crc32(b) & 0xffffffff)
    if not vals:
        return ()
    return tuple(heapq.nsmallest(k, vals))


def norm_for_ratio(text: str) -> str:
    return " ".join(norm_words(text, limit=100000))


def similarity(a: str, b: str) -> float:
    if rf_ratio is not None:
        return float(rf_ratio(a, b)) / 100.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b: self.p[b] = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-in", default="data/corpus_clean_candidate.jsonl")
    ap.add_argument("--audit-in", default="data/corpus_clean_audit_candidate.jsonl")
    ap.add_argument("--model-out", default="data/corpus_clean_dedup_candidate.jsonl")
    ap.add_argument("--audit-out", default="data/corpus_clean_dedup_audit_candidate.jsonl")
    ap.add_argument("--report-dir", default="outputs/near_dedup")
    ap.add_argument("--sketch-size", type=int, default=32)
    ap.add_argument("--min-shared-sketch", type=int, default=4)
    ap.add_argument("--report-threshold", type=float, default=0.94,
                    help="pairs at/above this similarity are written for review")
    ap.add_argument("--remove-threshold", type=float, default=0.985,
                    help="only extremely close pairs at/above this are auto-deduplicated")
    args = ap.parse_args()

    model = read_jsonl(Path(args.model_in))
    audit = read_jsonl(Path(args.audit_in))
    audit_by_id = {r["article_id"]: r for r in audit}
    if len(model) != len(audit_by_id):
        print("WARNING: model/audit sizes differ; joining by article_id")

    print(f"Loaded {len(model):,} cleaned articles")
    sketches = []
    inv = defaultdict(list)
    for i, r in enumerate(model):
        sk = sketch(r["text_clean"], k=args.sketch_size)
        sketches.append(sk)
        for h in sk:
            inv[h].append(i)

    # Count sketch collisions for candidate pairs. Ignore very large buckets,
    # which arise from generic language and are not useful for near-copy blocking.
    pair_shared = Counter()
    for inds in inv.values():
        if len(inds) < 2 or len(inds) > 80:
            continue
        for x in range(len(inds)):
            a = inds[x]
            for y in range(x + 1, len(inds)):
                b = inds[y]
                if a > b: a, b = b, a
                pair_shared[(a, b)] += 1

    candidates = [(pair, n) for pair, n in pair_shared.items() if n >= args.min_shared_sketch]
    print(f"Candidate pairs after sketch blocking: {len(candidates):,}")

    norm_cache = {}
    review_pairs = []
    dsu = DSU(len(model))
    review_dsu = DSU(len(model))
    auto_pairs = 0

    # V7: BBC.com and BBC.co.uk URLs with the same /news/articles/<id> path are
    # the same underlying article, even when different snapshots/extractors make
    # their text similarity fall below the generic 0.985 threshold. Force these
    # canonical same-article rows into one dedup component.
    canonical_bbc = defaultdict(list)
    for i, r in enumerate(model):
        if r.get("portal") != "bbc":
            continue
        a = audit_by_id.get(r.get("article_id"), {})
        u = a.get("final_url") or a.get("url") or ""
        path = (urlparse(u).path or "").rstrip("/").casefold()
        if re.fullmatch(r"/news/articles/[a-z0-9]+", path):
            canonical_bbc[path].append(i)

    canonical_group_rows = []
    canonical_duplicate_extra_rows = 0
    for path, inds in sorted(canonical_bbc.items()):
        if len(inds) < 2:
            continue
        canonical_duplicate_extra_rows += len(inds) - 1
        base = inds[0]
        for i in inds[1:]:
            dsu.union(base, i)
            review_dsu.union(base, i)
        canonical_group_rows.append({
            "canonical_path": path,
            "n": len(inds),
            "labels": ";".join(sorted({f"{model[i]['portal']}/{model[i]['topic']}" for i in inds})),
            "article_ids": ";".join(model[i]["article_id"] for i in inds),
            "urls": " || ".join((audit_by_id.get(model[i]["article_id"], {}).get("final_url") or
                                  audit_by_id.get(model[i]["article_id"], {}).get("url") or "") for i in inds),
            "titles": " || ".join(model[i]["title_clean"] for i in inds),
        })

    for (a, b), shared in candidates:
        ra, rb = model[a], model[b]
        # Very different lengths cannot be near-identical copies.
        la, lb = len(ra["text_clean"]), len(rb["text_clean"])
        if min(la, lb) / max(1, max(la, lb)) < 0.80:
            continue
        if a not in norm_cache: norm_cache[a] = norm_for_ratio(ra["text_clean"])
        if b not in norm_cache: norm_cache[b] = norm_for_ratio(rb["text_clean"])
        sim = similarity(norm_cache[a], norm_cache[b])
        if sim >= args.report_threshold:
            review_dsu.union(a, b)
            review_pairs.append({
                "article_id_a": ra["article_id"], "portal_a": ra["portal"], "topic_a": ra["topic"],
                "title_a": ra["title_clean"],
                "article_id_b": rb["article_id"], "portal_b": rb["portal"], "topic_b": rb["topic"],
                "title_b": rb["title_clean"],
                "shared_sketch_hashes": shared,
                "body_similarity": round(sim, 6),
                "cross_portal": ra["portal"] != rb["portal"],
                "cross_topic": ra["topic"] != rb["topic"],
            })
        if sim >= args.remove_threshold:
            dsu.union(a, b); auto_pairs += 1

    groups = defaultdict(list)
    for i in range(len(model)):
        groups[dsu.find(i)].append(i)

    keep_idx = set()
    removed = []
    group_report = []
    gid_num = 0
    for inds in groups.values():
        if len(inds) == 1:
            keep_idx.add(inds[0]); continue
        gid_num += 1
        labels = {(model[i]["portal"], model[i]["topic"]) for i in inds}
        group_report.append({
            "near_dup_group": gid_num,
            "n": len(inds),
            "labels": ";".join(sorted(f"{p}/{t}" for p,t in labels)),
            "article_ids": ";".join(model[i]["article_id"] for i in inds),
            "titles": " || ".join(model[i]["title_clean"] for i in inds),
        })
        if len(labels) > 1:
            # Ambiguous nearly-identical content across labels: remove entire component.
            for i in inds:
                removed.append((i, "near_duplicate_cross_label_remove_all"))
        else:
            # Same class/topic: BBC.co.uk extraction is structurally cleaner in
            # this scrape (no prepended visible H1). Prefer it whenever a
            # duplicate component contains a BBC.co.uk copy; otherwise keep the
            # most complete cleaned article as before.
            def keep_key(i):
                a = audit_by_id.get(model[i].get("article_id"), {})
                host = (urlparse(a.get("final_url") or a.get("url") or "").hostname or "").casefold()
                safer_bbc_host = int(model[i].get("portal") == "bbc" and host in {"bbc.co.uk", "www.bbc.co.uk"})
                return (safer_bbc_host, len(model[i]["text_clean"]), model[i]["article_id"])
            keep = max(inds, key=keep_key)
            keep_idx.add(keep)
            for i in inds:
                if i != keep:
                    removed.append((i, "near_duplicate_same_label_keep_one"))

    # Retained pairs in the broader review band (e.g. 0.94-0.985) are NOT
    # automatically deleted, but they must never be allowed to land in different
    # train/validation/test splits. Attach a deterministic split-group id for the
    # next stage. This preserves data while preventing revision leakage.
    surviving_indices = sorted(keep_idx)
    review_components = defaultdict(list)
    for i in surviving_indices:
        review_components[review_dsu.find(i)].append(i)
    split_group_by_idx = {}
    nontrivial_split_groups = 0
    for inds in review_components.values():
        ids = sorted(model[i]["article_id"] for i in inds)
        if len(inds) > 1:
            nontrivial_split_groups += 1
            digest = hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:16]
            gid = f"near_{digest}"
        else:
            gid = f"single_{ids[0]}"
        for i in inds:
            split_group_by_idx[i] = gid

    final_model = []
    for i in surviving_indices:
        r = dict(model[i])
        r["duplicate_split_group"] = split_group_by_idx[i]
        final_model.append(r)
    final_ids = {r["article_id"] for r in final_model}
    final_audit = []
    for i in surviving_indices:
        aid = model[i]["article_id"]
        if aid in audit_by_id:
            a = dict(audit_by_id[aid])
            a["duplicate_split_group"] = split_group_by_idx[i]
            final_audit.append(a)
    final_model.sort(key=lambda r:(r["topic"],r["portal"],r["year_month"],r["article_id"]))
    final_audit.sort(key=lambda r:(r["topic"],r["portal"],r["year_month"],r["article_id"]))

    out_model = Path(args.model_out); out_audit = Path(args.audit_out)
    write_jsonl(out_model, final_model); write_jsonl(out_audit, final_audit)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    review_pairs.sort(key=lambda r:r["body_similarity"], reverse=True)
    write_csv(report_dir/"near_duplicate_pairs_for_review.csv", review_pairs)
    write_csv(report_dir/"auto_near_duplicate_groups.csv", group_report)
    write_csv(report_dir/"canonical_bbc_url_duplicate_groups.csv", canonical_group_rows)
    removed_rows=[]
    for i,reason in removed:
        r=model[i]; a=audit_by_id.get(r["article_id"],{})
        removed_rows.append({
            "article_id":r["article_id"],"portal":r["portal"],"topic":r["topic"],
            "title":r["title_clean"],"url":a.get("final_url") or a.get("url"),"reason":reason,
        })
    write_csv(report_dir/"removed_near_duplicates.csv", removed_rows)

    counts=Counter((r["portal"],r["topic"]) for r in final_model)
    summary={
        "input_articles":len(model),
        "sketch_candidate_pairs":len(candidates),
        "review_pairs":len(review_pairs),
        "auto_pairs_at_remove_threshold":auto_pairs,
        "canonical_bbc_url_duplicate_groups":len(canonical_group_rows),
        "canonical_bbc_url_duplicate_extra_rows":canonical_duplicate_extra_rows,
        "auto_groups":len(group_report),
        "removed_articles":len(removed),
        "output_articles":len(final_model),
        "retained_nontrivial_duplicate_split_groups":nontrivial_split_groups,
        "report_threshold":args.report_threshold,
        "remove_threshold":args.remove_threshold,
        "counts_by_portal_topic":{f"{p}/{t}":n for (p,t),n in sorted(counts.items())},
        "rapidfuzz_used":rf_ratio is not None,
    }
    (report_dir/"near_dedup_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print("\n=== NEAR-DEDUP COMPLETE ===")
    print(json.dumps(summary, indent=2))
    print("\nReview near_duplicate_pairs_for_review.csv before freezing the corpus.")

if __name__ == "__main__":
    main()
