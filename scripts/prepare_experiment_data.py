#!/usr/bin/env python3
"""Prepare the frozen V12 corpus for leakage-safe portal-classification experiments.

This script DOES NOT modify the frozen corpus. It creates a deterministic experiment
manifest and metadata implementing the fixed design:

  * topics handled independently
  * adaptive year-month body-length bins: K=3 -> 2 -> 1
  * length cutoffs from portal-equal-weighted empirical quantiles
  * common cutoffs for all portals in a topic/month
  * every accepted (month, length-bin) must have >= T articles for EACH portal
  * unsupported months are excluded from the experimental dataset, not deleted
  * split AFTER binning, stratified by topic/month/length-bin/portal
  * 60/20/20 train/validation/test
  * all eligible articles are assigned to exactly one split
  * validation/test keep all assigned articles and receive deterministic balance weights
  * training keeps all assigned articles; a separate epoch sampler performs balanced cycling

Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

WORD_RE = re.compile(r"\b\w+(?:['’]\w+)?\b", re.UNICODE)
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_u64(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e}") from e
            rows.append(row)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            f.write("\n")


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def weighted_quantile_threshold(values_and_weights: Sequence[Tuple[int, float]], q: float) -> int:
    """Return an observed-value threshold where weighted empirical CDF first reaches q.

    Articles with value <= threshold belong to the lower bin. Equal word counts are
    never split across two bins. Portal totals are equal-weighted by the caller.
    """
    if not values_and_weights:
        raise ValueError("Cannot compute weighted quantile of an empty collection")
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1), got {q}")
    ordered = sorted(values_and_weights, key=lambda x: x[0])
    total = sum(w for _, w in ordered)
    target = q * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative + 1e-15 >= target:
            return int(value)
    return int(ordered[-1][0])


def assign_bin(body_words: int, cutoffs: Sequence[int]) -> int:
    # cutoff belongs to the lower bin: <= c0 -> 0, c0 < x <= c1 -> 1, ...
    return sum(body_words > c for c in cutoffs)


def propose_length_bins(
    month_rows: Sequence[Mapping[str, Any]],
    portals: Sequence[str],
    k: int,
) -> Tuple[List[int], Dict[int, Counter]]:
    if k == 1:
        counts = {0: Counter(r["portal"] for r in month_rows)}
        return [], counts

    portal_counts = Counter(r["portal"] for r in month_rows)
    if any(portal_counts[p] == 0 for p in portals):
        return [], {}

    # Equal total mass per portal. Scaling by 1/3 is unnecessary for quantiles.
    vw: List[Tuple[int, float]] = [
        (int(r["body_words"]), 1.0 / portal_counts[r["portal"]]) for r in month_rows
    ]
    cutoffs = [weighted_quantile_threshold(vw, j / k) for j in range(1, k)]

    # Tied lengths can collapse adjacent quantiles. Do not create degenerate bins.
    if len(set(cutoffs)) != len(cutoffs):
        return cutoffs, {}

    counts: Dict[int, Counter] = {b: Counter() for b in range(k)}
    for r in month_rows:
        b = assign_bin(int(r["body_words"]), cutoffs)
        counts[b][r["portal"]] += 1
    return cutoffs, counts


def choose_adaptive_bins(
    month_rows: Sequence[Mapping[str, Any]],
    portals: Sequence[str],
    kmax: int,
    threshold: int,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    for k in range(kmax, 0, -1):
        cutoffs, counts = propose_length_bins(month_rows, portals, k)
        valid_shape = bool(counts) and len(counts) == k
        mins: List[int] = []
        if valid_shape:
            mins = [min(counts[b].get(p, 0) for p in portals) for b in range(k)]
        passed = valid_shape and all(m >= threshold for m in mins)
        attempts.append({
            "k": k,
            "cutoffs_words_inclusive": cutoffs,
            "counts_by_bin_portal": {
                str(b): {p: counts[b].get(p, 0) for p in portals} for b in sorted(counts)
            },
            "minimum_portal_count_by_bin": mins,
            "passed": passed,
        })
        if passed:
            return {
                "eligible": True,
                "num_length_bins": k,
                "cutoffs_words_inclusive": cutoffs,
                "counts_by_bin_portal": {
                    str(b): {p: counts[b].get(p, 0) for p in portals} for b in range(k)
                },
                "minimum_portal_count_by_bin": mins,
                "attempts": attempts,
            }

    whole_counts = Counter(r["portal"] for r in month_rows)
    return {
        "eligible": False,
        "num_length_bins": 0,
        "cutoffs_words_inclusive": [],
        "counts_by_bin_portal": {},
        "minimum_portal_count_by_bin": [],
        "exclusion_reason": (
            f"month cannot satisfy minimum balanced support T={threshold}; "
            + ", ".join(f"{p}={whole_counts.get(p, 0)}" for p in portals)
        ),
        "attempts": attempts,
    }


def allocate_balanced_core_counts(
    m: int,
    fractions: Mapping[str, float],
    stratum_key: Tuple[str, str, int],
    base_seed: int,
    core_heldout_balance: MutableMapping[str, Dict[str, int]],
) -> Dict[str, int]:
    """Partition the stratum's balanced core m into 60/20/20 counts.

    The same core quota is guaranteed for every portal. Because at least one portal has
    exactly m articles, this preserves *all* pre-split balanced capacity across the three
    splits: sum_s min_p n[s,p] == m for every stratum.
    """
    train_frac = float(fractions["train"])
    val_frac = float(fractions["validation"])
    test_frac = float(fractions["test"])
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, abs_tol=1e-12):
        raise ValueError("split fractions must sum to 1")
    if not math.isclose(val_frac, test_frac, abs_tol=1e-12):
        raise ValueError("This implementation expects equal validation and test fractions")

    heldout = round_half_up(m * (val_frac + test_frac))
    heldout = max(2, min(m - 1, heldout))
    train_n = m - heldout
    val_n = heldout // 2
    test_n = heldout // 2
    if heldout % 2:
        topic, ym, b = stratum_key
        cur = core_heldout_balance[topic]
        if cur["validation"] < cur["test"]:
            val_n += 1
        elif cur["test"] < cur["validation"]:
            test_n += 1
        elif stable_u64(base_seed, "core-heldout-extra", topic, ym, b) % 2 == 0:
            val_n += 1
        else:
            test_n += 1
    if train_n < 1 or val_n < 1 or test_n < 1:
        raise AssertionError(f"Invalid balanced-core split for m={m}: {(train_n, val_n, test_n)}")
    topic, _ym, _b = stratum_key
    core_heldout_balance[topic]["validation"] += val_n
    core_heldout_balance[topic]["test"] += test_n
    return {"train": train_n, "validation": val_n, "test": test_n}


def largest_remainder_counts(
    n: int,
    fractions: Mapping[str, float],
    base_seed: int,
    *key_parts: object,
) -> Dict[str, int]:
    """Allocate n surplus observations approximately by the target fractions."""
    if n == 0:
        return {s: 0 for s in SPLITS}
    raw = {s: n * float(fractions[s]) for s in SPLITS}
    out = {s: int(math.floor(raw[s])) for s in SPLITS}
    remaining = n - sum(out.values())
    order = sorted(
        SPLITS,
        key=lambda s: (raw[s] - out[s], stable_u64(base_seed, "surplus-round", *key_parts, s)),
        reverse=True,
    )
    for s in order[:remaining]:
        out[s] += 1
    if sum(out.values()) != n:
        raise AssertionError("Largest-remainder allocation failed")
    return out


def spread_labels_over_sorted_lengths(
    n: int,
    quotas: Mapping[str, int],
    base_seed: int,
    key: Tuple[str, str, int, str],
) -> List[str]:
    """Create an exact-quota split schedule spread across the ordered length ranks.

    At each rank, assign the split that is currently most behind its ideal cumulative
    quota. This gives train/validation/test similar within-bin length coverage instead
    of assigning one contiguous length block to a split. Hashes only resolve ties.
    """
    assigned = {s: 0 for s in SPLITS}
    tie_priority = sorted(SPLITS, key=lambda s: stable_u64(base_seed, "split-tie", *key, s))
    tie_rank = {s: i for i, s in enumerate(tie_priority)}
    labels: List[str] = []

    for pos in range(n):
        candidates = [s for s in SPLITS if assigned[s] < quotas[s]]
        scored = []
        for s in candidates:
            ideal_cumulative = (pos + 1) * quotas[s] / n
            deficit = ideal_cumulative - assigned[s]
            scored.append((deficit, -tie_rank[s], s))
        _, _, chosen = max(scored)
        labels.append(chosen)
        assigned[chosen] += 1

    if assigned != dict(quotas):
        raise AssertionError(f"Split schedule quota mismatch: assigned={assigned}, quotas={quotas}")
    return labels


def portal_topic_table(rows: Sequence[Mapping[str, Any]], portals: Sequence[str], topics: Sequence[str]) -> Dict[str, Any]:
    c = Counter((r["portal"], r["topic"]) for r in rows)
    return {
        p: {t: c[(p, t)] for t in topics} for p in portals
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiment_config.json")
    ap.add_argument("--force", action="store_true", help="Replace an existing output directory")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (project_root / config["input_corpus"]).resolve()
    out_dir = (project_root / config["output_dir"]).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Frozen corpus not found: {input_path}")
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                "Refusing to overwrite a frozen experiment assignment. Use --force only intentionally."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    actual_hash = sha256_file(input_path)
    expected_hash = config["expected_corpus_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen corpus SHA-256 mismatch. Refusing to create splits.\n"
            f"expected: {expected_hash}\nactual:   {actual_hash}\nfile:     {input_path}"
        )

    portals = list(config["expected_portals"])
    topics = list(config["expected_topics"])
    kmax = int(config["max_length_bins"])
    threshold = int(config["min_balanced_cell_support"])
    fractions = config["split_fractions"]
    base_seed = int(config["base_seed"])

    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError("Frozen corpus is empty")

    required = {"article_id", "portal", "topic", "year_month", "model_text", "duplicate_split_group", "text_clean_sha1"}
    for i, r in enumerate(rows):
        missing = required - set(r)
        if missing:
            raise RuntimeError(f"Row {i} missing required fields: {sorted(missing)}")
        r["body_words"] = len(WORD_RE.findall(r["model_text"]))
        r["body_chars"] = len(r["model_text"])

    ids = [r["article_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dup_ids = [k for k, v in Counter(ids).items() if v > 1]
        raise RuntimeError(f"article_id is not unique; examples: {dup_ids[:10]}")

    observed_portals = sorted({r["portal"] for r in rows})
    observed_topics = sorted({r["topic"] for r in rows})
    if sorted(portals) != observed_portals:
        raise RuntimeError(f"Portal mismatch: expected {sorted(portals)}, observed {observed_portals}")
    if sorted(topics) != observed_topics:
        raise RuntimeError(f"Topic mismatch: expected {sorted(topics)}, observed {observed_topics}")

    # Final V12 should have only singleton duplicate groups. Fail closed if that changes.
    dg = Counter(r["duplicate_split_group"] for r in rows)
    non_single = {g: n for g, n in dg.items() if n > 1}
    if non_single:
        examples = list(sorted(non_single.items()))[:10]
        raise RuntimeError(
            "Non-singleton duplicate_split_group values found. This splitter intentionally "
            f"fails closed rather than risk cross-split duplicate leakage. Examples: {examples}"
        )

    by_topic_month: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_topic_month[(r["topic"], r["year_month"])].append(r)

    bin_spec: Dict[str, Dict[str, Any]] = {t: {} for t in topics}
    eligible_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []

    for topic in topics:
        months = sorted(ym for (t, ym) in by_topic_month if t == topic)
        for ym in months:
            month_rows = by_topic_month[(topic, ym)]
            decision = choose_adaptive_bins(month_rows, portals, kmax, threshold)
            bin_spec[topic][ym] = decision
            if not decision["eligible"]:
                for r in month_rows:
                    r["experiment_eligible"] = False
                    r["exclusion_reason"] = decision["exclusion_reason"]
                    r["length_bin"] = None
                    excluded_rows.append(r)
                continue

            cutoffs = decision["cutoffs_words_inclusive"]
            for r in month_rows:
                r["experiment_eligible"] = True
                r["exclusion_reason"] = None
                r["length_bin"] = assign_bin(r["body_words"], cutoffs)
                eligible_rows.append(r)

    # Deterministic split within topic/month/length_bin/portal.
    cells: Dict[Tuple[str, str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in eligible_rows:
        cells[(r["topic"], r["year_month"], int(r["length_bin"]), r["portal"])].append(r)

    core_heldout_balance: MutableMapping[str, Dict[str, int]] = defaultdict(
        lambda: {"validation": 0, "test": 0}
    )
    split_allocations: Dict[str, Dict[str, int]] = {}

    # Allocate jointly by stratum first so the balanced minimum m is partitioned once and
    # guaranteed in every portal. Then distribute each portal's surplus approximately 60/20/20.
    strata = sorted({(t, ym, b) for (t, ym, b, _p) in cells})
    for stratum_key in strata:
        topic, ym, b = stratum_key
        counts = {p: len(cells[(topic, ym, b, p)]) for p in portals}
        m = min(counts.values())
        core = allocate_balanced_core_counts(m, fractions, stratum_key, base_seed, core_heldout_balance)

        for portal in portals:
            key = (topic, ym, b, portal)
            cell = cells[key]
            surplus_n = len(cell) - m
            surplus = largest_remainder_counts(surplus_n, fractions, base_seed, *key)
            quotas = {s: core[s] + surplus[s] for s in SPLITS}
            if sum(quotas.values()) != len(cell):
                raise AssertionError(f"Split quotas do not sum to cell size for {key}")
            if any(quotas[s] < core[s] for s in SPLITS):
                raise AssertionError(f"Portal lost balanced-core coverage for {key}")
            split_allocations["|".join(map(str, key))] = quotas

            # Primary ordering is body length; hash breaks exact-length ties without label-order artifacts.
            ordered = sorted(
                cell,
                key=lambda r: (r["body_words"], stable_u64(base_seed, "article-tie", *key, r["article_id"])),
            )
            labels = spread_labels_over_sorted_lengths(len(ordered), quotas, base_seed, key)
            for r, split in zip(ordered, labels):
                r["split"] = split

    for r in excluded_rows:
        r["split"] = "excluded"

    # Split-specific balance weights and balanced capacity.
    split_stratum_counts: Dict[Tuple[str, str, str, int], Counter] = defaultdict(Counter)
    for r in eligible_rows:
        split_stratum_counts[(r["split"], r["topic"], r["year_month"], int(r["length_bin"]))][r["portal"]] += 1

    split_stratum_min: Dict[Tuple[str, str, str, int], int] = {}
    for skey, counts in split_stratum_counts.items():
        m = min(counts.get(p, 0) for p in portals)
        if m < 1:
            raise RuntimeError(f"A split stratum lost portal coverage: {skey} counts={dict(counts)}")
        split_stratum_min[skey] = m

    for r in eligible_rows:
        skey = (r["split"], r["topic"], r["year_month"], int(r["length_bin"]))
        n_portal = split_stratum_counts[skey][r["portal"]]
        m = split_stratum_min[skey]
        r["stratum_portal_count_in_split"] = n_portal
        r["stratum_min_portal_count_in_split"] = m
        r["balance_weight"] = m / n_portal

    for r in excluded_rows:
        r["stratum_portal_count_in_split"] = None
        r["stratum_min_portal_count_in_split"] = None
        r["balance_weight"] = None

    # Master manifest: all frozen articles, including explicitly excluded rows.
    manifest_fields = [
        "article_id", "text_clean_sha1", "duplicate_split_group", "portal", "topic", "year_month",
        "body_words", "body_chars", "experiment_eligible", "exclusion_reason", "length_bin", "split",
        "stratum_portal_count_in_split", "stratum_min_portal_count_in_split", "balance_weight",
    ]
    manifest_rows = [
        {k: r.get(k) for k in manifest_fields}
        for r in sorted(rows, key=lambda x: (x["topic"], x["year_month"], x["portal"], x["article_id"]))
    ]
    manifest_path = out_dir / "experiment_manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)

    # Convenience heldout manifests. Every assigned heldout article appears exactly once.
    for split in ("validation", "test"):
        out_csv = out_dir / f"{split}_weights.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["article_id", "portal", "topic", "year_month", "length_bin", "body_words", "balance_weight"])
            for r in sorted((x for x in eligible_rows if x["split"] == split), key=lambda x: x["article_id"]):
                w.writerow([
                    r["article_id"], r["portal"], r["topic"], r["year_month"], r["length_bin"],
                    r["body_words"], f'{r["balance_weight"]:.12g}',
                ])

    # Exclusion ledger.
    with (out_dir / "excluded_articles.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "portal", "topic", "year_month", "body_words", "reason"])
        for r in sorted(excluded_rows, key=lambda x: (x["topic"], x["year_month"], x["portal"], x["article_id"])):
            w.writerow([r["article_id"], r["portal"], r["topic"], r["year_month"], r["body_words"], r["exclusion_reason"]])

    # Train buckets keep every training article. The sampler takes per_epoch_per_portal from each portal list.
    train_buckets: List[Dict[str, Any]] = []
    train_by_stratum: Dict[Tuple[str, str, int], Dict[str, List[str]]] = defaultdict(lambda: {p: [] for p in portals})
    for r in eligible_rows:
        if r["split"] == "train":
            train_by_stratum[(r["topic"], r["year_month"], int(r["length_bin"]))][r["portal"]].append(r["article_id"])
    for (topic, ym, b), by_portal in sorted(train_by_stratum.items()):
        for p in portals:
            by_portal[p].sort()
        m = min(len(by_portal[p]) for p in portals)
        train_buckets.append({
            "topic": topic,
            "year_month": ym,
            "length_bin": b,
            "per_epoch_per_portal": m,
            "article_ids_by_portal": by_portal,
            "counts_by_portal": {p: len(by_portal[p]) for p in portals},
        })
    write_json(out_dir / "train_buckets.json", {"base_seed": base_seed, "portals": portals, "buckets": train_buckets})

    # Length-bin specification includes all attempts, which makes the 3->2->1 fallback auditable.
    length_bins_obj = {
        "method": "portal-equal-weighted empirical body-word quantiles, common cutoffs per topic/year_month",
        "max_length_bins": kmax,
        "min_balanced_cell_support": threshold,
        "bin_rule": "bin 0: words <= cutoff[0]; middle bins: previous cutoff < words <= current cutoff; final: words > last cutoff",
        "topics": bin_spec,
    }
    write_json(out_dir / "length_bins.json", length_bins_obj)

    # Detailed split/capacity summary.
    split_counts = Counter((r["split"], r["topic"], r["portal"]) for r in rows)
    split_topic_totals = Counter((r["split"], r["topic"]) for r in rows)
    capacity = defaultdict(lambda: defaultdict(int))
    for (split, topic, ym, b), m in split_stratum_min.items():
        capacity[split][topic] += 3 * m

    bin_count_distribution: Dict[str, Counter] = {t: Counter() for t in topics}
    excluded_months: Dict[str, List[str]] = {t: [] for t in topics}
    for t in topics:
        for ym, spec in bin_spec[t].items():
            if spec["eligible"]:
                bin_count_distribution[t][spec["num_length_bins"]] += 1
            else:
                excluded_months[t].append(ym)

    summary = {
        "frozen_corpus_articles": len(rows),
        "frozen_corpus_counts_by_portal_topic": portal_topic_table(rows, portals, topics),
        "experiment_eligible_articles": len(eligible_rows),
        "excluded_articles": len(excluded_rows),
        "eligible_counts_by_portal_topic": portal_topic_table(eligible_rows, portals, topics),
        "excluded_months_by_topic": excluded_months,
        "accepted_months_by_number_of_length_bins": {
            t: {str(k): bin_count_distribution[t][k] for k in sorted(bin_count_distribution[t])} for t in topics
        },
        "split_counts_by_topic_portal": {
            split: {
                t: {p: split_counts[(split, t, p)] for p in portals} for t in topics
            } for split in SPLITS
        },
        "split_totals_by_topic": {
            split: {t: split_topic_totals[(split, t)] for t in topics} for split in SPLITS
        },
        "balanced_capacity_articles_by_split_topic": {
            split: {t: capacity[split][t] for t in topics} for split in SPLITS
        },
        "balanced_capacity_per_portal_by_split_topic": {
            split: {t: capacity[split][t] // 3 for t in topics} for split in SPLITS
        },
        "pre_split_balanced_capacity_by_topic": {
            t: sum(
                3 * min(spec["counts_by_bin_portal"][str(b)][p] for p in portals)
                for ym, spec in bin_spec[t].items() if spec["eligible"]
                for b in range(spec["num_length_bins"])
            ) for t in topics
        },
    }
    write_json(out_dir / "split_summary.json", summary)

    # Human-readable CSV of every accepted/rejected month/bin.
    with (out_dir / "stratum_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "topic", "year_month", "eligible", "num_length_bins", "length_bin",
            "lower_exclusive_words", "upper_inclusive_words",
            *[f"all_{p}" for p in portals], "all_min",
            *[f"train_{p}" for p in portals], "train_min",
            *[f"validation_{p}" for p in portals], "validation_min",
            *[f"test_{p}" for p in portals], "test_min",
        ])
        for t in topics:
            for ym in sorted(bin_spec[t]):
                spec = bin_spec[t][ym]
                if not spec["eligible"]:
                    w.writerow([t, ym, False, 0, "", "", "", *([""] * (len(portals) + 1) * 4)])
                    continue
                cuts = spec["cutoffs_words_inclusive"]
                k = spec["num_length_bins"]
                for b in range(k):
                    lower = "" if b == 0 else cuts[b - 1]
                    upper = "" if b == k - 1 else cuts[b]
                    allc = spec["counts_by_bin_portal"][str(b)]
                    row = [t, ym, True, k, b, lower, upper]
                    row += [allc[p] for p in portals] + [min(allc[p] for p in portals)]
                    for split in SPLITS:
                        sc = split_stratum_counts[(split, t, ym, b)]
                        row += [sc[p] for p in portals] + [min(sc[p] for p in portals)]
                    w.writerow(row)

    # Audits. These are deliberately redundant: fail loudly before anything is trained.
    checks: Dict[str, Any] = {}
    checks["input_sha256_matches"] = actual_hash == expected_hash
    checks["manifest_has_every_frozen_article_once"] = len(manifest_rows) == len(rows) == len({r["article_id"] for r in manifest_rows})
    checks["eligible_plus_excluded_equals_frozen"] = len(eligible_rows) + len(excluded_rows) == len(rows)
    checks["all_eligible_assigned_exactly_one_split"] = all(r.get("split") in SPLITS for r in eligible_rows)
    checks["all_excluded_have_excluded_split"] = all(r.get("split") == "excluded" for r in excluded_rows)
    checks["no_non_singleton_duplicate_groups"] = not non_single
    checks["all_accepted_pre_split_cells_meet_threshold"] = all(
        min(spec["counts_by_bin_portal"][str(b)][p] for p in portals) >= threshold
        for t in topics for spec in bin_spec[t].values() if spec["eligible"]
        for b in range(spec["num_length_bins"])
    )
    checks["all_split_strata_contain_all_portals"] = all(
        all(counts.get(p, 0) >= 1 for p in portals) for counts in split_stratum_counts.values()
    )
    # Splitting must not destroy any pre-split balanced capacity.
    checks["split_preserves_all_pre_split_balanced_capacity"] = True
    for t in topics:
        pre = sum(
            min(spec["counts_by_bin_portal"][str(b)][p] for p in portals)
            for _ym, spec in bin_spec[t].items() if spec["eligible"]
            for b in range(spec["num_length_bins"])
        )
        post = sum(
            split_stratum_min[(split, t, ym, b)]
            for split in SPLITS
            for ym, spec in bin_spec[t].items() if spec["eligible"]
            for b in range(spec["num_length_bins"])
        )
        if pre != post:
            checks["split_preserves_all_pre_split_balanced_capacity"] = False
    checks["all_balance_weights_positive_and_at_most_one"] = all(
        0.0 < float(r["balance_weight"]) <= 1.0 for r in eligible_rows
    )
    checks["balance_weights_equalize_portal_mass_within_each_split_stratum"] = True
    weight_mass_details: Dict[str, Dict[str, float]] = {}
    for skey, counts in split_stratum_counts.items():
        split, topic, ym, b = skey
        mass = {p: 0.0 for p in portals}
        for r in eligible_rows:
            if (r["split"], r["topic"], r["year_month"], int(r["length_bin"])) == skey:
                mass[r["portal"]] += float(r["balance_weight"])
        target = split_stratum_min[skey]
        ok = all(math.isclose(mass[p], target, rel_tol=1e-12, abs_tol=1e-9) for p in portals)
        if not ok:
            checks["balance_weights_equalize_portal_mass_within_each_split_stratum"] = False
        weight_mass_details["|".join(map(str, skey))] = mass

    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Experiment-data audit failed: {failed}")

    audit = {
        "all_checks_passed": True,
        "checks": checks,
        "input_corpus_sha256": actual_hash,
        "n_frozen": len(rows),
        "n_eligible": len(eligible_rows),
        "n_excluded": len(excluded_rows),
        "weight_mass_by_split_stratum": weight_mass_details,
    }
    write_json(out_dir / "audit_report.json", audit)

    resolved = dict(config)
    resolved["input_corpus_absolute_at_generation"] = str(input_path)
    resolved["output_dir_absolute_at_generation"] = str(out_dir)
    resolved["actual_corpus_sha256"] = actual_hash
    resolved["body_words_regex"] = WORD_RE.pattern
    resolved["split_assignment_method"] = (
        "For each topic/year_month/length_bin, partition the minimum portal count as a shared balanced core across 60/20/20, then allocate each portal's surplus by largest remainder; "
        "articles sorted by body_words with seeded hash tie-break; split labels interleaved over "
        "length rank to preserve within-bin length distributions."
    )
    resolved["heldout_evaluation_method"] = (
        "Use every validation/test article once per evaluation. Weight each article by "
        "m_s/n_{s,p}, where m_s is the minimum portal count in that split's topic/year_month/length_bin stratum."
    )
    write_json(out_dir / "experiment_config_resolved.json", resolved)

    # Hash generated core artifacts after they exist.
    core = [
        "experiment_manifest.jsonl", "length_bins.json", "train_buckets.json",
        "validation_weights.csv", "test_weights.csv", "excluded_articles.csv",
        "stratum_summary.csv", "split_summary.json", "audit_report.json", "experiment_config_resolved.json",
    ]
    hashes = {name: sha256_file(out_dir / name) for name in core}
    write_json(out_dir / "ARTIFACT_HASHES.json", hashes)

    print("=== EXPERIMENT DATA PREPARATION COMPLETE ===")
    print(f"Frozen corpus:       {len(rows):,}")
    print(f"Experiment eligible: {len(eligible_rows):,}")
    print(f"Excluded:            {len(excluded_rows):,}")
    print()
    print("Eligible counts by portal/topic:")
    eligible_table = summary["eligible_counts_by_portal_topic"]
    for p in portals:
        print(f"  {p:10s} " + "  ".join(f"{t}={eligible_table[p][t]:4d}" for t in topics))
    print()
    print("Excluded months:")
    for t in topics:
        print(f"  {t}: {', '.join(excluded_months[t]) if excluded_months[t] else '(none)'}")
    print()
    print("Adaptive bin counts (months):")
    for t in topics:
        print(f"  {t}: {summary['accepted_months_by_number_of_length_bins'][t]}")
    print()
    print("Actual split totals by topic:")
    for split in SPLITS:
        print(f"  {split:10s} " + "  ".join(f"{t}={summary['split_totals_by_topic'][split][t]:4d}" for t in topics))
    print()
    print("Balanced capacity per portal by split/topic:")
    for split in SPLITS:
        print(f"  {split:10s} " + "  ".join(f"{t}={summary['balanced_capacity_per_portal_by_split_topic'][split][t]:3d}" for t in topics))
    print()
    print(f"Output directory: {out_dir}")
    print("All audits passed. Do not alter the generated split assignment after model training begins.")


if __name__ == "__main__":
    main()
