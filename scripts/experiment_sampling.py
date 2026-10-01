#!/usr/bin/env python3
"""Deterministic balanced training-epoch sampler for experiment_data/train_buckets.json.

For every topic/year_month/length_bin stratum, each epoch selects exactly the minimum
training portal count from EACH portal. Surplus samples are cycled through a fixed,
seeded circular permutation, so within a portal/stratum exposure is as even as possible
and no article repeats within one epoch. The complete selected epoch is then globally
shuffled with an epoch-specific deterministic seed before batching.

Can be imported or used as a CLI to materialize an epoch ID list.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def stable_u64(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seeded_permutation(items: Sequence[str], seed: int) -> List[str]:
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def circular_take(permutation: Sequence[str], start: int, count: int) -> List[str]:
    n = len(permutation)
    if count > n:
        raise ValueError(f"Cannot take {count} unique samples from a portal bucket of size {n}")
    return [permutation[(start + i) % n] for i in range(count)]


class BalancedEpochSampler:
    def __init__(
        self,
        train_buckets_path: str | Path,
        *,
        sampler_seed: int | None = None,
        global_shuffle_seed: int | None = None,
    ):
        self.path = Path(train_buckets_path)
        obj = json.loads(self.path.read_text(encoding="utf-8"))

        self.base_seed = int(obj["base_seed"])

        # Backward compatibility:
        # if no overrides are supplied, behavior is exactly the old behavior.
        self.sampler_seed = (
            self.base_seed if sampler_seed is None else int(sampler_seed)
        )
        self.global_shuffle_seed = (
            self.base_seed
            if global_shuffle_seed is None
            else int(global_shuffle_seed)
        )

        self.portals = list(obj["portals"])
        self.buckets = list(obj["buckets"])

        self._permutations: Dict[tuple, List[str]] = {}

        for b in self.buckets:
            bucket_key = (
                b["topic"],
                b["year_month"],
                int(b["length_bin"]),
            )
            m = int(b["per_epoch_per_portal"])

            for portal in self.portals:
                ids = list(b["article_ids_by_portal"][portal])

                if m < 1 or m > len(ids):
                    raise ValueError(
                        f"Invalid bucket quota {m} for "
                        f"{bucket_key}/{portal} with n={len(ids)}"
                    )

                seed = stable_u64(
                    self.sampler_seed,
                    "portal-permutation",
                    *bucket_key,
                    portal,
                )

                self._permutations[
                    (*bucket_key, portal)
                ] = seeded_permutation(ids, seed)

    def article_ids_for_epoch(
        self,
        epoch: int,
        topic: str | None = None,
    ) -> List[str]:

        if epoch < 0:
            raise ValueError("epoch must be >= 0")

        selected: List[str] = []

        for b in self.buckets:
            if topic is not None and b["topic"] != topic:
                continue

            bucket_key = (
                b["topic"],
                b["year_month"],
                int(b["length_bin"]),
            )

            m = int(b["per_epoch_per_portal"])

            for portal in self.portals:
                perm = self._permutations[
                    (*bucket_key, portal)
                ]

                # Same fair circular cycling as before.
                start = (epoch * m) % len(perm)

                chosen = circular_take(
                    perm,
                    start,
                    m,
                )

                if len(chosen) != len(set(chosen)):
                    raise AssertionError(
                        "Sampler repeated an article within "
                        "a portal/bucket/epoch"
                    )

                selected.extend(chosen)

        # Same global shuffle as before, except the seed can
        # now optionally be overridden.
        global_seed = stable_u64(
            self.global_shuffle_seed,
            "global-epoch-shuffle",
            topic or "ALL",
            epoch,
        )

        random.Random(global_seed).shuffle(selected)

        if len(selected) != len(set(selected)):
            raise AssertionError(
                "An article was selected more than once "
                "in the same epoch"
            )

        return selected

    def expected_epoch_size(
        self,
        topic: str | None = None,
    ) -> int:
        return sum(
            3 * int(b["per_epoch_per_portal"])
            for b in self.buckets
            if topic is None or b["topic"] == topic
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-buckets", default="experiment_data/train_buckets.json")
    ap.add_argument("--epoch", type=int, required=True, help="Zero-based epoch number")
    ap.add_argument("--topic", choices=["topicA", "topicB"], default=None)
    ap.add_argument("--output", required=True, help="CSV path for the selected, globally shuffled article IDs")
    args = ap.parse_args()

    sampler = BalancedEpochSampler(args.train_buckets)
    ids = sampler.article_ids_for_epoch(args.epoch, args.topic)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "article_id"])
        for i, article_id in enumerate(ids):
            w.writerow([i, article_id])
    print(f"Wrote {len(ids):,} article IDs for epoch {args.epoch} topic={args.topic or 'ALL'} to {out}")


if __name__ == "__main__":
    main()
