#!/usr/bin/env python3
"""Run post-split diagnostics for the frozen experiment-data design.

Purpose
-------
This script answers two development-stage questions *without touching the test set*:

1. Did the month x body-length control actually reduce portal predictability from
   body word count on the fixed validation split?
2. Can the generated train/validation manifests drive a simple text classifier
   end-to-end (TF-IDF smoke test)?

The test split is deliberately NOT used for any model-based diagnostic. It remains
sealed for final evaluation after model selection.

Required third-party packages: numpy, scikit-learn.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    from sklearn.svm import LinearSVC
except Exception as exc:  # pragma: no cover - user-facing environment check
    raise SystemExit(
        "Experiment diagnostics require numpy and scikit-learn in the active environment.\n"
        f"Import error: {exc}"
    ) from exc

WORD_RE = re.compile(r"\b\w+(?:['’]\w+)?\b", re.UNICODE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def weighted_standardize_fit_transform(
    x_train: np.ndarray,
    w_train: np.ndarray,
    x_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    mu = np.average(x_train, axis=0, weights=w_train)
    var = np.average((x_train - mu) ** 2, axis=0, weights=w_train)
    sd = np.sqrt(np.maximum(var, 1e-12))
    return (
        (x_train - mu) / sd,
        (x_other - mu) / sd,
        {"weighted_train_mean": mu.tolist(), "weighted_train_std": sd.tolist()},
    )


def metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    weights: Sequence[float],
    labels: Sequence[str],
) -> Dict[str, Any]:
    w = np.asarray(weights, dtype=float)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        yt, yp, labels=list(labels), average=None, sample_weight=w, zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=list(labels), sample_weight=w)
    return {
        "accuracy": float(accuracy_score(yt, yp, sample_weight=w)),
        "macro_f1": float(f1_score(yt, yp, labels=list(labels), average="macro", sample_weight=w, zero_division=0)),
        "total_weight": float(w.sum()),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support_weight": float(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {
            labels[i]: {labels[j]: float(cm[i, j]) for j in range(len(labels))}
            for i in range(len(labels))
        },
    }


def body_word_feature(r: Mapping[str, Any], _text: str) -> List[float]:
    return [math.log1p(int(r["body_words"]))]


def body_words_chars_features(r: Mapping[str, Any], _text: str) -> List[float]:
    return [math.log1p(int(r["body_words"])), math.log1p(int(r["body_chars"]))]


def structural_features(r: Mapping[str, Any], text: str) -> List[float]:
    n = max(1, len(text))
    return [
        math.log1p(int(r["body_words"])),
        math.log1p(int(r["body_chars"])),
        math.log1p(text.count(".")),
        math.log1p(text.count(",")),
        math.log1p(text.count("?") + text.count("!")),
        math.log1p(text.count('"') + text.count("“") + text.count("”")),
        sum(ch.isdigit() for ch in text) / n,
        sum(not ch.isascii() for ch in text) / n,
    ]


def build_month_only_weights(
    manifest: Sequence[Mapping[str, Any]], portals: Sequence[str]
) -> Dict[Tuple[str, str, str, str], float]:
    counts: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    for r in manifest:
        if r.get("experiment_eligible") and r["split"] in {"train", "validation"}:
            counts[(r["split"], r["topic"], r["year_month"])][r["portal"]] += 1

    out: Dict[Tuple[str, str, str, str], float] = {}
    for key, c in counts.items():
        if any(c.get(p, 0) == 0 for p in portals):
            raise RuntimeError(f"Month-only diagnostic lost portal coverage for {key}: {dict(c)}")
        m = min(c[p] for p in portals)
        for p in portals:
            out[(*key, p)] = m / c[p]
    return out


def weight_for(
    r: Mapping[str, Any],
    mode: str,
    month_weights: Mapping[Tuple[str, str, str, str], float],
) -> float:
    if mode == "month_only":
        return float(month_weights[(r["split"], r["topic"], r["year_month"], r["portal"])])
    if mode == "month_plus_length":
        return float(r["balance_weight"])
    raise ValueError(mode)


def fit_numeric_diagnostic(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    text_by_id: Mapping[str, str],
    portals: Sequence[str],
    feature_fn,
    feature_names: Sequence[str],
    weight_mode: str,
    month_weights: Mapping[Tuple[str, str, str, str], float],
    seed: int,
) -> Dict[str, Any]:
    x_train = np.asarray(
        [feature_fn(r, text_by_id[r["article_id"]]) for r in train_rows], dtype=float
    )
    x_val = np.asarray(
        [feature_fn(r, text_by_id[r["article_id"]]) for r in val_rows], dtype=float
    )
    y_train = [r["portal"] for r in train_rows]
    y_val = [r["portal"] for r in val_rows]
    w_train = np.asarray([weight_for(r, weight_mode, month_weights) for r in train_rows], dtype=float)
    w_val = np.asarray([weight_for(r, weight_mode, month_weights) for r in val_rows], dtype=float)

    x_train_z, x_val_z, scaling = weighted_standardize_fit_transform(x_train, w_train, x_val)
    clf = LogisticRegression(max_iter=3000, random_state=seed)
    clf.fit(x_train_z, y_train, sample_weight=w_train)
    pred = clf.predict(x_val_z)

    return {
        "feature_names": list(feature_names),
        "weighting": weight_mode,
        "n_train": len(train_rows),
        "n_validation": len(val_rows),
        "scaling": scaling,
        "controlled_validation_metrics": metrics(y_val, pred, w_val, portals),
        "same_predictions_unweighted_validation_metrics": metrics(y_val, pred, [1.0] * len(y_val), portals),
        "coefficients": {
            str(label): {
                feature_names[j]: float(clf.coef_[i, j])
                for j in range(len(feature_names))
            }
            for i, label in enumerate(clf.classes_)
        },
        "intercepts": {str(label): float(clf.intercept_[i]) for i, label in enumerate(clf.classes_)},
    }


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _v, w in pairs)
    if total <= 0:
        return float("nan")
    target = q * total
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= target:
            return float(v)
    return float(pairs[-1][0])


def length_distribution_rows(
    val_rows: Sequence[Mapping[str, Any]],
    portals: Sequence[str],
    month_weights: Mapping[Tuple[str, str, str, str], float],
) -> List[Dict[str, Any]]:
    out = []
    for topic in sorted({r["topic"] for r in val_rows}):
        for mode in ("month_only", "month_plus_length"):
            for portal in portals:
                rr = [r for r in val_rows if r["topic"] == topic and r["portal"] == portal]
                vals = [float(r["body_words"]) for r in rr]
                w = [weight_for(r, mode, month_weights) for r in rr]
                mean = float(np.average(np.asarray(vals), weights=np.asarray(w)))
                out.append({
                    "topic": topic,
                    "weighting": mode,
                    "portal": portal,
                    "n_articles": len(rr),
                    "effective_weight": float(sum(w)),
                    "weighted_mean_body_words": mean,
                    "weighted_q25_body_words": weighted_quantile(vals, w, 0.25),
                    "weighted_median_body_words": weighted_quantile(vals, w, 0.50),
                    "weighted_q75_body_words": weighted_quantile(vals, w, 0.75),
                })
    return out


def tfidf_smoke_test(
    topic: str,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    text_by_id: Mapping[str, str],
    portals: Sequence[str],
    month_weights: Mapping[Tuple[str, str, str, str], float],
    seed: int,
    out_dir: Path,
    max_features: int,
    min_df: int,
    top_n: int,
) -> Dict[str, Any]:
    train_text = [text_by_id[r["article_id"]] for r in train_rows]
    val_text = [text_by_id[r["article_id"]] for r in val_rows]
    y_train = [r["portal"] for r in train_rows]
    y_val = [r["portal"] for r in val_rows]
    w_train = np.asarray([float(r["balance_weight"]) for r in train_rows], dtype=float)
    w_val = np.asarray([float(r["balance_weight"]) for r in val_rows], dtype=float)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.97,
        max_features=max_features,
        sublinear_tf=True,
        lowercase=True,
        strip_accents=None,
    )
    x_train = vec.fit_transform(train_text)
    x_val = vec.transform(val_text)
    clf = LinearSVC(C=1.0, random_state=seed)
    clf.fit(x_train, y_train, sample_weight=w_train)
    pred = clf.predict(x_val)

    terms = np.asarray(vec.get_feature_names_out())
    top_rows = []
    for i, label in enumerate(clf.classes_):
        idx = np.argsort(clf.coef_[i])[-top_n:][::-1]
        for rank, j in enumerate(idx, 1):
            top_rows.append({
                "topic": topic,
                "portal": str(label),
                "rank": rank,
                "feature": str(terms[j]),
                "coefficient": float(clf.coef_[i, j]),
            })
    write_csv(out_dir / f"tfidf_top_features_{topic}.csv", top_rows)

    return {
        "purpose": "pipeline smoke test only; not the final reported TF-IDF experiment",
        "training_weighting": "month_plus_length",
        "validation_weighting": "month_plus_length",
        "n_train": len(train_rows),
        "n_validation": len(val_rows),
        "vocab_size": int(len(terms)),
        "settings": {
            "ngram_range": [1, 2],
            "min_df": min_df,
            "max_df": 0.97,
            "max_features": max_features,
            "sublinear_tf": True,
            "classifier": "LinearSVC(C=1.0)",
        },
        "controlled_validation_metrics": metrics(y_val, pred, w_val, portals),
        "same_predictions_unweighted_validation_metrics": metrics(y_val, pred, [1.0] * len(y_val), portals),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiment_config.json")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    exp_dir = (root / config["output_dir"]).resolve()
    manifest_path = exp_dir / "experiment_manifest.jsonl"
    corpus_path = (root / config["input_corpus"]).resolve()
    out_dir = exp_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}; run preparation first")
    if sha256_file(corpus_path) != config["expected_corpus_sha256"]:
        raise RuntimeError("Frozen corpus hash changed before diagnostics; refusing to continue")

    diag_cfg = config.get("diagnostics", {})
    seed = int(diag_cfg.get("seed", config.get("base_seed", 20260901)))
    max_features = int(diag_cfg.get("tfidf_max_features", 45000))
    min_df = int(diag_cfg.get("tfidf_min_df", 3))
    top_n = int(diag_cfg.get("tfidf_top_features_per_portal", 30))
    near_chance_tolerance = float(diag_cfg.get("length_word_accuracy_near_chance_tolerance", 0.05))

    portals = list(config["expected_portals"])
    topics = list(config["expected_topics"])
    chance = 1.0 / len(portals)

    manifest_all = read_jsonl(manifest_path)
    eligible = [r for r in manifest_all if r.get("experiment_eligible")]
    train_val = [r for r in eligible if r["split"] in {"train", "validation"}]
    # Intentionally do not perform any model diagnostic on test.
    if not any(r["split"] == "test" for r in eligible):
        raise RuntimeError("Manifest contains no test rows")

    corpus_rows = read_jsonl(corpus_path)
    needed_ids = {r["article_id"] for r in train_val}
    text_by_id: Dict[str, str] = {}
    for r in corpus_rows:
        aid = r["article_id"]
        if aid in needed_ids:
            text = r.get("model_text") or ""
            text_by_id[aid] = text
    missing_text = sorted(needed_ids - set(text_by_id))
    if missing_text:
        raise RuntimeError(f"Missing model_text for {len(missing_text)} train/validation rows; examples={missing_text[:5]}")

    # Verify manifest length metadata against the frozen body strings used by the real models.
    length_mismatches = []
    by_id_manifest = {r["article_id"]: r for r in train_val}
    for aid, text in text_by_id.items():
        m = by_id_manifest[aid]
        words = len(WORD_RE.findall(text))
        chars = len(text)
        if words != int(m["body_words"]) or chars != int(m["body_chars"]):
            length_mismatches.append((aid, words, chars, m["body_words"], m["body_chars"]))
    if length_mismatches:
        raise RuntimeError(f"Manifest/corpus length mismatch; examples={length_mismatches[:5]}")

    month_weights = build_month_only_weights(eligible, portals)
    summary: Dict[str, Any] = {
        "chance_accuracy_for_balanced_3_class_problem": chance,
        "test_set_policy": (
            "SEALED: no classifier is fit/evaluated on test here. Test is only mechanically "
            "assigned/audited by prepare_experiment_data.py and remains untouched until final evaluation."
        ),
        "primary_length_control_variable": "body_words",
        "primary_length_control_interpretation": (
            "The bins are defined from body word count, so the primary success diagnostic is a "
            "word-count-only classifier. Character count conditional on word count partly reflects "
            "average word length / lexical style and is therefore reported only as a secondary proxy."
        ),
        "topics": {},
    }

    dist_rows = length_distribution_rows(
        [r for r in eligible if r["split"] == "validation"], portals, month_weights
    )
    write_csv(out_dir / "validation_length_distribution.csv", dist_rows)

    for topic in topics:
        train_rows = [r for r in eligible if r["split"] == "train" and r["topic"] == topic]
        val_rows = [r for r in eligible if r["split"] == "validation" and r["topic"] == topic]
        if not train_rows or not val_rows:
            raise RuntimeError(f"Missing train/validation rows for {topic}")

        word_before = fit_numeric_diagnostic(
            train_rows, val_rows, text_by_id, portals,
            body_word_feature, ["log1p_body_words"], "month_only", month_weights, seed,
        )
        word_after = fit_numeric_diagnostic(
            train_rows, val_rows, text_by_id, portals,
            body_word_feature, ["log1p_body_words"], "month_plus_length", month_weights, seed,
        )
        size_before = fit_numeric_diagnostic(
            train_rows, val_rows, text_by_id, portals,
            body_words_chars_features, ["log1p_body_words", "log1p_body_chars"],
            "month_only", month_weights, seed,
        )
        size_after = fit_numeric_diagnostic(
            train_rows, val_rows, text_by_id, portals,
            body_words_chars_features, ["log1p_body_words", "log1p_body_chars"],
            "month_plus_length", month_weights, seed,
        )
        structure_after = fit_numeric_diagnostic(
            train_rows, val_rows, text_by_id, portals,
            structural_features,
            [
                "log1p_body_words", "log1p_body_chars", "log1p_periods", "log1p_commas",
                "log1p_question_exclamation", "log1p_quote_marks", "digit_fraction", "non_ascii_fraction",
            ],
            "month_plus_length", month_weights, seed,
        )
        tfidf = tfidf_smoke_test(
            topic, train_rows, val_rows, text_by_id, portals, month_weights, seed, out_dir,
            max_features=max_features, min_df=min_df, top_n=top_n,
        )

        before_acc = word_before["controlled_validation_metrics"]["accuracy"]
        after_acc = word_after["controlled_validation_metrics"]["accuracy"]
        before_f1 = word_before["controlled_validation_metrics"]["macro_f1"]
        after_f1 = word_after["controlled_validation_metrics"]["macro_f1"]
        near_chance = abs(after_acc - chance) <= near_chance_tolerance

        summary["topics"][topic] = {
            "n_train": len(train_rows),
            "n_validation": len(val_rows),
            "length_word_only_before_control_month_only": word_before,
            "length_word_only_after_control_month_plus_length": word_after,
            "length_word_only_change": {
                "accuracy_after_minus_before": after_acc - before_acc,
                "macro_f1_after_minus_before": after_f1 - before_f1,
                "absolute_accuracy_distance_from_chance_before": abs(before_acc - chance),
                "absolute_accuracy_distance_from_chance_after": abs(after_acc - chance),
                "after_accuracy_is_near_chance": near_chance,
                "near_chance_tolerance": near_chance_tolerance,
            },
            "secondary_words_plus_chars_before_control": size_before,
            "secondary_words_plus_chars_after_control": size_after,
            "structure_only_after_control": structure_after,
            "tfidf_smoke_test": tfidf,
        }

    # Compact comparison table for quick human review.
    comparison_rows = []
    for topic in topics:
        t = summary["topics"][topic]
        for label, obj in [
            ("word_count_only_before_month_control", t["length_word_only_before_control_month_only"]),
            ("word_count_only_after_month_length_control", t["length_word_only_after_control_month_plus_length"]),
            ("words_plus_chars_before_month_control", t["secondary_words_plus_chars_before_control"]),
            ("words_plus_chars_after_month_length_control", t["secondary_words_plus_chars_after_control"]),
            ("structure_only_after_month_length_control", t["structure_only_after_control"]),
        ]:
            m = obj["controlled_validation_metrics"]
            comparison_rows.append({
                "topic": topic,
                "diagnostic": label,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "chance_accuracy": chance,
            })
        tfm = t["tfidf_smoke_test"]["controlled_validation_metrics"]
        comparison_rows.append({
            "topic": topic,
            "diagnostic": "tfidf_smoke_after_month_length_control",
            "accuracy": tfm["accuracy"],
            "macro_f1": tfm["macro_f1"],
            "chance_accuracy": chance,
        })
    write_csv(out_dir / "diagnostic_comparison.csv", comparison_rows)

    summary_path = out_dir / "diagnostic_summary.json"
    write_json(summary_path, summary)

    hashes = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "DIAGNOSTIC_HASHES.json":
            hashes[path.name] = sha256_file(path)
    write_json(out_dir / "DIAGNOSTIC_HASHES.json", hashes)

    print("\n=== EXPERIMENT DIAGNOSTICS COMPLETE ===")
    print("Model-based diagnostics used TRAIN -> VALIDATION only. TEST REMAINS SEALED.")
    print(f"Chance accuracy: {chance:.3f}")
    for topic in topics:
        t = summary["topics"][topic]
        wb = t["length_word_only_before_control_month_only"]["controlled_validation_metrics"]
        wa = t["length_word_only_after_control_month_plus_length"]["controlled_validation_metrics"]
        sa = t["secondary_words_plus_chars_after_control"]["controlled_validation_metrics"]
        st = t["structure_only_after_control"]["controlled_validation_metrics"]
        tf = t["tfidf_smoke_test"]["controlled_validation_metrics"]
        flag = "YES" if t["length_word_only_change"]["after_accuracy_is_near_chance"] else "NO"
        print(f"\n{topic}:")
        print(f"  word-count only BEFORE length control: acc={wb['accuracy']:.3f}, macro-F1={wb['macro_f1']:.3f}")
        print(f"  word-count only AFTER  length control: acc={wa['accuracy']:.3f}, macro-F1={wa['macro_f1']:.3f}")
        print(f"  after word-count accuracy near chance (+/-{near_chance_tolerance:.2f})? {flag}")
        print(f"  words+chars secondary AFTER control:  acc={sa['accuracy']:.3f}, macro-F1={sa['macro_f1']:.3f}")
        print(f"  structure-only AFTER control:         acc={st['accuracy']:.3f}, macro-F1={st['macro_f1']:.3f}")
        print(f"  TF-IDF smoke validation:              acc={tf['accuracy']:.3f}, macro-F1={tf['macro_f1']:.3f}")
    print(f"\nDiagnostics written to: {out_dir}")
    print("TF-IDF is a smoke test only; do not treat it as the final reported baseline.")


if __name__ == "__main__":
    main()
