#!/usr/bin/env python3
"""Final direct-TF-IDF classical model selection for the portal-classification project.

Run from the project root:
    python run_classical_tfidf_final.py

The test set is sealed: test metadata are counted for integrity, but test article text is never loaded.
The script is resumable at the TF-IDF-configuration level and uses the known-good joblib ``loky``
process backend with exactly one native numerical thread per worker.
"""
from __future__ import annotations

# Set before NumPy/SciPy/sklearn imports to prevent nested BLAS/OpenMP oversubscription.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
from joblib import Parallel, delayed, parallel_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC, SVC
import sklearn

PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.modeling_protocol_final import (  # noqa: E402
    CLASSICAL_BACKEND,
    CLASSICAL_FAMILIES,
    CLASSICAL_INNER_THREADS,
    CLASSICAL_MODEL_GRID,
    CLASSICAL_N_JOBS,
    EXPECTED_CORPUS_SHA256,
    EXPECTED_MANIFEST_SHA256,
    PORTALS,
    TEXT_FIELD,
    TFIDF_FIXED,
    TFIDF_GRID,
    TOPICS,
)

CORPUS_PATH = PROJECT_ROOT / "frozen_data" / "corpus_frozen.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "experiment_data" / "experiment_manifest.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "modeling_results" / "classical_tfidf_final"
LABEL_TO_ID = {p: i for i, p in enumerate(PORTALS)}
ID_TO_LABEL = {i: p for p, i in LABEL_TO_ID.items()}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_jsonable(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def stable_ids_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def atomic_csv_dump(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def tfidf_config_id(cfg: Mapping[str, Any]) -> str:
    ng = tuple(cfg["ngram_range"])
    return f"ng{ng[0]}{ng[1]}_df{int(cfg['min_df'])}_sub{int(bool(cfg['sublinear_tf']))}"


def make_vectorizer(cfg: Mapping[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        **TFIDF_FIXED,
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=int(cfg["min_df"]),
        sublinear_tf=bool(cfg["sublinear_tf"]),
        dtype=np.float64,
    )


def normalized_fit_weights(df: pd.DataFrame) -> np.ndarray:
    w = df.balance_weight.to_numpy(np.float64)
    return w / w.mean()


def metrics(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    labels = np.arange(len(PORTALS))
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, sample_weight=w, zero_division=0
    )
    return {
        "weighted_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", sample_weight=w, zero_division=0)),
        "weighted_accuracy": float(accuracy_score(y_true, y_pred, sample_weight=w)),
        "raw_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "raw_accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision_by_class": {PORTALS[i]: float(p[i]) for i in labels},
        "weighted_recall_by_class": {PORTALS[i]: float(r[i]) for i in labels},
        "weighted_f1_by_class": {PORTALS[i]: float(f[i]) for i in labels},
        "weighted_support_by_class": {PORTALS[i]: float(support[i]) for i in labels},
        "weighted_confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels, sample_weight=w).tolist(),
        "raw_confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def make_estimator(spec: Mapping[str, Any]):
    fam = spec["family"]
    c = float(spec["C"])
    if fam == "logistic_regression":
        return LogisticRegression(C=c, solver="lbfgs", max_iter=5000, tol=1e-4, random_state=0)
    if fam == "linear_svc":
        return LinearSVC(C=c, loss="squared_hinge", dual="auto", max_iter=20000, tol=1e-4, random_state=0)
    common = dict(C=c, cache_size=512, shrinking=True, probability=False, decision_function_shape="ovr", random_state=0)
    if fam == "svc_linear":
        return SVC(kernel="linear", **common)
    if fam == "svc_rbf":
        return SVC(kernel="rbf", gamma=spec["gamma"], **common)
    if fam == "svc_poly":
        return SVC(kernel="poly", degree=int(spec["degree"]), gamma=spec["gamma"], coef0=float(spec["coef0"]), **common)
    if fam == "svc_sigmoid":
        return SVC(kernel="sigmoid", gamma=spec["gamma"], coef0=float(spec["coef0"]), **common)
    raise ValueError(fam)


def fit_one(spec: Mapping[str, Any], Xtr, ytr, fit_w, Xva, yva, train_w_raw, val_w) -> Dict[str, Any]:
    t0 = time.time()
    caught = []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        model = make_estimator(spec)
        # libsvm-backed SVC estimators sort sparse CSR indices in-place during
        # fit/predict. Under joblib/loky, large sparse backing arrays may be
        # auto-memmapped read-only, so give every libsvm SVC task its own
        # writable sparse copies. LR/LinearSVC do not need this copy.
        needs_writable_sparse = str(spec["family"]).startswith("svc_")
        X_fit = Xtr.copy() if needs_writable_sparse else Xtr
        model.fit(X_fit, ytr, sample_weight=fit_w)
        Xtr_pred = Xtr.copy() if needs_writable_sparse else Xtr
        Xva_pred = Xva.copy() if needs_writable_sparse else Xva
        tr_pred = model.predict(Xtr_pred)
        va_pred = model.predict(Xva_pred)
        caught = [f"{w.category.__name__}: {w.message}" for w in ws]
    tr = metrics(ytr, tr_pred, train_w_raw)
    va = metrics(yva, va_pred, val_w)
    return {
        **dict(spec),
        "fit_seconds": time.time() - t0,
        "train_weighted_macro_f1": tr["weighted_macro_f1"],
        "train_weighted_accuracy": tr["weighted_accuracy"],
        "validation_weighted_macro_f1": va["weighted_macro_f1"],
        "validation_weighted_accuracy": va["weighted_accuracy"],
        "validation_raw_macro_f1": va["raw_macro_f1"],
        "validation_raw_accuracy": va["raw_accuracy"],
        "validation_weighted_f1_by_class": va["weighted_f1_by_class"],
        "validation_weighted_confusion_matrix": va["weighted_confusion_matrix"],
        "validation_raw_confusion_matrix": va["raw_confusion_matrix"],
        "warnings": caught,
    }


def load_data():
    rows = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("experiment_eligible") and r.get("split") in {"train", "validation", "test"}:
                rows.append(r)
    manifest = pd.DataFrame(rows)
    expected = {
        ("train", "topicA"): 811, ("validation", "topicA"): 272, ("test", "topicA"): 281,
        ("train", "topicB"): 721, ("validation", "topicB"): 245, ("test", "topicB"): 237,
    }
    assert manifest.groupby(["split", "topic"]).size().to_dict() == expected
    assert manifest.article_id.is_unique

    model_manifest = manifest[manifest.split.isin(["train", "validation"])].copy()
    allowed = set(model_manifest.article_id)
    text_by_id = {}
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            aid = r.get("article_id")
            if aid in allowed:
                text_by_id[aid] = r[TEXT_FIELD]
    assert set(text_by_id) == allowed
    model_manifest["text"] = model_manifest.article_id.map(text_by_id)
    assert model_manifest.text.notna().all()

    topic_data = {}
    summary = []
    for topic in TOPICS:
        tr = model_manifest[(model_manifest.topic == topic) & (model_manifest.split == "train")].sort_values("article_id").reset_index(drop=True)
        va = model_manifest[(model_manifest.topic == topic) & (model_manifest.split == "validation")].sort_values("article_id").reset_index(drop=True)
        tr["label_id"] = tr.portal.map(LABEL_TO_ID).astype(int)
        va["label_id"] = va.portal.map(LABEL_TO_ID).astype(int)
        topic_data[topic] = {"train": tr, "validation": va}
        for split, df in [("train", tr), ("validation", va)]:
            for portal in PORTALS:
                s = df[df.portal == portal]
                summary.append({"topic": topic, "split": split, "portal": portal, "raw_n": len(s), "effective_weight_mass": float(s.balance_weight.sum())})
    return manifest, topic_data, pd.DataFrame(summary)


def rank_best(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(
        ["validation_weighted_macro_f1", "validation_weighted_accuracy", "tfidf_order", "grid_order"],
        ascending=[False, False, True, True], kind="mergesort"
    ).iloc[0]


def stringify_params(row: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ["C", "gamma", "degree", "coef0"]
    return {k: row[k] for k in keys if k in row and not (isinstance(row[k], float) and np.isnan(row[k]))}


def save_top_features(model, vectorizer: TfidfVectorizer, family: str, topic: str, top_n: int = 40) -> None:
    if not hasattr(model, "coef_"):
        return
    names = np.asarray(vectorizer.get_feature_names_out())
    coef_raw = model.coef_
    coef = coef_raw.toarray() if hasattr(coef_raw, "toarray") else np.asarray(coef_raw)
    rows = []
    if family in {"logistic_regression", "linear_svc"} and coef.shape[0] == len(PORTALS):
        labels = PORTALS
    elif family == "svc_linear" and coef.shape[0] == 3:
        labels = ["aljazeera_vs_bbc", "aljazeera_vs_guardian", "bbc_vs_guardian"]
    else:
        labels = [f"row_{i}" for i in range(coef.shape[0])]
    for i, label in enumerate(labels):
        order = np.argsort(coef[i])
        for rank, j in enumerate(order[-top_n:][::-1], 1):
            rows.append({"topic": topic, "family": family, "coefficient_row": label, "direction": "positive", "rank": rank, "feature": names[j], "coefficient": float(coef[i, j])})
        for rank, j in enumerate(order[:top_n], 1):
            rows.append({"topic": topic, "family": family, "coefficient_row": label, "direction": "negative", "rank": rank, "feature": names[j], "coefficient": float(coef[i, j])})
    atomic_csv_dump(pd.DataFrame(rows), OUTPUT_DIR / "interpretability" / f"{topic}__{family}__top_coefficients.csv")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Delete the final classical output directory before running.")
    args = parser.parse_args()

    if args.overwrite and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in [CORPUS_PATH, MANIFEST_PATH, PROJECT_ROOT / "scripts" / "modeling_protocol_final.py"]:
        if not p.exists():
            raise FileNotFoundError(p)
    actual_corpus = sha256_file(CORPUS_PATH)
    actual_manifest = sha256_file(MANIFEST_PATH)
    assert actual_corpus == EXPECTED_CORPUS_SHA256
    assert actual_manifest == EXPECTED_MANIFEST_SHA256

    protocol = {
        "stage_name": "classical_tfidf_final",
        "corpus_sha256": actual_corpus,
        "manifest_sha256": actual_manifest,
        "test_policy": "Test metadata counted for integrity only; test article text is never loaded/transformed/predicted/evaluated.",
        "tfidf_fixed": TFIDF_FIXED,
        "tfidf_grid": TFIDF_GRID,
        "classical_model_grid": CLASSICAL_MODEL_GRID,
        "selection_metric": "validation_weighted_macro_f1",
        "fit_weights": "frozen train balance_weight globally rescaled to mean 1",
        "validation_weights": "frozen validation balance_weight",
        "parallelism": {"backend": CLASSICAL_BACKEND, "n_jobs": CLASSICAL_N_JOBS, "native_threads_per_worker": CLASSICAL_INNER_THREADS},
    }
    protocol_hash = sha256_jsonable(protocol)
    protocol["protocol_hash"] = protocol_hash
    pp = OUTPUT_DIR / "PROTOCOL.json"
    if pp.exists() and json.loads(pp.read_text()) != protocol:
        raise RuntimeError("Existing final classical output uses a different protocol. Use --overwrite or archive it.")
    atomic_json_dump(protocol, pp)
    atomic_json_dump({
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "protocol_hash": protocol_hash,
    }, OUTPUT_DIR / "ENVIRONMENT.json")

    manifest, topic_data, data_summary = load_data()
    atomic_csv_dump(data_summary, OUTPUT_DIR / "DATA_SPLIT_SUMMARY.csv")

    print("=== FINAL CLASSICAL TF-IDF MODEL SELECTION ===")
    print("TEST SET: SEALED (train -> validation only)")
    print("TF-IDF configs:", len(TFIDF_GRID), "classifier configs/representation:", len(CLASSICAL_MODEL_GRID))
    print(f"Parallel backend: {CLASSICAL_BACKEND} processes, n_jobs={CLASSICAL_N_JOBS}, native threads/worker={CLASSICAL_INNER_THREADS}")

    all_records = []
    global_start = time.time()
    total_rep = len(TOPICS) * len(TFIDF_GRID)
    rep_counter = 0
    for topic in TOPICS:
        tr = topic_data[topic]["train"]; va = topic_data[topic]["validation"]
        ytr = tr.label_id.to_numpy(np.int64); yva = va.label_id.to_numpy(np.int64)
        fit_w = normalized_fit_weights(tr); train_w_raw = tr.balance_weight.to_numpy(np.float64); val_w = va.balance_weight.to_numpy(np.float64)
        for tfidf_order, cfg in enumerate(TFIDF_GRID):
            rep_counter += 1
            tid = tfidf_config_id(cfg)
            rep_dir = OUTPUT_DIR / "per_representation" / topic / tid
            result_path = rep_dir / "results.csv"
            meta_path = rep_dir / "tfidf_metadata.json"
            if result_path.exists() and meta_path.exists():
                old = pd.read_csv(result_path)
                if len(old) == len(CLASSICAL_MODEL_GRID):
                    all_records.extend(old.to_dict("records"))
                    print(f"[{rep_counter:02d}/{total_rep}] reuse {topic} {tid}")
                    continue

            t_vec = time.time()
            vec = make_vectorizer(cfg)
            Xtr = vec.fit_transform(tr.text.tolist())
            Xva = vec.transform(va.text.tolist())
            vocab_hash = sha256_jsonable(vec.vocabulary_)
            idf_hash = hashlib.sha256(np.asarray(vec.idf_, dtype=np.float64).tobytes()).hexdigest()
            atomic_json_dump({
                "protocol_hash": protocol_hash, "topic": topic, "tfidf_config_id": tid,
                "tfidf_config": {**cfg, "ngram_range": list(cfg["ngram_range"])},
                "n_features": int(Xtr.shape[1]), "vectorize_seconds": time.time() - t_vec,
                "vocabulary_sha256": vocab_hash, "idf_sha256": idf_hash,
                "train_ids_sha256": stable_ids_hash(tr.article_id.tolist()), "validation_ids_sha256": stable_ids_hash(va.article_id.tolist()),
            }, meta_path)

            print(f"[{rep_counter:02d}/{total_rep}] {topic} {tid}: {Xtr.shape[1]:,} features; fitting {len(CLASSICAL_MODEL_GRID)} models")
            with parallel_config(backend=CLASSICAL_BACKEND, n_jobs=CLASSICAL_N_JOBS, inner_max_num_threads=CLASSICAL_INNER_THREADS):
                recs = Parallel()(delayed(fit_one)(spec, Xtr, ytr, fit_w, Xva, yva, train_w_raw, val_w) for spec in CLASSICAL_MODEL_GRID)
            for r in recs:
                r.update({
                    "protocol_hash": protocol_hash, "topic": topic, "tfidf_order": tfidf_order,
                    "tfidf_config_id": tid, "ngram_range": str(tuple(cfg["ngram_range"])), "min_df": int(cfg["min_df"]),
                    "sublinear_tf": bool(cfg["sublinear_tf"]), "n_tfidf_features": int(Xtr.shape[1]),
                })
            rdf = pd.DataFrame(recs)
            atomic_csv_dump(rdf, result_path)
            all_records.extend(recs)
            best_now = rdf.sort_values(["validation_weighted_macro_f1", "validation_weighted_accuracy", "grid_order"], ascending=[False, False, True]).iloc[0]
            print(f"    best: {best_now.family} val_wF1={best_now.validation_weighted_macro_f1:.4f}; elapsed total={(time.time()-global_start)/60:.1f} min")

    all_df = pd.DataFrame(all_records)
    atomic_csv_dump(all_df.sort_values(["topic", "tfidf_order", "grid_order"]), OUTPUT_DIR / "all_results.csv")

    winners = []
    for topic in TOPICS:
        for fam in CLASSICAL_FAMILIES:
            best = rank_best(all_df[(all_df.topic == topic) & (all_df.family == fam)])
            winners.append(best.to_dict())
    best_df = pd.DataFrame(winners)
    atomic_csv_dump(best_df, OUTPUT_DIR / "best_by_family.csv")

    # Refit each selected model once, verify its validation score, save exact bundle/predictions and interpretable coefficients.
    for _, win in best_df.iterrows():
        topic, fam = win.topic, win.family
        tr = topic_data[topic]["train"]; va = topic_data[topic]["validation"]
        cfg = next(c for c in TFIDF_GRID if tfidf_config_id(c) == win.tfidf_config_id)
        spec = next(s for s in CLASSICAL_MODEL_GRID if s["family"] == fam and int(s["grid_order"]) == int(win.grid_order))
        vec = make_vectorizer(cfg); Xtr = vec.fit_transform(tr.text.tolist()); Xva = vec.transform(va.text.tolist())
        model = make_estimator(spec); model.fit(Xtr, tr.label_id.to_numpy(np.int64), sample_weight=normalized_fit_weights(tr))
        pred = model.predict(Xva)
        m = metrics(va.label_id.to_numpy(np.int64), pred, va.balance_weight.to_numpy(np.float64))
        if abs(m["weighted_macro_f1"] - float(win.validation_weighted_macro_f1)) > 1e-10:
            raise RuntimeError(f"Selected refit mismatch for {topic}/{fam}")
        out_dir = OUTPUT_DIR / "selected_models" / topic / fam
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": vec, "model": model, "metadata": {"protocol_hash": protocol_hash, "topic": topic, "family": fam, "tfidf_config": cfg, "classifier_spec": spec}}, out_dir / "model_bundle.joblib", compress=3)
        atomic_json_dump({"topic": topic, "family": fam, "tfidf_config": {**cfg, "ngram_range": list(cfg["ngram_range"])}, "classifier_spec": spec, "validation_metrics": m, "protocol_hash": protocol_hash, "test_evaluated": False}, out_dir / "metadata.json")
        pdf = va[["article_id", "portal", "year_month", "length_bin", "balance_weight"]].copy()
        pdf["predicted_portal"] = [ID_TO_LABEL[int(x)] for x in pred]
        atomic_csv_dump(pdf, OUTPUT_DIR / "validation_predictions" / f"{topic}__{fam}.csv")
        save_top_features(model, vec, fam, topic)

    # Compact selected-family comparison plot per topic.
    for topic in TOPICS:
        s = best_df[best_df.topic == topic].copy()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(s.family, s.validation_weighted_macro_f1)
        ax.set_ylabel("Validation weighted macro-F1")
        ax.set_title(f"{topic}: direct TF-IDF classical models")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        p = OUTPUT_DIR / "plots" / f"{topic}_best_classical_models.png"; p.parent.mkdir(parents=True, exist_ok=True); fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)

    report = [
        "# Final direct TF-IDF classical model-selection report", "",
        f"- Protocol hash: `{protocol_hash}`", f"- Corpus SHA256: `{actual_corpus}`", f"- Manifest SHA256: `{actual_manifest}`",
        "- Test remained sealed: test article text was never loaded, transformed, predicted, or evaluated.", "",
        "## Selected validation configurations", "",
        "| Topic | Family | TF-IDF | Parameters | Val wMacro-F1 | Val wAcc | Train wMacro-F1 |", "|---|---|---|---|---:|---:|---:|",
    ]
    for _, r in best_df.sort_values(["topic", "family"]).iterrows():
        report.append(f"| {r.topic} | {r.family} | `{r.tfidf_config_id}` | `{stringify_params(r)}` | {r.validation_weighted_macro_f1:.4f} | {r.validation_weighted_accuracy:.4f} | {r.train_weighted_macro_f1:.4f} |")
    report += ["", f"Total wall time: {(time.time()-global_start)/60:.1f} minutes.", ""]
    (OUTPUT_DIR / "MODEL_SELECTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("\n=== COMPLETE ===")
    print(best_df[["topic", "family", "tfidf_config_id", "validation_weighted_macro_f1", "validation_weighted_accuracy"]].to_string(index=False))
    print("TEST REMAINS SEALED.")


if __name__ == "__main__":
    main()
