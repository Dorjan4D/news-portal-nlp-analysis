"""Frozen configuration shared by the final TF-IDF/SVD model-selection programs.

Place this file at ``scripts/modeling_protocol_final.py`` inside ``nlp_portal_project``.
It deliberately contains configuration only (no data loading and no test access).
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, List

# Frozen corpus / experiment identifiers.
EXPECTED_CORPUS_SHA256 = "c07d1fcc31fb3fe5e92a308f95e2ea6578c391b6cb6a3d38216573c0e1c6e12e"
EXPECTED_MANIFEST_SHA256 = "282e1544747c4fe06898a7250043c6d813a8d890b9c88bd40e47bc5f7046ba9f"

TOPICS = ["topicA", "topicB"]
PORTALS = ["aljazeera", "bbc", "guardian"]
TEXT_FIELD = "model_text"

# Final common TF-IDF family. This strictly contains the old 12 settings and adds min_df=10.
TFIDF_FIXED: Dict[str, Any] = {
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
    "lowercase": True,
    "max_df": 1.0,
    "max_features": None,
    "stop_words": None,
    "token_pattern": r"(?u)\b\w\w+\b",
}
TFIDF_GRID: List[Dict[str, Any]] = [
    {"ngram_range": ng, "min_df": min_df, "sublinear_tf": sublinear}
    for ng in [(1, 1), (1, 2)]
    for min_df in [2, 3, 5, 10]
    for sublinear in [False, True]
]

# Weighted SVD: every previous rank plus the largest common power-of-two rank that is feasible.
SVD_COMPONENTS = [64, 128, 256, 512]
SVD_SEED = 20260902
SVD_N_ITER = 7

# Classical regularization / kernel grids. These are used identically for direct TF-IDF and SVD.
C_LINEAR = [0.01, 0.1, 1.0, 10.0, 100.0]
C_KERNEL = [0.1, 1.0, 10.0, 100.0]
RBF_GAMMAS = ["scale", 0.1, 0.3, 1.0, 3.0]
POLY_DEGREES = [2, 3]
POLY_GAMMAS = ["scale", 0.1]
POLY_COEF0 = [0.0, 1.0]
SIGMOID_GAMMAS = ["scale", 0.1, 0.3]
SIGMOID_COEF0 = [0.0, 1.0]


def classical_model_grid() -> List[Dict[str, Any]]:
    """Return the deterministic 91-configuration classical grid in frozen order."""
    rows: List[Dict[str, Any]] = []
    order = 0

    def add(family: str, **params: Any) -> None:
        nonlocal order
        rows.append({"family": family, "grid_order": order, **params})
        order += 1

    for c in C_LINEAR:
        add("logistic_regression", C=c)
    for c in C_LINEAR:
        add("linear_svc", C=c)
    for c in C_LINEAR:
        add("svc_linear", C=c)
    for c, gamma in product(C_KERNEL, RBF_GAMMAS):
        add("svc_rbf", C=c, gamma=gamma)
    for c, degree, gamma, coef0 in product(C_KERNEL, POLY_DEGREES, POLY_GAMMAS, POLY_COEF0):
        add("svc_poly", C=c, degree=degree, gamma=gamma, coef0=coef0)
    for c, gamma, coef0 in product(C_KERNEL, SIGMOID_GAMMAS, SIGMOID_COEF0):
        add("svc_sigmoid", C=c, gamma=gamma, coef0=coef0)
    assert len(rows) == 91
    return rows


CLASSICAL_MODEL_GRID = classical_model_grid()
CLASSICAL_FAMILIES = [
    "logistic_regression",
    "linear_svc",
    "svc_linear",
    "svc_rbf",
    "svc_poly",
    "svc_sigmoid",
]

# Known-good CPU parallelism from the previous successful classical run.
CLASSICAL_N_JOBS = 4
CLASSICAL_BACKEND = "loky"
CLASSICAL_INNER_THREADS = 1

# Final MLP search. All values from the previous MLP experiment are retained unchanged.
MLP_SEARCH_MASTER_SEED = 20260903
MLP_CONFIRMATION_MASTER_SEEDS = [20260911, 20260917, 20260923, 20260929, 20261005]
MLP_CANONICAL_STAGE1 = {
    "hidden_layers": [128],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
}
MLP_ARCHITECTURES = [[64], [128], [128, 64]]
MLP_DROPOUTS = [0.0, 0.3]
MLP_LEARNING_RATES = [3e-4, 1e-3]
MLP_WEIGHT_DECAYS = [0.0, 1e-3]
MLP_BATCH_SIZE = 32
MLP_MAX_EPOCHS = 150
MLP_MIN_EPOCHS = 15
MLP_PATIENCE = 15
MLP_MIN_DELTA = 0.001
DIRECT_STAGE1_KEEP = 4
SVD_STAGE1_KEEP = 6
MLP_STAGE2_KEEP = 6

# Final transformer expansion. The transformer notebook records these independently as part of its protocol.
ROBERTA_MODEL_NAME = "FacebookAI/roberta-base"
ROBERTA_MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
TRANSFORMER_SEARCH_MASTER_SEED = 20260904
# FE keeps the original five confirmation seeds so completed FE artifacts remain valid.
TRANSFORMER_CONFIRMATION_MASTER_SEEDS = [20260912, 20260918, 20260924, 20260930, 20261006]
# Final FT revision uses the first three of the same paired seed schedule to reduce runtime.
TRANSFORMER_FT_CONFIRMATION_MASTER_SEEDS = [20260912, 20260918, 20260924]
FE_LEARNING_RATES = [3e-4, 1e-3, 3e-3, 1e-2]
FE_WEIGHT_DECAYS = [0.0, 0.01]
FE_MAX_EPOCHS = 80
FE_PATIENCE = 10
FT_LEARNING_RATES = [1e-5, 2e-5, 3e-5]
FT_WEIGHT_DECAYS = [0.0, 0.01]
# Final revised FT schedule (2026-09-06). All six LR/WD combinations are searched
# on every FT architecture. Training is capped at 25 epochs with patience 5;
# a best epoch near the cap is reported as a limitation rather than triggering
# another validation-driven expansion of the training horizon.
FT_MAX_EPOCHS = 25
FT_PATIENCE = 5
