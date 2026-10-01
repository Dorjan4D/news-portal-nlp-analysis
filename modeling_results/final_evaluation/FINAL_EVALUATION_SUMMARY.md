# Final evaluation summary

- Evaluation protocol hash: `f009cba211347e3538cc8fd0877030ed128c4f42ff4ce0f80cb153adba1e5a68`
- Corpus SHA256: `c07d1fcc31fb3fe5e92a308f95e2ea6578c391b6cb6a3d38216573c0e1c6e12e`
- Manifest SHA256: `282e1544747c4fe06898a7250043c6d813a8d890b9c88bd40e47bc5f7046ba9f`
- Predictive training/model selection: frozen before test unseal.
- Test set: unsealed only in this notebook after the pre-unseal integrity audit.

## Primary configuration comparison

Neural configurations are summarized by the mean and sample SD across all predefined confirmation seeds. Classical models and random-init controls are single predefined instances.

| Source | Target | Model | Test weighted macro-F1 | Seed SD |
|---|---|---|---:|---:|
| topicA | topicA | FT-ATTN | 0.9215 | 0.0104 |
| topicA | topicA | FT-MEAN | 0.9133 | 0.0284 |
| topicA | topicA | FT-512 | 0.9083 | 0.0219 |
| topicA | topicA | Direct TF-IDF MLP | 0.8893 | 0.0064 |
| topicA | topicA | Direct TF-IDF classical | 0.8741 | — |
| topicA | topicA | Weighted-SVD MLP | 0.8639 | 0.0143 |
| topicA | topicA | Weighted-SVD classical | 0.8497 | — |
| topicA | topicA | FE-MEAN | 0.7267 | 0.0275 |
| topicA | topicA | RANDOM-MEAN | 0.6919 | — |
| topicA | topicA | FE-ATTN | 0.6373 | 0.0106 |
| topicA | topicA | FE-512 | 0.6274 | 0.0355 |
| topicA | topicA | RANDOM-512 | 0.3659 | — |
| topicA | topicA | RANDOM-ATTN | 0.1742 | — |
| topicA | topicB | FT-ATTN | 0.8863 | 0.0483 |
| topicA | topicB | FT-MEAN | 0.8766 | 0.0276 |
| topicA | topicB | FT-512 | 0.8662 | 0.0086 |
| topicA | topicB | RANDOM-MEAN | 0.6498 | — |
| topicA | topicB | Direct TF-IDF classical | 0.5738 | — |
| topicA | topicB | Weighted-SVD classical | 0.5556 | — |
| topicA | topicB | Weighted-SVD MLP | 0.5161 | 0.0221 |
| topicA | topicB | Direct TF-IDF MLP | 0.4940 | 0.0267 |
| topicA | topicB | RANDOM-512 | 0.3733 | — |
| topicA | topicB | FE-ATTN | 0.2811 | 0.0599 |
| topicA | topicB | FE-MEAN | 0.2461 | 0.0291 |
| topicA | topicB | FE-512 | 0.2189 | 0.0412 |
| topicA | topicB | RANDOM-ATTN | 0.1667 | — |
| topicB | topicA | FT-ATTN | 0.7607 | 0.0274 |
| topicB | topicA | FT-MEAN | 0.7166 | 0.0242 |
| topicB | topicA | FT-512 | 0.7055 | 0.0450 |
| topicB | topicA | Weighted-SVD classical | 0.5094 | — |
| topicB | topicA | Direct TF-IDF classical | 0.4599 | — |
| topicB | topicA | Weighted-SVD MLP | 0.4445 | 0.0566 |
| topicB | topicA | FE-512 | 0.3879 | 0.0576 |
| topicB | topicA | FE-MEAN | 0.3876 | 0.0217 |
| topicB | topicA | Direct TF-IDF MLP | 0.3609 | 0.0247 |
| topicB | topicA | FE-ATTN | 0.3593 | 0.0408 |
| topicB | topicA | RANDOM-MEAN | 0.2768 | — |
| topicB | topicA | RANDOM-ATTN | 0.2768 | — |
| topicB | topicA | RANDOM-512 | 0.2505 | — |
| topicB | topicB | FT-MEAN | 0.9564 | 0.0196 |
| topicB | topicB | FT-ATTN | 0.9448 | 0.0178 |
| topicB | topicB | FT-512 | 0.9278 | 0.0339 |
| topicB | topicB | Weighted-SVD classical | 0.8913 | — |
| topicB | topicB | Direct TF-IDF classical | 0.8824 | — |
| topicB | topicB | Weighted-SVD MLP | 0.8770 | 0.0142 |
| topicB | topicB | Direct TF-IDF MLP | 0.8004 | 0.0158 |
| topicB | topicB | FE-MEAN | 0.7996 | 0.0136 |
| topicB | topicB | FE-512 | 0.7524 | 0.0137 |
| topicB | topicB | FE-ATTN | 0.7363 | 0.0328 |
| topicB | topicB | RANDOM-MEAN | 0.5006 | — |
| topicB | topicB | RANDOM-ATTN | 0.5006 | — |
| topicB | topicB | RANDOM-512 | 0.4808 | — |

## Predefined neural ensemble predictors

| Source | Target | Model | Mean seed wF1 | SD | Soft-ensemble wF1 |
|---|---|---|---:|---:|---:|
| topicA | topicA | FT-ATTN | 0.9215 | 0.0104 | 0.9293 |
| topicA | topicA | FT-MEAN | 0.9133 | 0.0284 | 0.9258 |
| topicA | topicA | FT-512 | 0.9083 | 0.0219 | 0.9349 |
| topicA | topicA | direct_mlp | 0.8893 | 0.0064 | 0.9007 |
| topicA | topicA | svd_mlp | 0.8639 | 0.0143 | 0.8755 |
| topicA | topicA | FE-MEAN | 0.7267 | 0.0275 | 0.7505 |
| topicA | topicA | FE-ATTN | 0.6373 | 0.0106 | 0.6329 |
| topicA | topicA | FE-512 | 0.6274 | 0.0355 | 0.6553 |
| topicA | topicB | FT-ATTN | 0.8863 | 0.0483 | 0.9022 |
| topicA | topicB | FT-MEAN | 0.8766 | 0.0276 | 0.8862 |
| topicA | topicB | FT-512 | 0.8662 | 0.0086 | 0.8691 |
| topicA | topicB | svd_mlp | 0.5161 | 0.0221 | 0.5231 |
| topicA | topicB | direct_mlp | 0.4940 | 0.0267 | 0.5070 |
| topicA | topicB | FE-ATTN | 0.2811 | 0.0599 | 0.2918 |
| topicA | topicB | FE-MEAN | 0.2461 | 0.0291 | 0.2592 |
| topicA | topicB | FE-512 | 0.2189 | 0.0412 | 0.2210 |
| topicB | topicA | FT-ATTN | 0.7607 | 0.0274 | 0.7983 |
| topicB | topicA | FT-MEAN | 0.7166 | 0.0242 | 0.7460 |
| topicB | topicA | FT-512 | 0.7055 | 0.0450 | 0.7435 |
| topicB | topicA | svd_mlp | 0.4445 | 0.0566 | 0.4230 |
| topicB | topicA | FE-512 | 0.3879 | 0.0576 | 0.3839 |
| topicB | topicA | FE-MEAN | 0.3876 | 0.0217 | 0.3739 |
| topicB | topicA | direct_mlp | 0.3609 | 0.0247 | 0.3542 |
| topicB | topicA | FE-ATTN | 0.3593 | 0.0408 | 0.3458 |
| topicB | topicB | FT-MEAN | 0.9564 | 0.0196 | 0.9572 |
| topicB | topicB | FT-ATTN | 0.9448 | 0.0178 | 0.9588 |
| topicB | topicB | FT-512 | 0.9278 | 0.0339 | 0.9438 |
| topicB | topicB | svd_mlp | 0.8770 | 0.0142 | 0.8791 |
| topicB | topicB | direct_mlp | 0.8004 | 0.0158 | 0.7966 |
| topicB | topicB | FE-MEAN | 0.7996 | 0.0136 | 0.8118 |
| topicB | topicB | FE-512 | 0.7524 | 0.0137 | 0.7677 |
| topicB | topicB | FE-ATTN | 0.7363 | 0.0328 | 0.7683 |

## Attention-analysis policy

ATTN is compared with the token-count-weighted MEAN prior using only two primary diagnostics: balance-weighted mean article-level TV distance and a balance-weighted absolute-position log2(ATTN/prior) profile with raw/weighted support. Multi-chunk coverage is reported as context because one-chunk articles necessarily have TV=0. The article-level scorer is small and order-agnostic (768→128→1 plus the token-count prior), so positional patterns describe content occurring at positions rather than explicit ordinal-position awareness.

## Interpretation policy

All rankings above are descriptive comparisons among models frozen before test access. No result in this notebook is used to retrain, retune, recalibrate, alter ensemble membership, or replace the validation-selected configurations. Confidence/entropy and attention diagnostics are descriptive only; softmax scores are not assumed calibrated, and chunk attention is not treated as a causal explanation of model reasoning.
