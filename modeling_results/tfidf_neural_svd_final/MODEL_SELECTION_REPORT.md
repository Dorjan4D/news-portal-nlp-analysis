# Final TF-IDF / SVD / MLP model-selection report

- Protocol hash: `936e424d488c95ab4c3b8df23af51aea4759ceb3e380139958cf1876d70057c9`
- Corpus SHA256: `c07d1fcc31fb3fe5e92a308f95e2ea6578c391b6cb6a3d38216573c0e1c6e12e`
- Manifest SHA256: `282e1544747c4fe06898a7250043c6d813a8d890b9c88bd40e47bc5f7046ba9f`
- Test remained sealed throughout this notebook.

## Selected MLPs

| Topic | Family | Representation | Hidden | Dropout | LR | WD | Mean val wMacro-F1 | SD |
|---|---|---|---|---:|---:|---:|---:|---:|
| topicA | direct | `direct__ng12_df10_sub1` | `[128, 64]` | 0.3 | 0.001 | 0.001 | 0.9021 | 0.0042 |
| topicA | svd | `svd__ng12_df10_sub1__k0256` | `[128, 64]` | 0 | 0.001 | 0 | 0.9186 | 0.0024 |
| topicB | direct | `direct__ng12_df10_sub0` | `[64]` | 0.3 | 0.001 | 0 | 0.8124 | 0.0046 |
| topicB | svd | `svd__ng11_df10_sub1__k0512` | `[128, 64]` | 0.3 | 0.0003 | 0 | 0.8947 | 0.0043 |

## Best SVD classical models

| Topic | Family | Representation | Val wMacro-F1 | Val wAcc |
|---|---|---|---:|---:|
| topicA | linear_svc | `svd__ng12_df10_sub1__k0256` | 0.9055 | 0.9054 |
| topicA | logistic_regression | `svd__ng12_df10_sub1__k0512` | 0.9050 | 0.9048 |
| topicA | svc_linear | `svd__ng12_df10_sub1__k0512` | 0.8966 | 0.8969 |
| topicA | svc_poly | `svd__ng12_df3_sub1__k0512` | 0.9017 | 0.9004 |
| topicA | svc_rbf | `svd__ng12_df10_sub1__k0064` | 0.8534 | 0.8533 |
| topicA | svc_sigmoid | `svd__ng12_df5_sub1__k0512` | 0.9003 | 0.8999 |
| topicB | linear_svc | `svd__ng11_df10_sub1__k0512` | 0.8902 | 0.8913 |
| topicB | logistic_regression | `svd__ng11_df10_sub1__k0512` | 0.8918 | 0.8931 |
| topicB | svc_linear | `svd__ng11_df2_sub1__k0512` | 0.8935 | 0.8932 |
| topicB | svc_poly | `svd__ng11_df10_sub1__k0512` | 0.8858 | 0.8860 |
| topicB | svc_rbf | `svd__ng11_df10_sub0__k0512` | 0.7778 | 0.7788 |
| topicB | svc_sigmoid | `svd__ng12_df10_sub1__k0512` | 0.9062 | 0.9064 |

The classical SVD benchmark was diagnostic/independent and did not influence either MLP shortlist. Final test evaluation must load the exact selected bundles/checkpoints and must not refit TF-IDF, SVD, or neural/classical heads on target-topic data.
