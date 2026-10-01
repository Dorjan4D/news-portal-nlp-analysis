# Final direct TF-IDF classical model-selection report

- Protocol hash: `e17602edbdfe5e440fcac47c3e1b490e2071a37cd88fb4e057422ca85552e8b1`
- Corpus SHA256: `c07d1fcc31fb3fe5e92a308f95e2ea6578c391b6cb6a3d38216573c0e1c6e12e`
- Manifest SHA256: `282e1544747c4fe06898a7250043c6d813a8d890b9c88bd40e47bc5f7046ba9f`
- Test remained sealed: test article text was never loaded, transformed, predicted, or evaluated.

## Selected validation configurations

| Topic | Family | TF-IDF | Parameters | Val wMacro-F1 | Val wAcc | Train wMacro-F1 |
|---|---|---|---|---:|---:|---:|
| topicA | linear_svc | `ng12_df10_sub1` | `{'C': 100.0}` | 0.9203 | 0.9202 | 1.0000 |
| topicA | logistic_regression | `ng12_df10_sub1` | `{'C': 100.0}` | 0.9068 | 0.9066 | 1.0000 |
| topicA | svc_linear | `ng12_df2_sub1` | `{'C': 10.0}` | 0.9061 | 0.9053 | 1.0000 |
| topicA | svc_poly | `ng12_df10_sub1` | `{'C': 100.0, 'gamma': 0.1, 'degree': 2.0, 'coef0': 1.0}` | 0.9045 | 0.9045 | 1.0000 |
| topicA | svc_rbf | `ng12_df2_sub1` | `{'C': 100.0, 'gamma': 0.1}` | 0.8971 | 0.8963 | 1.0000 |
| topicA | svc_sigmoid | `ng12_df10_sub1` | `{'C': 100.0, 'gamma': 'scale', 'coef0': 1.0}` | 0.9179 | 0.9175 | 1.0000 |
| topicB | linear_svc | `ng11_df10_sub1` | `{'C': 1.0}` | 0.9107 | 0.9112 | 1.0000 |
| topicB | logistic_regression | `ng11_df10_sub1` | `{'C': 10.0}` | 0.8959 | 0.8977 | 1.0000 |
| topicB | svc_linear | `ng11_df5_sub1` | `{'C': 1.0}` | 0.8870 | 0.8865 | 0.9989 |
| topicB | svc_poly | `ng11_df5_sub1` | `{'C': 10.0, 'gamma': 0.1, 'degree': 2.0, 'coef0': 1.0}` | 0.8799 | 0.8799 | 1.0000 |
| topicB | svc_rbf | `ng11_df10_sub1` | `{'C': 100.0, 'gamma': 0.1}` | 0.8776 | 0.8774 | 1.0000 |
| topicB | svc_sigmoid | `ng11_df10_sub1` | `{'C': 1.0, 'gamma': 'scale', 'coef0': 0.0}` | 0.8947 | 0.8941 | 0.9860 |

Total wall time: 60.5 minutes.
