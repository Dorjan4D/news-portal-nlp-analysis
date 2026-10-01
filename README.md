# Linguistic and Narrative Differences Across News Portals on Selected Topics

NLP and machine-learning project investigating whether news outlets exhibit systematic linguistic and narrative-related patterns that can be detected from article text.

The project uses **outlet classification** as a measurable proxy: models attempt to distinguish articles from **Al Jazeera, BBC, and The Guardian** using cleaned article bodies after reducing obvious source-identifying clues.

Two topics are studied:

- **Topic A:** Israel / Gaza / Palestine and related regional conflict reporting
- **Topic B:** economy, finance, markets, fiscal and monetary policy, and trade

In addition to normal same-topic evaluation, models trained on one topic are evaluated directly on the other topic to examine how well outlet-related patterns generalize across subjects.

## Key results

The cleaned article bodies contain substantial portal-predictive information.

| Training → Test topic | Direct TF-IDF | FT-512 | FT-MEAN | FT-ATTN |
|---|---:|---:|---:|---:|
| A → A | 0.8741 | 0.9083 ± 0.0219 | 0.9133 ± 0.0284 | **0.9215 ± 0.0104** |
| A → B | 0.5738 | 0.8662 ± 0.0086 | 0.8766 ± 0.0276 | **0.8863 ± 0.0483** |
| B → A | 0.4599 | 0.7055 ± 0.0450 | 0.7166 ± 0.0242 | **0.7607 ± 0.0274** |
| B → B | 0.8824 | 0.9278 ± 0.0339 | **0.9564 ± 0.0196** | 0.9448 ± 0.0178 |

Values are **balance-weighted macro-F1**. Neural results are mean ± sample standard deviation across predefined confirmation seeds.

Direct TF-IDF performs strongly on same-topic classification but loses considerably more performance when transferred to the other topic. Fully fine-tuned RoBERTa retains substantially more of its performance across topics.

Attention pooling is competitive but is not consistently better than weighted mean pooling.

For the complete methodology, results, analyses, and limitations, see the **[final report](report/final_report.pdf)**.

## Dataset

An original corpus of **2,764 English-language news articles** was collected from Al Jazeera, BBC, and The Guardian.

After publication-month and article-length controls, **2,567 articles** remained eligible for the modeling experiments.

The data pipeline included:

- article collection and extraction;
- page-type and topic filtering;
- cleaning and source-clue reduction;
- exact and near-duplicate removal;
- corpus auditing;
- publication-month and article-length controls;
- stratified train / validation / sealed-test splitting;
- balanced training and weighted evaluation.

The classifiers use only the **cleaned article body**. Headlines are retained as metadata and for separate checks but are not used as input to the reported models.

The full article corpus is **not distributed in this public repository**. The repository instead contains the collection and processing code, experiment metadata, aggregate diagnostics, model-selection results, and final analyses.

## Models and representations

Several NLP and machine-learning approaches are compared.

### TF-IDF and classical models

Word-level TF-IDF representations are evaluated with multiple classical classifiers, including:

- Logistic Regression
- LinearSVC
- linear SVC
- RBF SVC
- polynomial SVC
- sigmoid SVC

LinearSVC was selected on validation data for both topics.

### TF-IDF, SVD, and neural models

The project also evaluates:

- MLP classifiers directly on TF-IDF features;
- weighted TruncatedSVD representations at multiple dimensionalities;
- classical classifiers on SVD representations;
- MLP classifiers on SVD representations.

### RoBERTa

RoBERTa-base is evaluated under three encoder setups:

- **FE** — pretrained encoder kept frozen;
- **FT** — full fine-tuning of the pretrained encoder;
- **RANDOM** — randomly initialized RoBERTa controls.

Because many articles exceed a single Transformer input, articles are split into non-overlapping chunks and three document-level representation strategies are compared:

- **512** — use only the first chunk;
- **MEAN** — token-count-weighted mean of all chunk representations;
- **ATTN** — learned attention pooling over all chunk representations.

## Leakage control and evaluation

A major goal of the experimental design was to reduce simple shortcuts that could make outlet classification artificially easy.

The preprocessing pipeline removes or neutralizes signals such as explicit outlet names, structural bylines and author credits, recurring source or program attributions, URLs, social-media handles, boilerplate, and extraction artifacts.

Exact and near-duplicate detection is performed before dataset splitting.

Publication month and article length are also treated as potential shortcuts. The experimental subset is constructed using common month/length groups across outlets, followed by stratified training, validation, and test splits.

Model selection uses validation data only. The final test set remains sealed until model and representation choices are fixed.

The primary metric is **balance-weighted macro-F1**, allowing the three outlets to contribute equally while retaining all eligible validation and test articles.

## Additional analyses

Beyond the main classification experiments, the project includes:

- TF-IDF feature analysis;
- cross-topic generalization analysis;
- confusion matrices;
- attention-pooling analysis;
- weighted PCA of learned article representations;
- confidence and entropy analysis;
- statistical comparisons between selected models;
- random-initialized Transformer sanity controls.

These analyses provide additional information about model behavior rather than treating individual features, attention weights, or PCA dimensions as complete explanations of model decisions.

## Repository structure

```text
.
├── scripts/
│   ├── 02_collect_urls_gdelt.py
│   ├── 02c_make_disjoint_candidates.py
│   ├── 03_scrape_articles.py
│   ├── 04_build_clean_corpus.py
│   ├── 05_near_dedup.py
│   ├── 06_audit_clean_corpus.py
│   ├── experiment_sampling.py
│   ├── modeling_protocol_final.py
│   ├── prepare_experiment_data.py
│   └── run_experiment_diagnostics.py
│
├── experiment_data/
│   └── experiment metadata, split information, weights, and diagnostics
│
├── outputs/
│   └── retained corpus-building and audit summaries
│
├── modeling_results/
│   ├── classical_tfidf_final/
│   ├── tfidf_neural_svd_final/
│   ├── roberta_transformer_final/
│   └── final_evaluation/
│
├── report/
│   └── final_report.pdf
│
├── run_classical_tfidf_final.py
├── tfidf_neural_svd_final.ipynb
├── roberta_transformer_final.ipynb
├── final_test_evaluation_and_analysis.ipynb
├── experiment_config.json
└── README.md
```

Large trained-model checkpoints, intermediate representations, caches, article-level prediction artifacts, and the complete article corpus are intentionally not included in the public repository.

## Reproducibility

The repository retains the main source code, experiment configuration, fixed split metadata, weighting information, random seeds, model-selection summaries, aggregate results, and analysis outputs used in the project.

The complete article corpus and trained model checkpoints are not distributed in this public version. Reproducing the reported experiments therefore requires rebuilding the dataset and retraining the models.

An exact reconstruction of the original corpus cannot be guaranteed by scraping the source websites again. Web pages may be edited, removed, become unavailable, or change structure over time, and extraction behavior may therefore differ from the original collection process. The repository should consequently be viewed as supporting reproduction of the methodology and experimental pipeline rather than guaranteed reconstruction of the exact original dataset.

Neural-model results may also vary slightly across hardware and software environments even when the same experimental settings and random seeds are used.

## Authors and contributions

Developed as a two-student university project by **Dorjan Štrbac** and **Fran Kufrin** at the **University of Zagreb, Faculty of Electrical Engineering and Computing (FER)**.

Dorjan Štrbac led the project work across corpus construction, experimental design, model experimentation, evaluation, analysis, and preparation of the final report.

Generative AI was used as a coding and development aid during implementation, under iterative human direction and review.

## Report

The complete project report is available here:

**[Linguistic and Narrative Differences Across News Portals on Selected Topics — Final Report](report/final_report.pdf)**
