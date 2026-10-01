# RoBERTa transformer modeling report

Base/FE protocol hash: `5fa51702be769922b580121397830bc7c99f9a614e6aca58ff8f850629124ba1`
Revised FT protocol hash: `3f31db7ebf13f3e6857f609c56aae0cfcf02891b835e5c2dacfe15e18d25c8e3`
Random-init control protocol hash: `9efe780c0f50e591a04e6c9e4212fd5e3b8f1cc52a456fd9d1def93127f966d1`

**TEST STATUS: SEALED — test text was not loaded/tokenized/embedded/evaluated.**

## Tokenization / chunking

- Backbone architecture/tokenizer reference: `FacebookAI/roberta-base`
- Non-overlapping content chunk size: 510 tokens (+ RoBERTa special tokens, max sequence 512).
- Hierarchical article chunk cap: none; all train/validation articles have 100% token coverage.
- Maximum train/validation chunks/article: 10
- Fraction of train/validation articles requiring >1 chunk: 0.911
- `<unk>` fraction: 0.00000000

## Hardware

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Selected inference chunk batch: 32
- Selected FT/random-512 physical batch: 8; effective article batch remains 16.
- Selected hierarchical chunk microbatch: 8.
- Gradient-cache equivalence audit passed: True

## Revised FT protocol

- Stage 1 searches all six LR/WD settings on each of FT-512, FT-MEAN and FT-ATTN for both topics (36 runs).
- Stage 2 confirms the top two configs per topic/architecture over three fresh paired seeds (36 runs; 72 FT runs total).
- Early stopping: patience 5, after the topic-specific minimum fair-sampler coverage requirement; maximum 25 epochs.
- Scheduler: 10% warmup + linear decay across the 25-epoch maximum step schedule.
- If selected FT runs peak at/near epoch 25 or are still improving there, that boundary is reported as a limitation rather than extending training after seeing validation results.
- AMP overflow steps are skipped/recovered by GradScaler and counted; they are excluded from finite gradient-norm summaries.

## Selected FE/FT validation results (confirmation-seed mean ± SD)

- **topicA FE-512**: weighted macro-F1 `0.6961 ± 0.0311`, weighted accuracy `0.6963`, LR `0.003`, WD `0`, mean best epoch `28.6`, n=`5` seeds.
- **topicA FE-MEAN**: weighted macro-F1 `0.7787 ± 0.0280`, weighted accuracy `0.7774`, LR `0.01`, WD `0`, mean best epoch `38.4`, n=`5` seeds.
- **topicA FE-ATTN**: weighted macro-F1 `0.7083 ± 0.0104`, weighted accuracy `0.7071`, LR `0.01`, WD `0`, mean best epoch `21.8`, n=`5` seeds.
- **topicA FT-512**: weighted macro-F1 `0.9399 ± 0.0064`, weighted accuracy `0.9400`, LR `1e-05`, WD `0.01`, mean best epoch `15.3`, n=`3` seeds.
- **topicA FT-MEAN**: weighted macro-F1 `0.9473 ± 0.0196`, weighted accuracy `0.9473`, LR `1e-05`, WD `0`, mean best epoch `12.0`, n=`3` seeds.
- **topicA FT-ATTN**: weighted macro-F1 `0.9539 ± 0.0044`, weighted accuracy `0.9539`, LR `1e-05`, WD `0.01`, mean best epoch `17.0`, n=`3` seeds.
- **topicB FE-512**: weighted macro-F1 `0.7418 ± 0.0122`, weighted accuracy `0.7453`, LR `0.01`, WD `0.01`, mean best epoch `39.8`, n=`5` seeds.
- **topicB FE-MEAN**: weighted macro-F1 `0.7688 ± 0.0139`, weighted accuracy `0.7718`, LR `0.01`, WD `0`, mean best epoch `38.4`, n=`5` seeds.
- **topicB FE-ATTN**: weighted macro-F1 `0.7074 ± 0.0198`, weighted accuracy `0.7126`, LR `0.01`, WD `0.01`, mean best epoch `24.2`, n=`5` seeds.
- **topicB FT-512**: weighted macro-F1 `0.9413 ± 0.0157`, weighted accuracy `0.9410`, LR `2e-05`, WD `0.01`, mean best epoch `17.0`, n=`3` seeds.
- **topicB FT-MEAN**: weighted macro-F1 `0.9675 ± 0.0142`, weighted accuracy `0.9675`, LR `3e-05`, WD `0`, mean best epoch `14.7`, n=`3` seeds.
- **topicB FT-ATTN**: weighted macro-F1 `0.9753 ± 0.0136`, weighted accuracy `0.9753`, LR `3e-05`, WD `0.01`, mean best epoch `17.0`, n=`3` seeds.

## Post-selection random-initialized RoBERTa control

- This control was defined after FE/FT selection was complete and does not alter the FE/FT model-selection protocol.
- Encoder parameters are randomly initialized from the pinned RoBERTa-base configuration; the same pretrained tokenizer/vocabulary is retained.
- One predefined setting is used for every topic/architecture: LR `0.0001`, WD `0.01`, AdamW, `10%` warmup + linear decay, fixed `100` epochs.
- There is no patience/early stopping; every run completes the full epoch budget. The retained checkpoint is the best validation weighted macro-F1 checkpoint (then accuracy, then lower CE).
- All Stage-1-equivalent randomness uses search master seed `20260904`; the only new RNG stream is deterministic random encoder initialization, shared across 512/MEAN/ATTN within each topic.
- Because this is one setting and one seed, it is a reasonable control rather than an exhaustive scratch-training optimum.
- The 100-epoch horizon is not expanded after validation is inspected; a best epoch at/near 100 or a still-improving boundary curve is reported as a limitation.

- **topicA RANDOM-512**: best validation weighted macro-F1 `0.3627`, weighted accuracy `0.4611`, best epoch `2/100`, final-epoch weighted macro-F1 `0.1667`, n=`1`.
- **topicA RANDOM-ATTN**: best validation weighted macro-F1 `0.1678`, weighted accuracy `0.3333`, best epoch `7/100`, final-epoch weighted macro-F1 `0.1667`, n=`1`.
- **topicA RANDOM-MEAN**: best validation weighted macro-F1 `0.7388`, weighted accuracy `0.7362`, best epoch `63/100`, final-epoch weighted macro-F1 `0.7206`, n=`1`.
- **topicB RANDOM-512**: best validation weighted macro-F1 `0.5218`, weighted accuracy `0.6216`, best epoch `40/100`, final-epoch weighted macro-F1 `0.5171`, n=`1`.
- **topicB RANDOM-ATTN**: best validation weighted macro-F1 `0.5424`, weighted accuracy `0.6486`, best epoch `11/100`, final-epoch weighted macro-F1 `0.1667`, n=`1`.
- **topicB RANDOM-MEAN**: best validation weighted macro-F1 `0.5424`, weighted accuracy `0.6486`, best epoch `11/100`, final-epoch weighted macro-F1 `0.1667`, n=`1`.

## Gradient clipping diagnostics

Global gradient-norm clipping threshold: `1.0`. Reported norms are measured before clipping.
- **topicA FE-512**: mean clipped-step fraction `0.812`, mean pre-clip norm `1.977`, maximum pre-clip norm across selected seed runs `7.043`.
- **topicA FE-MEAN**: mean clipped-step fraction `0.811`, mean pre-clip norm `1.969`, maximum pre-clip norm across selected seed runs `6.870`.
- **topicA FE-ATTN**: mean clipped-step fraction `0.798`, mean pre-clip norm `1.926`, maximum pre-clip norm across selected seed runs `6.864`.
- **topicA FT-512**: mean clipped-step fraction `0.756`, mean pre-clip norm `20.341`, maximum pre-clip norm across selected seed runs `230.488`, mean AMP-skipped optimizer steps/run `5.00`.
- **topicA FT-MEAN**: mean clipped-step fraction `0.854`, mean pre-clip norm `17.924`, maximum pre-clip norm across selected seed runs `192.037`, mean AMP-skipped optimizer steps/run `3.67`.
- **topicA FT-ATTN**: mean clipped-step fraction `0.742`, mean pre-clip norm `14.299`, maximum pre-clip norm across selected seed runs `162.987`, mean AMP-skipped optimizer steps/run `3.67`.
- **topicB FE-512**: mean clipped-step fraction `0.780`, mean pre-clip norm `1.880`, maximum pre-clip norm across selected seed runs `8.942`.
- **topicB FE-MEAN**: mean clipped-step fraction `0.775`, mean pre-clip norm `1.864`, maximum pre-clip norm across selected seed runs `9.063`.
- **topicB FE-ATTN**: mean clipped-step fraction `0.782`, mean pre-clip norm `1.872`, maximum pre-clip norm across selected seed runs `9.052`.
- **topicB FT-512**: mean clipped-step fraction `0.641`, mean pre-clip norm `12.647`, maximum pre-clip norm across selected seed runs `182.389`, mean AMP-skipped optimizer steps/run `4.33`.
- **topicB FT-MEAN**: mean clipped-step fraction `0.715`, mean pre-clip norm `10.137`, maximum pre-clip norm across selected seed runs `114.879`, mean AMP-skipped optimizer steps/run `3.33`.
- **topicB FT-ATTN**: mean clipped-step fraction `0.676`, mean pre-clip norm `9.174`, maximum pre-clip norm across selected seed runs `162.455`, mean AMP-skipped optimizer steps/run `3.67`.
- **topicA RANDOM-512**: clipped-step fraction `1.000`, mean pre-clip norm `6.271`, maximum pre-clip norm `318.570`, AMP-skipped optimizer steps `2`.
- **topicA RANDOM-MEAN**: clipped-step fraction `0.419`, mean pre-clip norm `3.768`, maximum pre-clip norm `150.764`, AMP-skipped optimizer steps `2`.
- **topicA RANDOM-ATTN**: clipped-step fraction `1.000`, mean pre-clip norm `5.975`, maximum pre-clip norm `37.103`, AMP-skipped optimizer steps `0`.
- **topicB RANDOM-512**: clipped-step fraction `1.000`, mean pre-clip norm `5.260`, maximum pre-clip norm `54.237`, AMP-skipped optimizer steps `0`.
- **topicB RANDOM-MEAN**: clipped-step fraction `1.000`, mean pre-clip norm `6.327`, maximum pre-clip norm `103.937`, AMP-skipped optimizer steps `0`.
- **topicB RANDOM-ATTN**: clipped-step fraction `1.000`, mean pre-clip norm `6.336`, maximum pre-clip norm `60.174`, AMP-skipped optimizer steps `0`.

## Saved artifacts

- Selected FE heads (five seeds), selected FT checkpoints (three seeds), and all six RANDOM_INIT best-validation checkpoints are under `selected_models/`.
- Validation predictions are retained for FE/FT confirmation models and all random-init controls.
- ATTN confirmation/control runs retain per-article/per-chunk validation attention weights.
- Training histories include per-epoch pre-clipping gradient-norm statistics; FT and random-init runs also retain phase timing/runtime-status diagnostics.
- Random-init control results are summarized separately in `RANDOM_INIT_CONTROL_VALIDATION_SUMMARY.csv` and `random_init_control/all_results.csv`.
- Search/confirmation tables, sampler audit, tokenization audit, hardware preflight, and plots are retained.
- Representation-space PCA/UMAP remains deferred to later post-test analysis.
