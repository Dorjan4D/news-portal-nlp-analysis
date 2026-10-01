#!/usr/bin/env bash
set -euo pipefail

# Run from the PROJECT ROOT (the directory containing data/, scripts/, outputs/).
# Requires data/articles_raw.jsonl.

python scripts/04_build_clean_corpus.py \
  --inp data/articles_raw.jsonl \
  --out-dir data \
  --report-dir outputs/corpus_build \
  --start 2024-07-28 \
  --end 2026-01-27 \
  --min-words 150

python scripts/05_near_dedup.py \
  --model-in data/corpus_clean_candidate.jsonl \
  --audit-in data/corpus_clean_audit_candidate.jsonl \
  --model-out data/corpus_clean_dedup_candidate.jsonl \
  --audit-out data/corpus_clean_dedup_audit_candidate.jsonl \
  --report-dir outputs/near_dedup \
  --report-threshold 0.94 \
  --remove-threshold 0.985

python scripts/06_audit_clean_corpus.py \
  --model data/corpus_clean_dedup_candidate.jsonl \
  --audit data/corpus_clean_dedup_audit_candidate.jsonl \
  --author-list outputs/corpus_build/global_author_names.txt \
  --out-dir outputs/corpus_audit \
  --sample-per-cell 50

echo
echo "Corpus stage finished."
echo "DO NOT create train/val/test yet. Review:"
echo "  outputs/corpus_build/build_report.json"
echo "  outputs/near_dedup/near_dedup_summary.json"
echo "  outputs/near_dedup/near_duplicate_pairs_for_review.csv"
echo "  outputs/near_dedup/canonical_bbc_url_duplicate_groups.csv"
echo "  outputs/corpus_audit/audit_summary.json"
echo "  outputs/corpus_audit/manual_quality_sample.csv"
echo "  outputs/corpus_audit/manual_quality_sample_secondary.csv"
echo "  outputs/corpus_audit/bbc_com_heading_policy_violations.csv"
echo "  outputs/corpus_audit/aljazeera_leading_heading_policy_violations.csv"
echo "  outputs/corpus_audit/body_start_integrity_violations.csv"
echo "  outputs/corpus_audit/residual_leakage_hits.csv"
echo "  outputs/corpus_audit/tfidf_top_features_topicA.csv"
echo "  outputs/corpus_audit/tfidf_top_features_topicB.csv"
