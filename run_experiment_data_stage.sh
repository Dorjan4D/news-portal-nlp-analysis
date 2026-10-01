#!/usr/bin/env bash
set -euo pipefail

# Run from the project root, where experiment_config.json and frozen_data/ live.
# Any arguments (for example --force) are passed to the deterministic preparation step.
python scripts/prepare_experiment_data.py --config experiment_config.json "$@"
python scripts/run_experiment_diagnostics.py --config experiment_config.json
