#!/usr/bin/env bash
# Shared local/GitLab/GitHub checks; use the already installed locked environment.
set -euo pipefail
REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_DIR"
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GIT_LFS_SKIP_SMUDGE=1
export WANDB_MODE=offline
export UV_CACHE_DIR="$REPOSITORY_DIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$REPOSITORY_DIR/.venv"
export PYTHONPATH="$REPOSITORY_DIR/src:$REPOSITORY_DIR/packages/openpi-client/src"
mkdir -p .cache/ci

uv lock --check --offline
CHECK_PATHS=(
  training_configs tests
  scripts/create_delta_paper_experiments.py
  src/openpi/models/pi0.py src/openpi/models/pi0_config.py src/openpi/models/pi0_delta_paper.py
  src/openpi/models/pi0_test.py src/openpi/models/gemma.py
  src/openpi/shared/delta_paper.py src/openpi/training/delta_paper.py
  src/openpi/training/delta_paper_experiment.py src/openpi/training/delta_paper_stats.py
  src/openpi/training/delta_paper_ui.py
  scripts/training_ui.py scripts/training_ui_config.py
  scripts/run_training.py scripts/compute_experiment_norm.py scripts/train.py scripts/train_reproducibility_test.py
  src/openpi/training/norm_provenance.py examples/fr3_wuji/compute_norm_stats.py
  src/openpi/shared/local_storage.py src/openpi/training/tracking.py
  src/openpi/policies/fr3_sharpa_policy.py src/openpi/policies/fr3_sharpa_protocol.py
  src/openpi/policies/single_franka_policy.py src/openpi/policies/lemon_policy.py
  src/openpi/training/task_view.py src/openpi/training/task_view_norm.py
  src/openpi/training/config.py src/openpi/training/checkpoints.py
  src/openpi/training/sampling.py src/openpi/training/sampling_test.py
  src/openpi/training/data_loader.py src/openpi/training/reproducibility.py
  src/openpi/training/reproducibility_test.py
  src/openpi/transforms_test.py src/openpi/training/data_loader_test.py
  examples/fr3_wuji/verify_dataset.py
)
bash scripts/python.sh -m ruff check "${CHECK_PATHS[@]}"
bash scripts/python.sh -m ruff format --check "${CHECK_PATHS[@]}"
bash -n scripts/setup_training_env.sh scripts/check_training.sh scripts/python.sh scripts/training_runtime.sh

for experiment in \
  training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1 \
  training_configs.experiments.tomato_wuji.baseline.tomato_fr3_fullft_v1 \
  training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_lora_v1 \
  training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_fullft_v1 \
  training_configs.experiments.cup_grasping.baseline.cup_single_franka_lora_v1 \
  training_configs.experiments.cup_grasping.baseline.cup_single_franka_fullft_v1 \
  training_configs.experiments.lemon_scooping.baseline.lemon_ur_o30_lora_v1 \
  training_configs.experiments.lemon_scooping.baseline.lemon_ur_o30_fullft_v1; do
  bash scripts/python.sh -m scripts.run_training "$experiment" --dry-run
done

bash scripts/python.sh -m pytest tests \
  src/openpi/models/pi0_test.py \
  src/openpi/training/sampling_test.py src/openpi/training/reproducibility_test.py \
  src/openpi/training/weight_loaders_test.py src/openpi/training/data_loader_test.py \
  src/openpi/policies/fr3_wuji_policy_test.py src/openpi/transforms_test.py \
  src/openpi/shared/normalize_test.py scripts/train_reproducibility_test.py \
  -q -m 'not manual' --junitxml=.cache/ci/junit.xml
uv build --wheel --out-dir .cache/ci/wheels
