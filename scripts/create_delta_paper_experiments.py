"""Create ignored personal Sharpa paper configs; never register or edit a baseline."""

import argparse
import re

from openpi.shared.local_storage import new_artifact_directory
from training_configs import common

TRIALS = (
    ("t0", "global_quantile", 0.0),
    ("t1", "timestamp_zscore_arms", 0.0),
    ("t2", "global_quantile", 0.5),
    ("t3", "timestamp_zscore_arms", 0.5),
    ("tg", "global_zscore_arms", 0.0),
)


def source(owner, trial, mode, beta, dataset="single"):
    if dataset not in ("single", "no-block-one"):
        raise ValueError("Unknown paper dataset preset")
    builder = "build_no_block_one_dataset" if dataset == "no-block-one" else "build_default_dataset"
    return (
        f'"""Local Sharpa fullft {trial}: {mode}, noise beta={beta}; not a public baseline."""\n\n'
        f"from openpi.training.delta_paper_experiment import {builder}\n"
        "from openpi.training.delta_paper_experiment import build_experiment\n\n\n"
        "def build_dataset():\n"
        f'    return {builder}(owner="{owner}", trial="{trial}")\n\n\n'
        "def build_config():\n"
        f'    return build_experiment(owner="{owner}", trial="{trial}", mode="{mode}", beta={beta}, dataset=build_dataset())\n'
    )


def create(owner, dataset="single"):
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", owner) is None:
        raise ValueError("Owner must be an English directory-safe identifier")
    if dataset not in ("single", "no-block-one"):
        raise ValueError("Unknown paper dataset preset")
    prefix = "no_block_one_" if dataset == "no-block-one" else ""
    directory = common.repository_root() / "training_configs/experiments/tomato_sharpa/template" / owner
    # Validate before creating any artifact directories.
    source(owner, "t3", "timestamp_zscore_arms", 0.5, dataset)
    with new_artifact_directory(directory) as staging:
        for trial, mode, beta in TRIALS:
            (staging / f"delta_paper_{prefix}{trial}.py").write_text(source(owner, trial, mode, beta, dataset))

    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset", choices=("single", "no-block-one"), default="single")
    args = parser.parse_args()
    print(
        f"Created 5 local configs in {create(args.owner, args.dataset)}; Git ignores them. Existing directories are never overwritten."
    )


if __name__ == "__main__":
    main()
