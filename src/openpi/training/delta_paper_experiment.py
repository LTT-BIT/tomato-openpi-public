"""Compose personal experiments from public recipe/data types, never from a baseline file."""

from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
import os
from pathlib import Path
import re

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_delta_paper import PaperPi0Config
from openpi.models.pi0_delta_paper import PaperWeightLoader
from openpi.shared import delta_paper as paper
from openpi.training.config import LeRobotDatasetSpec
from openpi.training.delta_paper import PaperSharpaData
from training_configs import common
from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe

PROMPT = "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket."
SOURCE = (
    "sshfs://user@192.168.10.202/data/data-fruits/real/harvest-tomato/data/clean-rosbag/sharpa/10.16_2_data_lerobot"
)


def _validate_labels(owner, trial):
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) for value in (owner, trial)):
        raise ValueError("Paper owner/trial must be directory-safe identifiers")


def build_default_dataset(*, owner, trial):
    _validate_labels(owner, trial)
    root = os.environ.get(
        "PI05_SHARPA_MULTIMODAL_DATASET_ROOT",
        os.environ.get("PI05_SHARPA_DATASET_ROOT", "/mnt/data/users/tairan/workspace/data/10.16_2_data_lerobot"),
    )
    identity = f"tomato_sharpa_1016_2_{owner}_delta_paper_{trial}_v1_30hz"
    return common.Dataset(
        version=identity,
        repo_id="fr3_sharpa/tomato_1016_2_multimodal_rgb_v1",
        root=root,
        assets_dir=os.environ.get("PI05_PAPER_ASSETS_DIR", str(common.repository_root() / "assets")),
        asset_id=identity,
        sources=(SOURCE, root),
        prompt_example=PROMPT,
    )


NO_BLOCK_ONE_ROOT = "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one"
NO_BLOCK_ONE_MEMBERS = (
    "A/1022_121_430_normal_lerobot",
    "A/1022_54_120_normal_lerobot",
    "A/1026_444_513_normal_lerobot",
    "B/1022_130_423_normal_lerobot",
    "B/1022_56_106_normal_lerobot",
    "B/1026_446_510_normal_lerobot",
)


@dataclass(frozen=True)
class PaperDatasetCollection(common.Dataset):
    members: tuple[common.Dataset, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        roots = [str(Path(member.root).expanduser().resolve()) for member in self.members]
        if len(roots) < 2 or len(set(roots)) != len(roots):
            raise ValueError("Paper collection requires distinct source roots")
        if len({member.repo_id for member in self.members}) != len(self.members):
            raise ValueError("Paper collection source identities must be unique")


def build_no_block_one_dataset(*, owner, trial, root=NO_BLOCK_ONE_ROOT):
    """Explicit source list; never move, merge, link or rewrite original data."""
    _validate_labels(owner, trial)
    root = Path(root)
    identity = f"tomato_sharpa_no_block_one_{owner}_delta_paper_{trial}_v1_30hz"
    assets_dir = os.environ.get("PI05_PAPER_ASSETS_DIR", str(common.repository_root() / "assets"))
    members = tuple(
        common.Dataset(
            version=f"no_block_one_{relative.replace('/', '_')}_v1",
            repo_id=f"fr3_sharpa/no_block_one_{relative.replace('/', '_')}",
            root=str(root / relative),
            assets_dir=assets_dir,
            asset_id=identity,
            sources=(str(root / relative),),
            prompt_example=PROMPT,
        )
        for relative in NO_BLOCK_ONE_MEMBERS
    )
    return PaperDatasetCollection(
        version=identity,
        repo_id="fr3_sharpa/no_block_one_v1",
        root=str(root),
        assets_dir=assets_dir,
        asset_id=identity,
        sources=tuple(member.root for member in members),
        prompt_example=PROMPT,
        members=members,
    )


def build_experiment(*, owner, trial, mode, beta, dataset=None):
    _validate_labels(owner, trial)
    if mode not in paper.MODES or not 0 <= beta < 1:
        raise ValueError("Unknown paper mode/beta")
    dataset = dataset or build_default_dataset(owner=owner, trial=trial)
    collection = isinstance(dataset, PaperDatasetCollection)
    label = "no_block_one_" if collection else ""
    base = build_recipe(
        name=f"pi05_{owner}_sharpa_{label}delta_paper_{trial}_fullft_v1",
        exp_name=f"{owner}_sharpa_{label}delta_paper_{trial}_s42_run01",
        embodiment=FR3_SHARPA_MULTIMODAL,
        dataset=dataset,
    )
    asset_dir = Path(dataset.assets_dir) / dataset.asset_id
    digest = paper.sha256(asset_dir / paper.BUNDLE) if (asset_dir / paper.BUNDLE).is_file() else None
    model = PaperPi0Config(
        **{f.name: getattr(base.model, f.name) for f in fields(Pi0Config)},
        paper_asset_dir=str(asset_dir),
        paper_asset_sha256=digest,
        paper_mode=mode,
        paper_beta=beta,
    )
    data = PaperSharpaData(
        repo_id=dataset.repo_id,
        assets=base.data.assets,
        base_config=base.data.base_config,
        lerobot_datasets=tuple(
            LeRobotDatasetSpec(repo_id=member.repo_id, root=member.root, weight=1.0) for member in dataset.members
        )
        if collection
        else base.data.lerobot_datasets,
        action_hz=30,
        video_backend="pyav",
        paper_mode=mode,
        paper_beta=beta,
    )
    return replace(
        base,
        model=model,
        data=data,
        batch_size=32,
        fsdp_devices=8,
        freeze_filter=model.get_freeze_filter(),
        weight_loader=PaperWeightLoader(
            params_path=base.weight_loader.params_path, skip_on_mismatch_regex=base.weight_loader.skip_on_mismatch_regex
        ),
        policy_metadata={
            **base.policy_metadata,
            "owner": owner,
            "paper_trial": trial,
            "paper_algorithm": data.paper_contract(),
            "training_action_semantics": "delta_arms_absolute_hands",
            "statistics_shape": {"state": [58], "actions": [50, 58]},
            "deployment_compatibility": "requires_new_paper_profile; existing_profiles_reject_Hx58_stats",
        },
    )
