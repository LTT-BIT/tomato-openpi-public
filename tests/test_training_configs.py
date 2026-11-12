"""Guard baseline equivalence, real parameter freezing, and owner isolation."""

from dataclasses import replace
import importlib
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import pytest

from openpi.training import config as cfg
from openpi.training.reproducibility import serialize_config
from openpi.training.sampling import SamplingGroup
from scripts.run_training import load_experiment
from scripts.run_training import resolve_experiment
from scripts.run_training import validate_config
from training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1 import build_dataset
from training_configs.recipes.pi05_lora_v1 import build_config as build_lora

BASE = "training_configs.experiments.tomato_wuji.baseline."


@pytest.mark.parametrize("module", ["tomato_fr3_lora_v1", "tomato_fr3_fullft_v1"])
def test_composition_preserves_baseline_training_semantics(module):
    current = load_experiment(BASE + module)
    snapshot = json.loads((Path(__file__).parent / "fixtures/baselines" / f"{module}.json").read_text())
    spec = current.data.lerobot_datasets[0]
    portable = replace(
        current,
        name=snapshot["config_fields"]["name"],
        exp_name=snapshot["config_fields"]["exp_name"],
        data=replace(
            current.data,
            assets=replace(current.data.assets, assets_dir=snapshot["path_placeholders"]["assets_dir"]),
            lerobot_datasets=(replace(spec, root=snapshot["path_placeholders"]["dataset_root"]),),
        ),
    )
    current_values = json.loads(serialize_config(portable))["fields"]
    for key in snapshot["excluded_fields"]:
        current_values.pop(key)
    # Preserve historical delta snapshots; explicitly account for the absolute baseline migration.
    snapshot["config_fields"]["data"]["fields"]["extra_delta_transform"] = False
    # Keep the historical snapshot intact; baseline weights now default to shared server storage.
    snapshot["config_fields"]["weight_loader"]["fields"]["params_path"] = "/mnt/data/public_ckpt/pi05/pi05_base/params"
    old_assets = snapshot["config_fields"]["data"]["fields"]["assets"]["fields"]
    old_assets["asset_id"] = old_assets["asset_id"].replace("_v1_30hz", "_v1_absolute_30hz")
    assert current_values == {
        **snapshot["config_fields"],
        "sampling_groups": [],
        "num_train_steps": 30_000,
        "batch_size": 8 if "lora" in module else 64,
    }
    expected_recipe = "pi05_lora_v1" if "lora" in module else "pi05_fullft_v1"
    assert current.policy_metadata["recipe_version"] == expected_recipe
    assert current.policy_metadata["baseline_version"] == expected_recipe
    expected_source = "fulltask_c0_v2" if "lora" in module else "fulltask_c0_v3_fullft"
    assert current.policy_metadata["source_baseline"] == expected_source
    assert "_v1_" in current.name
    assert "_v1_" in current.exp_name
    validate_config(current)


@pytest.mark.parametrize(
    ("module", "is_lora"),
    [
        ("tomato_fr3_lora_v1", True),
        ("tomato_fr3_fullft_v1", False),
        ("tomato_fr3_sharpa_lora_v1", True),
        ("tomato_fr3_sharpa_fullft_v1", False),
        ("cup_single_franka_lora_v1", True),
        ("cup_single_franka_fullft_v1", False),
    ],
)
def test_actual_model_trainable_scope(module, is_lora):
    task = "cup_grasping" if module.startswith("cup_") else "tomato_sharpa" if "_sharpa_" in module else "tomato_wuji"
    config = load_experiment(f"training_configs.experiments.{task}.baseline.{module}")

    def shapes():
        return nnx.state(config.model.create(jax.random.key(config.model_seed)), nnx.Param)

    params = jax.eval_shape(shapes)
    assert params["action_in_proj"]["kernel"].value.shape[0] == config.model.action_dim
    assert params["action_out_proj"]["kernel"].value.shape[-1] == config.model.action_dim
    trainable = params.filter(config.trainable_filter).flat_state()
    frozen = params.filter(config.freeze_filter).flat_state()
    names = [str(path) for path in trainable]
    assert any("img" in path for path in names)
    assert any("llm" in path for path in names)
    assert any("action_in_proj" in path for path in names)
    assert any("action_out_proj" in path for path in names)
    if is_lora:
        assert frozen
        assert any("llm" in str(path) for path in frozen)
        assert any("lora" in path for path in names)
        assert not any("lora" in str(path) for path in frozen)
    else:
        assert not frozen
        assert len(trainable) == len(params.flat_state())
        assert not any("lora" in path for path in names)


def test_fullft_selected_data_and_reproducibility_contract(monkeypatch):
    monkeypatch.setenv("PI05_SOURCE_DATA_DIR", "/test-data/harvest-tomato")
    config = load_experiment(BASE + "tomato_fr3_fullft_v1")
    assert config.policy_metadata["source_datasets"] == [
        "/test-data/harvest-tomato/0925/A",
        "/test-data/harvest-tomato/0925/B",
        "/test-data/harvest-tomato/1002_AB_190/full",
    ]
    assert config.num_train_steps == 30_000
    assert (config.seed, config.model_seed, config.data_seed, config.augmentation_seed) == (42, 42, 42001, 42002)
    assert config.sampling_mode == "epoch_shuffle"
    assert config.data.base_config.prompt_from_task
    assert config.data.action_hz == 30
    assert config.model.action_dim == 54
    assert config.model.action_horizon == 50
    assert not config.resume
    assert not config.overwrite
    assert config.wandb_enabled


def test_personal_override_does_not_mutate_baseline(monkeypatch, make_personal_experiment):
    baseline = load_experiment(BASE + "tomato_fr3_lora_v1")
    monkeypatch.setenv("PI05_DATASET_ROOT", "/alice/data/tomato_v1")
    monkeypatch.setenv("PI05_ASSETS_DIR", "/alice/assets")
    personal = load_experiment(make_personal_experiment(BASE + "tomato_fr3_lora_v1", peak_lr=1e-5))
    assert personal.data.lerobot_datasets[0].root == "/alice/data/tomato_v1"
    assert personal.data.assets.assets_dir == "/alice/assets"
    assert baseline.data.lerobot_datasets[0].root != personal.data.lerobot_datasets[0].root
    assert load_experiment(BASE + "tomato_fr3_lora_v1").data == baseline.data
    assert personal.lr_schedule.peak_lr == 1e-5
    assert baseline.lr_schedule.peak_lr == 2.5e-5
    assert personal.exp_name != baseline.exp_name
    assert personal.checkpoint_dir != baseline.checkpoint_dir
    overridden = resolve_experiment(BASE + "tomato_fr3_lora_v1", ["--batch-size", "4", "--exp-name", "alice_run01"])
    assert overridden.batch_size == 4
    assert baseline.batch_size == 8


def test_personal_sampling_groups_survive_cli(monkeypatch, make_personal_experiment):
    module_name = make_personal_experiment(BASE + "tomato_fr3_lora_v1")
    module = importlib.import_module(module_name)
    config = replace(
        module.build_config(),
        sampling_mode="weighted_replacement",
        sampling_groups=(
            SamplingGroup(name="A", episodes=(0, 2), weight=7),
            SamplingGroup(name="B", episodes=(1, 3), weight=3),
        ),
    )
    monkeypatch.setattr(module, "build_config", lambda: config)
    resolved = resolve_experiment(module_name, ["--exp-name", "weighted_trial", "--batch-size", "4"])
    assert resolved.sampling_mode == "weighted_replacement"
    assert resolved.sampling_groups == config.sampling_groups
    assert resolved.exp_name == "weighted_trial"
    assert resolved.batch_size == 4
    assert load_experiment(BASE + "tomato_fr3_lora_v1").sampling_groups == ()


def test_recipe_accepts_another_embodiment_without_fr3_dimension():
    class OtherEmbodiment:
        version = "test_other_v1"
        action_dim = 14
        raw_state_dims = (14,)
        action_hz = 20.0
        camera_keys = ()

        def data_config(self, dataset):
            return cfg.FakeDataConfig(repo_id=dataset.repo_id)

    config = build_lora(name="other", exp_name="other_s42", embodiment=OtherEmbodiment(), dataset=build_dataset())
    assert config.model.action_dim == 14
    assert config.policy_metadata["embodiment_version"] == "test_other_v1"
    assert isinstance(config.data, cfg.FakeDataConfig)


def test_cli_cannot_change_model_without_regenerating_freeze_filter():
    with pytest.raises(ValueError, match="versioned recipe"):
        resolve_experiment(BASE + "tomato_fr3_lora_v1", ["--model.paligemma-variant", "gemma_2b"])


@pytest.mark.parametrize("change", [{"exp_name": "../other"}, {"wandb_enabled": False}, {"overwrite": True}])
def test_reject_unsafe_experiment_settings(change):
    with pytest.raises(ValueError, match="identifier|W&B|overwriting"):
        validate_config(replace(load_experiment(BASE + "tomato_fr3_lora_v1"), **change))


@pytest.mark.parametrize("field", ["log_interval", "save_interval", "keep_period"])
@pytest.mark.parametrize("value", [0, -1, 0.5, True])
def test_reject_invalid_training_intervals_before_startup(field, value):
    with pytest.raises(ValueError, match=f"{field} must be"):
        replace(load_experiment(BASE + "tomato_fr3_lora_v1"), **{field: value})


def test_cli_rejects_zero_save_interval_without_reading_data():
    from scripts.run_training import main

    with pytest.raises(ValueError, match="save_interval must be"):
        main([BASE + "tomato_fr3_lora_v1", "--save-interval", "0", "--dry-run"])


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_horizon_is_rejected_before_model_or_data_loading(value):
    config = load_experiment(BASE + "tomato_fr3_lora_v1")
    with pytest.raises(ValueError, match="action_horizon must be"):
        replace(config, model=replace(config.model, action_horizon=value))
