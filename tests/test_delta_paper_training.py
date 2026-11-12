"""Paper algorithms are explicit, versioned and cannot alter public baselines."""
# ruff: noqa: PLC0415, SLF001

from dataclasses import fields
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import datasets
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_delta_paper import PaperPi0
from openpi.models.pi0_delta_paper import PaperPi0Config
from openpi.shared import delta_paper as paper
from openpi.shared import normalize
from openpi.training import config as cfg
from openpi.training import data_loader
from openpi.training.delta_paper_experiment import build_experiment
from openpi.training.delta_paper_stats import compute
from scripts import run_training
from training_configs import common


def numeric_prompt(data):
    return {
        **{key: value for key, value in data.items() if key != "prompt"},
        "tokenized_prompt": np.ones(4, np.int32),
        "tokenized_prompt_mask": np.ones(4, bool),
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    horizon = getattr(request, "param", 3)
    total_frames = 2 * horizon + 6
    root = tmp_path / "data"
    for name in ("meta", "data", "videos"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    monkeypatch.setenv("PI05_SHARPA_MULTIMODAL_DATASET_ROOT", str(root))
    monkeypatch.setenv("PI05_PAPER_ASSETS_DIR", str(tmp_path / "assets"))
    config = build_experiment(owner="test", trial="fixture", mode="global_quantile", beta=0)
    episodes, records = [], []
    all_ids = {"episode_index": [], "frame_index": []}
    for episode, length in enumerate((horizon + 4, horizon + 2)):
        state = np.zeros((length, 116), np.float32)
        action = np.zeros((length, 58), np.float32)
        state[:, 0] = np.arange(length) * 0.01 + episode * 0.02
        action[:, 0] = np.arange(length) * 0.02 + episode * 0.01
        action[:, 14:36] = 0.1  # Raw left hand stays absolute after canonical reorder.
        records.append((state, action))
        episodes.append({"episode_index": episode, "length": length})
        all_ids["episode_index"].extend([episode] * length)
        all_ids["frame_index"].extend(range(length))
        pq.write_table(
            pa.table(
                {
                    "observation.state": state.tolist(),
                    "action": action.tolist(),
                    "frame_index": list(range(length)),
                    "episode_index": [episode] * length,
                }
            ),
            root / f"data/episode_{episode:06d}.parquet",
        )
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_frames": total_frames,
        "total_episodes": 2,
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.state": {"shape": [116], "names": config.policy_metadata["source_state_joint_names"]},
            "action": {"shape": [58], "names": config.policy_metadata["source_action_joint_names"]},
            **{key: {"shape": [1, 1, 3]} for key in config.policy_metadata["camera_keys"]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Pick the tomato."}\n')
    metadata = SimpleNamespace(
        fps=30,
        tasks={0: "Pick the tomato."},
        features=info["features"],
        camera_keys=config.policy_metadata["camera_keys"],
    )

    class Source:
        def __init__(self, repo_id, *, root, delta_timestamps, **kwargs):
            self.root, self.meta = Path(root), metadata
            self.horizon = len(delta_timestamps["action"])
            self.hf_dataset = datasets.Dataset.from_dict(all_ids)

        def __len__(self):
            return total_frames

        def __getitem__(self, index):
            episode = all_ids["episode_index"][index]
            frame = all_ids["frame_index"][index]
            state, action = records[episode]
            return {
                "observation.state": state[frame].copy(),
                "action": action[np.minimum(frame + np.arange(self.horizon), len(action) - 1)].copy(),
                "episode_index": episode,
                "frame_index": frame,
                "task_index": 0,
                **{key: np.zeros((1, 1, 3), np.uint8) for key in metadata.camera_keys},
            }

    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *a, **kw: metadata)
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", Source)
    monkeypatch.setattr(cfg.ModelTransformFactory, "__call__", lambda *args: transforms.Group(inputs=(numeric_prompt,)))

    def build(trial="fixture", mode="global_quantile", beta=0):
        config = build_experiment(owner="test", trial=trial, mode=mode, beta=beta)
        return replace(config, model=replace(config.model, action_horizon=horizon), fsdp_devices=1, batch_size=2)

    return build, root, records


@pytest.mark.parametrize(
    ("mode", "beta"),
    [
        ("global_quantile", 0),
        ("timestamp_zscore_arms", 0),
        ("global_quantile", 0.5),
        ("timestamp_zscore_arms", 0.5),
        ("global_zscore_arms", 0),
    ],
)
def test_complete_stats_transform_roundtrip_and_bound_assets(prepared, mode, beta, tmp_path):
    build, root, records = prepared
    config = build(mode=mode, beta=beta)
    compute(config, batch_size=2)
    config = build(mode=mode, beta=beta)  # New launch pins the newly published bundle digest.
    report = run_training.preflight(config)
    assert report["training_windows"] == 8
    assert report["norm_provenance"] == "verified"
    directory = config.data.asset_directory(config.assets_dirs)
    stats = normalize.load(directory)
    assert stats["actions"].q01.shape == (3, 58)
    metadata, arrays = paper.load_bundle(directory, mode=mode, beta=beta, horizon=3, std_floor=1e-4)
    assert metadata["excluded_tail_starts"] == 4
    assert arrays["noise_cholesky"].shape == ((174, 174) if beta else (0, 0))
    data = config.data.create(config.assets_dirs, config.model)
    source = data_loader.create_torch_dataset(data, 3, config.model, sampling_mode="epoch_shuffle")
    assert len(source) == 8
    assert source._dataset.source_total_frames == 12
    np.testing.assert_array_equal(source._dataset.hf_dataset["frame_index"], [0, 1, 2, 3, 4, 0, 1, 2])
    repack = transforms.compose(data.repack_transforms.inputs)
    vector = transforms.compose(data.data_transforms.inputs)
    start = repack(source[1])
    raw = start["actions"]
    absolute = np.concatenate([raw[:, :7], raw[:, 14:36], raw[:, 7:14], raw[:, 36:58]], axis=-1)
    mixed = vector(start)
    encoded = transforms.Normalize(data.norm_stats, use_quantiles=True)(mixed)
    baseline_hands = encoded["actions"][:, np.r_[7:29, 36:58]].copy()
    encoded = transforms.compose(data.model_transforms.inputs)(encoded)
    np.testing.assert_array_equal(encoded["actions"][:, np.r_[7:29, 36:58]], baseline_hands)
    output = transforms.compose(data.model_transforms.outputs)(encoded)
    output = transforms.Unnormalize(data.norm_stats, use_quantiles=True)(output)
    output = transforms.compose(data.data_transforms.outputs)(output)
    np.testing.assert_allclose(output["actions"], absolute, atol=1e-6)
    destination = tmp_path / "checkpoint-assets"
    normalize.save(destination, data.norm_stats)
    data.save_extra_assets(destination)
    assert paper.sha256(destination / paper.BUNDLE) == config.model.paper_asset_sha256
    with pytest.raises(FileExistsError, match="already exists"):
        compute(config, batch_size=2)
    wrong = replace(
        config,
        data=replace(
            config.data, paper_mode="global_zscore_arms" if mode != "global_zscore_arms" else "global_quantile"
        ),
    )
    with pytest.raises(ValueError, match="mode differs"):
        run_training.preflight(wrong)
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Another task."}\n')
    with pytest.raises(ValueError, match="data or vector transforms differ"):
        run_training.preflight(config)


def test_timestamp_statistics_use_same_origin_and_no_padded_tail(prepared):
    build, _, _ = prepared
    config = build(mode="timestamp_zscore_arms")
    compute(config, batch_size=3)
    _, arrays = paper.load_bundle(
        config.data.asset_directory(config.assets_dirs), mode="timestamp_zscore_arms", beta=0, horizon=3, std_floor=1e-4
    )
    # Eight starts across two episodes; every horizon uses the SAME start state.
    starts = np.r_[np.arange(5) * 0.01, np.arange(3) * 0.01 - 0.01]
    np.testing.assert_allclose(arrays["arm_mean"][:, 0], starts.mean() + np.arange(3) * 0.02, atol=1e-8)
    np.testing.assert_allclose(arrays["arm_std"][:, 0], starts.std(), atol=1e-8)


def test_atomic_failure_and_refusal_of_legacy_stats(prepared, monkeypatch):
    build, _, _ = prepared
    config = build()
    directory = config.data.asset_directory(config.assets_dirs)
    original = Path.write_text

    def fail_manifest(path, *args, **kwargs):
        if path.name == paper.MANIFEST:
            raise OSError("interrupted paper publication")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest)
    with pytest.raises(OSError, match="interrupted"):
        compute(config, batch_size=3)
    assert not directory.exists()
    assert not list(directory.parent.glob(f".{directory.name}.building-*"))
    monkeypatch.setattr(Path, "write_text", original)
    with pytest.raises(FileNotFoundError):
        config.data.create(config.assets_dirs, config.model)


def test_noise_correlation_beta_zero_and_empirical_covariance():
    rng = np.random.default_rng(42)
    training = rng.normal(size=(2000, 6))
    training[:, 1] = training[:, 0] * 0.7 + training[:, 1] * 0.3
    training[:, 5] = 0
    chol = paper.correlation_factor(training, 0.5)
    np.testing.assert_allclose(np.diag(chol @ chol.T), 1, atol=1e-6)
    np.testing.assert_array_equal((chol @ chol.T)[5, :5], 0)

    class Toy(nnx.Module):
        noise = PaperPi0.noise

        def __init__(self, beta):
            self.paper_beta, self.action_horizon, self.action_dim = beta, 2, 3
            self.paper_noise_cholesky = nnx.Variable(jnp.asarray(chol))

    key = jax.random.key(7)
    np.testing.assert_array_equal(Toy(0).noise(key, (100, 2, 3)), jax.random.normal(key, (100, 2, 3)))
    samples = np.asarray(jax.jit(Toy(0.5).noise, static_argnums=1)(key, (50000, 2, 3))).reshape(-1, 6)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), chol @ chol.T, atol=0.02)
    with pytest.raises(ValueError, match="finite"):
        paper.correlation_factor([[float("nan"), 0], [0, 1]], 0.5)


def shrink_network(monkeypatch):
    from openpi.models import gemma
    from openpi.models import siglip

    original = siglip.Module
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda variant: gemma.Config(width=32, depth=1, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16),
    )
    monkeypatch.setattr(
        siglip,
        "Module",
        lambda num_classes, **kwargs: original(
            num_classes,
            width=32,
            depth=1,
            mlp_dim=64,
            num_heads=2,
            patch_size=(112, 112),
            pool_type="none",
            scan=False,
            dtype_mm="float32",
        ),
    )


def test_real_small_gemma_siglip_gradient_update_and_checkpoint_assets(prepared, monkeypatch, tmp_path):
    from openpi.models import model as model_lib
    from openpi.training import checkpoints
    from openpi.training import sharding
    from scripts import train

    build, _, _ = prepared
    config = build(mode="timestamp_zscore_arms", beta=0.5)
    compute(config, batch_size=3)
    config = build(mode="timestamp_zscore_arms", beta=0.5)
    config = replace(config, model=replace(config.model, dtype="float32", max_token_len=4))
    shrink_network(monkeypatch)
    model = config.model.create(jax.random.key(0))
    state = nnx.state(model)
    assert "paper_noise_cholesky" in state
    assert "paper_noise_cholesky" not in state.filter(config.trainable_filter)
    restored_base = {k: v for k, v in state.to_pure_dict().items() if k != "paper_noise_cholesky"}
    original_restore = model_lib.restore_params
    monkeypatch.setattr(model_lib, "restore_params", lambda *a, **kw: restored_base)
    monkeypatch.setattr("openpi.training.weight_loaders.download.maybe_download", lambda path: path)
    mesh = sharding.make_mesh(1)
    initialized, _ = train.init_train_state(config, jax.random.key(42), mesh, resume=False)
    obs, actions = config.model.fake_obs(2), config.model.fake_act(2)
    updated, metrics = jax.jit(lambda state: train.train_step(config, jax.random.key(42), state, (obs, actions)))(
        initialized
    )
    assert int(updated.step) == 1
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["grad_norm"])
    np.testing.assert_array_equal(
        updated.params["paper_noise_cholesky"].value, initialized.params["paper_noise_cholesky"].value
    )
    assert not np.array_equal(
        updated.params["action_out_proj"]["kernel"].value, initialized.params["action_out_proj"]["kernel"].value
    )
    data = config.data.create(config.assets_dirs, config.model)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoint", keep_period=10, overwrite=False, resume=False
    )
    checkpoints.save_state(manager, updated, SimpleNamespace(data_config=lambda: data), step=1)
    manager.wait_until_finished()
    asset = tmp_path / "checkpoint/1/assets" / data.asset_id
    assert (asset / paper.BUNDLE).is_file()
    assert paper.sha256(asset / paper.BUNDLE) == config.model.paper_asset_sha256
    manager.close()
    params = original_restore(tmp_path / "checkpoint/1/params", restore_type=np.ndarray)
    restored = config.model.load(params)
    trained = nnx.merge(updated.model_def, updated.params)
    np.testing.assert_allclose(
        restored.sample_actions(jax.random.key(3), obs, num_steps=2),
        trained.sample_actions(jax.random.key(3), obs, num_steps=2),
        atol=1e-6,
    )


def test_paper_beta_zero_real_model_matches_base_loss_and_inference(prepared, monkeypatch):
    from openpi.models.pi0 import Pi0

    build, _, _ = prepared
    config = build()
    compute(config, batch_size=3)
    config = build()
    config = replace(config, model=replace(config.model, dtype="float32", max_token_len=4))
    shrink_network(monkeypatch)
    new = config.model.create(jax.random.key(9))
    original = Pi0(
        Pi0Config(**{f.name: getattr(config.model, f.name) for f in fields(Pi0Config)}),
        rngs=nnx.Rngs(jax.random.key(9)),
    )
    obs, action = config.model.fake_obs(1), config.model.fake_act(1)
    np.testing.assert_array_equal(
        new.compute_loss(jax.random.key(5), obs, action), original.compute_loss(jax.random.key(5), obs, action)
    )
    np.testing.assert_array_equal(
        new.sample_actions(jax.random.key(5), obs, num_steps=2),
        original.sample_actions(jax.random.key(5), obs, num_steps=2),
    )


def test_model_rejects_pytorch_and_missing_pinned_assets():
    config = PaperPi0Config(pi05=True, action_dim=58)
    with pytest.raises(ValueError, match="reload configuration"):
        config.create(jax.random.key(0))
    with pytest.raises(ValueError, match="JAX-only"):
        config.load_pytorch(None, "unused")


@pytest.mark.parametrize("prepared", [50], indirect=True)
def test_full_50x58_matrix_and_loader_sampling_audit(prepared, tmp_path):
    build, _, _ = prepared
    config = build(mode="timestamp_zscore_arms", beta=0.5)
    compute(config, batch_size=2)
    config = build(mode="timestamp_zscore_arms", beta=0.5)
    config = replace(config, num_workers=0, num_train_steps=2)
    metadata, arrays = paper.load_bundle(
        config.data.asset_directory(config.assets_dirs),
        mode="timestamp_zscore_arms",
        beta=0.5,
        horizon=50,
        std_floor=1e-4,
    )
    assert arrays["noise_cholesky"].shape == (2900, 2900)
    assert arrays["arm_mean"].shape == (50, 14)
    assert metadata["training_windows"] == 8
    loader = data_loader.create_data_loader(config, shuffle=True, num_batches=2)
    audit = loader.prepare_reproducibility(config, resuming=False)
    assert audit
    obs, actions = next(iter(loader))
    assert actions.shape == (2, 50, 58)
    assert obs.state.shape == (2, 58)
    assert np.isfinite(actions).all()
    # Legacy preflight rejects this new asset family before a profile can silently misread it.
    baseline = run_training.load_experiment(
        "training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_fullft_v1"
    )
    baseline = replace(baseline, data=replace(baseline.data, assets=config.data.assets))
    with pytest.raises(ValueError, match="Invalid normalization actions"):
        run_training.check_norm(baseline, {})


@pytest.mark.parametrize("failure", ["bundle", "audit", "checkpoint_manifest"])
def test_corrupted_bundle_provenance_and_midrun_asset_changes_are_rejected(prepared, tmp_path, failure):
    build, _, _ = prepared
    config = build(mode="timestamp_zscore_arms", beta=0.5)
    compute(config, batch_size=3)
    config = build(mode="timestamp_zscore_arms", beta=0.5)
    directory = config.data.asset_directory(config.assets_dirs)
    view = config.data.create(config.assets_dirs, config.model)
    if failure == "bundle":
        with (directory / paper.BUNDLE).open("ab") as stream:
            stream.write(b"corruption")
        with pytest.raises(ValueError, match="digest mismatch"):
            run_training.preflight(config)
    elif failure == "audit":
        path = directory / "norm_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["contract"]["repo_id"] = "another-dataset"
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="provenance differ"):
            run_training.preflight(config)
    else:
        destination = tmp_path / "assets-copy"
        normalize.save(destination, view.norm_stats)
        path = directory / paper.MANIFEST
        metadata = json.loads(path.read_text())
        metadata["config_name"] = "modified-during-training"
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="changed during training"):
            view.save_extra_assets(destination)


def test_personal_generator_only_creates_ignored_task_owner_directory(tmp_path, monkeypatch):
    from scripts.create_delta_paper_experiments import create

    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    target = create("alice")
    assert target == tmp_path / "training_configs/experiments/tomato_sharpa/template/alice"
    assert len(list(target.glob("*.py"))) == 5
    before = {path.name: path.read_bytes() for path in target.glob("*.py")}
    with pytest.raises(FileExistsError):
        create("alice")
    assert before == {path.name: path.read_bytes() for path in target.glob("*.py")}
    with pytest.raises(ValueError, match="directory-safe"):
        create("../baseline")


def test_personal_dataset_definition_controls_all_model_and_statistics_paths(tmp_path, monkeypatch):
    from openpi.training.delta_paper_experiment import build_default_dataset

    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    dataset = replace(
        build_default_dataset(owner="alice", trial="t0"),
        version="new_training_batch_v2",
        repo_id="sharpa/new_training_batch_v2",
        root=str(tmp_path / "new-data"),
        asset_id="alice_new_batch_delta_t0_v2",
        sources=("prepared-training-split-v2",),
        prompt_example="Pick the tomato.",
    )
    config = build_experiment(owner="alice", trial="t0", mode="global_quantile", beta=0, dataset=dataset)
    assert config.model.paper_asset_dir == str(Path(dataset.assets_dir) / dataset.asset_id)
    assert config.data.repo_id == dataset.repo_id
    assert config.data.lerobot_datasets[0].root == dataset.root
    assert config.policy_metadata["source_datasets"] == list(dataset.sources)
    assert config.policy_metadata["dataset_version"] == dataset.version


@pytest.mark.parametrize("prepared", [3, 50], indirect=True)
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_six_original_directories_joint_stats_loader_and_audit(prepared, tmp_path, monkeypatch, beta):
    import shutil

    from openpi.training.delta_paper_experiment import NO_BLOCK_ONE_MEMBERS
    from openpi.training.delta_paper_experiment import build_no_block_one_dataset
    from openpi.training.reproducibility import _source_identity

    build, original, _ = prepared
    horizon = build().model.action_horizon
    parent = tmp_path / "no-block-one"
    all_windows = []
    fingerprints = {}
    for number, relative in enumerate(NO_BLOCK_ONE_MEMBERS):
        root = parent / relative
        shutil.copytree(original, root)
        info = json.loads((root / "meta/info.json").read_text())
        for key, value in info["features"].items():
            value["dtype"] = "video" if "images" in key else "float32"
        length0 = horizon + 4 + number
        episodes = [{"episode_index": 0, "length": length0}, {"episode_index": 1, "length": horizon + 2}]
        info["total_frames"] = sum(row["length"] for row in episodes)
        (root / "meta/info.json").write_text(json.dumps(info))
        (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
        for row in episodes:
            length, ep = row["length"], row["episode_index"]
            state = np.zeros((length, 116), np.float32)
            action = np.zeros((length, 58), np.float32)
            state[:, 0] = np.arange(length) * 0.01
            action[:, 0] = number * 0.1 + ep * 0.01 + np.arange(length) * 0.02
            action[:, 14:36] = 0.1
            pq.write_table(
                pa.table(
                    {
                        "observation.state": state.tolist(),
                        "action": action.tolist(),
                        "frame_index": list(range(length)),
                        "episode_index": [ep] * length,
                    }
                ),
                root / f"data/episode_{ep:06d}.parquet",
            )
            for start in range(length - horizon + 1):
                chunk = np.zeros((horizon, 58), np.float32)
                chunk[:, 0] = action[start : start + horizon, 0] - state[start, 0]
                all_windows.append(chunk)
        for path in root.rglob("*"):
            if path.is_file():
                fingerprints[path] = paper.sha256(path)

    def metadata(repo_id, *, root):
        info = json.loads((Path(root) / "meta/info.json").read_text())
        return SimpleNamespace(
            fps=30,
            tasks={0: "Pick the tomato."},
            features=info["features"],
            camera_keys=build().policy_metadata["camera_keys"],
        )

    class Source:
        def __init__(self, repo_id, *, root, delta_timestamps, **kwargs):
            self.root = Path(root)
            self.meta = metadata(repo_id, root=root)
            self.horizon = len(delta_timestamps["action"])
            self.tables = [pq.read_table(path).to_pydict() for path in sorted((self.root / "data").glob("*.parquet"))]
            self.ids = [(ep, frame) for ep, table in enumerate(self.tables) for frame in table["frame_index"]]
            self.hf_dataset = datasets.Dataset.from_dict(
                {"episode_index": [i[0] for i in self.ids], "frame_index": [i[1] for i in self.ids]}
            )

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, index):
            ep, frame = self.ids[index]
            table = self.tables[ep]
            return {
                "observation.state": np.asarray(table["observation.state"][frame], np.float32),
                "action": np.asarray(table["action"][frame : frame + self.horizon], np.float32),
                "episode_index": ep,
                "frame_index": frame,
                "task_index": 0,
                **{key: np.zeros((1, 1, 3), np.uint8) for key in self.meta.camera_keys},
            }

    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", metadata)
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", Source)

    def config():
        dataset = build_no_block_one_dataset(owner="test", trial="collection", root=str(parent))
        base = build_experiment(
            owner="test", trial="collection", mode="timestamp_zscore_arms", beta=beta, dataset=dataset
        )
        return replace(
            base, model=replace(base.model, action_horizon=horizon), fsdp_devices=1, batch_size=2, num_workers=0
        )

    before = config()
    run_training.preflight(before, require_norm=False)
    compute(before, batch_size=3)
    current = config()
    result = run_training.preflight(current)
    from scripts import training_ui_config as editor

    ui_config = editor.apply_overrides(current, editor.config_values(current))
    assert ui_config.data.lerobot_datasets == current.data.lerobot_datasets
    assert ui_config.model.paper_asset_sha256 == current.model.paper_asset_sha256
    assert run_training.preflight(ui_config)["norm_provenance"] == "verified"
    changed = editor.config_values(current)
    changed["asset_id"] += "_fresh_ui"
    fresh = editor.apply_overrides(current, changed)
    assert fresh.model.paper_asset_dir == str(Path(changed["assets_dir"]) / changed["asset_id"])
    assert fresh.model.paper_asset_sha256 is None
    with pytest.raises(ValueError, match="个人配置"):
        editor.apply_overrides(
            current,
            {
                **changed,
                "dataset_roots": list(reversed(changed["dataset_roots"])),
                "dataset_root": changed["dataset_roots"][-1],
            },
        )
    directory = current.data.asset_directory(current.assets_dirs)
    manifest, arrays = paper.load_bundle(
        directory, mode="timestamp_zscore_arms", beta=beta, horizon=horizon, std_floor=1e-4
    )
    assert result["frames"] == 12 * horizon + 51
    assert result["episodes"] == 12
    assert manifest["training_windows"] == 63
    assert [row["training_windows"] for row in manifest["datasets"]] == list(range(8, 14))
    assert sum(row["training_windows"] for row in manifest["datasets"]) == 63
    np.testing.assert_allclose(arrays["arm_mean"][:, 0], np.asarray(all_windows).mean(axis=0)[:, 0], atol=1e-7)
    view = current.data.create(current.assets_dirs, current.model)
    source = data_loader.create_torch_dataset(view, horizon, current.model, sampling_mode="epoch_shuffle")
    assert len(source) == 63
    assert len(np.unique(source.frame_ids[:, 0])) == 12
    identity, ids = _source_identity(view, source)
    assert len(identity["sources"]) == 6
    assert ids.shape == (63, 2)
    for row in range(len(source)):
        # A future window stays within the same source episode, with no padding.
        assert source[row]["action"].shape == (horizon, 58)
        np.testing.assert_allclose(np.diff(source[row]["action"][:, 0]), 0.02, atol=3e-7)
    loader = data_loader.create_data_loader(
        current, framework="jax", skip_norm_stats=False, shuffle=True, num_batches=1
    )
    assert loader.prepare_reproducibility(current, resuming=False)
    observation, actions = next(iter(loader))
    assert observation.state.shape == (2, 58)
    assert actions.shape == (2, horizon, 58)
    for path, expected in fingerprints.items():
        assert paper.sha256(path) == expected
    # Altering one source invalidates the shared statistics.
    changed = parent / NO_BLOCK_ONE_MEMBERS[-1] / "meta/tasks.jsonl"
    changed.write_text('{"task_index": 0, "task": "Another task."}\n')
    with pytest.raises(ValueError, match="differ"):
        run_training.preflight(current)


def test_no_block_one_generator_and_fixed_source_order(tmp_path, monkeypatch):
    from openpi.training.delta_paper_experiment import NO_BLOCK_ONE_MEMBERS
    from openpi.training.delta_paper_experiment import build_no_block_one_dataset
    from scripts.create_delta_paper_experiments import create

    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    generated = create("mengxiang", dataset="no-block-one")
    assert (generated / "delta_paper_no_block_one_t3.py").is_file()
    assert "build_no_block_one_dataset" in (generated / "delta_paper_no_block_one_t3.py").read_text()
    dataset = build_no_block_one_dataset(owner="mengxiang", trial="t3")
    assert len(dataset.members) == 6
    assert [str(Path(member.root).relative_to(dataset.root)) for member in dataset.members] == list(
        NO_BLOCK_ONE_MEMBERS
    )
    assert "no_block_one" in dataset.asset_id
    assert "1016_2" not in dataset.asset_id
    config = build_experiment(owner="mengxiang", trial="t3", mode="timestamp_zscore_arms", beta=0.5, dataset=dataset)
    assert len(config.data.lerobot_datasets) == 6
    assert config.policy_metadata["source_datasets"] == list(dataset.sources)
    assert config.model.paper_asset_dir == str(Path(dataset.assets_dir) / dataset.asset_id)
    with pytest.raises(ValueError, match="uniform"):
        replace(config.data, lerobot_datasets=(replace(config.data.lerobot_datasets[0], weight=2),))
