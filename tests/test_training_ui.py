"""Exercise UI config persistence, job isolation and process control without GPU training."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen

import numpy as np
import pytest

from openpi import transforms
from scripts import run_training
from scripts import training_ui as ui
from scripts import training_ui_config as editor
from training_configs import common

BASE = "training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1"


@pytest.mark.parametrize("entry", editor.baseline_catalog(), ids=lambda entry: entry["id"].split(".")[-1])
def test_all_baseline_defaults_roundtrip_and_personal_snapshot(entry):
    module = entry["id"]
    baseline = editor.baseline_config(module)
    assert baseline.weight_loader.params_path == common.SERVER_BASE_PARAMS
    values = editor.defaults(module)["values"]
    restored = editor.apply_overrides(baseline, values)
    assert restored == baseline
    values = {**values, "exp_name": "alice_ui_trial", "batch_size": 4, "peak_lr": 1e-5}
    scope = {}
    exec(compile(editor.personal_source(module, "alice", values), "personal.py", "exec"), scope)
    saved = scope["build_config"]()
    assert saved.exp_name == "alice_ui_trial"
    assert saved.batch_size == 4
    assert saved.lr_schedule.peak_lr == 1e-5
    assert saved.model == baseline.model
    assert saved.freeze_filter == baseline.freeze_filter
    assert saved.data.base_config == baseline.data.base_config
    assert saved.policy_metadata["ui_baseline"] == module
    assert editor.baseline_config(module) == baseline


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": True},
        {"batch_size": 0},
        {"num_workers": -1},
        {"peak_lr": float("nan")},
        {"decay_steps": 1000},
        {"action_horizon": 0},
        {"exp_name": "../escape"},
        {"asset_id": "../escape"},
        {"asset_id": "/escape"},
        {"sources": []},
        {"model": "unknown"},
        {"save_before_step": 100},
    ],
)
def test_invalid_edits_fail_before_saving(change):
    values = {**editor.defaults(BASE)["values"], **change}
    with pytest.raises((ValueError, TypeError)):
        editor.apply_overrides(editor.baseline_config(BASE), values)


@pytest.mark.parametrize(("task", "dimension", "second_arm"), [("tomato_wuji", 54, 27), ("tomato_sharpa", 58, 29)])
@pytest.mark.parametrize("method", ["lora", "fullft"])
def test_ui_delta_selection_preserves_arm_and_hand_semantics(task, dimension, second_arm, method):
    module = next(entry["id"] for entry in editor.baseline_catalog() if entry["task"] == task and method in entry["id"])
    original = editor.baseline_config(module)
    defaults = editor.defaults(module)
    assert defaults["protocol"]["actions"] == "absolute"
    assert defaults["protocol"]["delta_available"]
    values = {**defaults["values"], "action_semantics": "delta", "asset_id": "ui_delta_test"}
    assert editor.changed_parameters(module, values) == ["action_semantics"]
    config = editor.apply_overrides(original, values)
    assert config.data.extra_delta_transform
    assert "delta" in config.name
    assert "absolute" not in config.name
    assert config.policy_metadata["training_action_semantics"] == "delta_arm_joint_target_absolute_hand"
    assert config.data.assets.asset_id == "ui_delta_test"
    _, vectors = config.data.create_vector_transforms(config.model)
    assert isinstance(vectors.inputs[-1], transforms.DeltaActions)
    assert isinstance(vectors.outputs[0], transforms.AbsoluteActions)
    state = np.arange(dimension, dtype=np.float32)
    absolute = np.stack((state + 2, state + 3))
    delta = vectors.inputs[-1]({"state": state, "actions": absolute.copy()})["actions"]
    for start in (0, second_arm):
        np.testing.assert_allclose(delta[:, start : start + 7], [[2] * 7, [3] * 7])
    np.testing.assert_allclose(delta[:, 7:second_arm], absolute[:, 7:second_arm])
    np.testing.assert_allclose(delta[:, second_arm + 7 :], absolute[:, second_arm + 7 :])
    restored = vectors.outputs[0]({"state": state, "actions": delta.copy()})["actions"]
    np.testing.assert_allclose(restored, absolute)
    scope = {}
    exec(compile(editor.personal_source(module, "alice", values), "personal.py", "exec"), scope)
    saved = scope["build_config"]()
    assert saved.data == config.data
    assert saved.name == config.name
    assert saved.policy_metadata["training_action_semantics"] == config.policy_metadata["training_action_semantics"]
    assert editor.baseline_config(module) == original


def test_ui_delta_rejects_old_stats_and_unsupported_tasks():
    baseline = editor.baseline_config(BASE)
    values = editor.defaults(BASE)["values"]
    with pytest.raises(ValueError, match="新的 asset_id"):
        editor.apply_overrides(baseline, {**values, "action_semantics": "delta"})
    with pytest.raises(ValueError, match="只能选择"):
        editor.apply_overrides(baseline, {**values, "action_semantics": "unknown"})
    legacy = {key: value for key, value in values.items() if key != "action_semantics"}
    assert editor.apply_overrides(baseline, legacy) == baseline
    cup = next(entry["id"] for entry in editor.baseline_catalog() if entry["task"] == "cup_grasping")
    assert not editor.defaults(cup)["protocol"]["delta_available"]
    with pytest.raises(ValueError, match="仅番茄"):
        editor.apply_overrides(
            editor.baseline_config(cup),
            {**editor.defaults(cup)["values"], "action_semantics": "delta", "asset_id": "new"},
        )


def test_grouped_sampling_and_data_identity_are_saved():
    values = {
        **editor.defaults(BASE)["values"],
        "dataset_root": "/tmp/new-ui-dataset",
        "repo_id": "team/new",
        "dataset_version": "new_v1",
        "sources": ["capture_20261014"],
        "sampling_mode": "weighted_replacement",
        "sampling_groups": [
            {"name": "A", "episodes": [0, 2], "weight": 7},
            {"name": "B", "episodes": [1], "weight": 3},
        ],
    }
    source = editor.personal_source(BASE, "alice", values)
    scope = {}
    exec(compile(source, "personal.py", "exec"), scope)
    config = scope["build_config"]()
    assert config.data.repo_id == config.policy_metadata["dataset_repo_id"] == "team/new"
    assert config.data.lerobot_datasets[0].root == config.policy_metadata["dataset_root"] == "/tmp/new-ui-dataset"
    assert config.policy_metadata["source_datasets"] == ["capture_20261014"]
    assert [group.weight for group in config.sampling_groups] == [7, 3]
    assert config.sampling_groups[0].episodes == (0, 2)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    catalog = editor.baseline_catalog()
    for entry in catalog:
        source = common.repository_root() / entry["file"]
        destination = tmp_path / entry["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(editor, "baseline_catalog", lambda: catalog)
    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        ui,
        "gpu_inventory",
        lambda: {
            "gpus": [
                {"uuid": "GPU-test0", "index": "0", "name": "test", "total_mb": 80000, "used_mb": 0, "utilization": 0},
                {
                    "uuid": "GPU-test1",
                    "index": "1",
                    "name": "test",
                    "total_mb": 80000,
                    "used_mb": 20000,
                    "utilization": 90,
                },
            ],
            "error": None,
        },
    )
    values = editor.defaults(BASE)["values"]
    values["exp_name"] = "alice_test_ui"
    values["peak_lr"] = 1e-5
    return {
        "baseline": BASE,
        "owner": "alice",
        "values": values,
        "gpus": ["GPU-test0"],
        "auto_norm": True,
        "wandb_mode": "offline",
        "stats_batch_size": 8,
    }


@pytest.fixture
def fake_launch(monkeypatch):
    def launch(state):
        directory = ui.job_directory(state["id"])
        directory.mkdir(parents=True)
        state.update(worker_pid=os.getpid(), worker_start=ui.process_identity(os.getpid()))
        ui.atomic_json(directory / "state.json", state)
        if "values" in state:
            ui.atomic_json(directory / "run.json", {k: state[k] for k in ("module", "config_sha256", "values")})
        (directory / "ready").touch()
        (directory / "console.log").touch()
        return state

    monkeypatch.setattr(ui, "launch_worker", launch)


def test_launch_saves_personal_file_without_mutating_baseline(workspace, fake_launch):
    state = ui.start_job(workspace)
    source = Path(state["config_path"]).read_bytes()
    assert state["module"].endswith("template.alice.alice_test_ui")
    assert hashlib.sha256(source).hexdigest() == state["config_sha256"]
    assert state["wandb_mode"] == "offline"
    with pytest.raises(ValueError, match="预留"):
        ui.start_job({**workspace, "values": {**workspace["values"], "exp_name": "another"}})
    with pytest.raises(ValueError, match="正在使用"):
        ui.start_job({**workspace, "gpus": ["GPU-test1"]})
    with pytest.raises(ValueError, match="同一实验已有任务"):
        ui.start_job({**workspace, "gpus": ["GPU-test1"], "allow_busy": True})
    state["status"] = "completed"
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    with pytest.raises(FileExistsError):
        ui.start_job({**workspace, "gpus": ["GPU-test1"], "allow_busy": True})


def test_ui_delta_launch_records_mode_and_independent_statistics(workspace, fake_launch):
    original = editor.defaults(BASE)["values"]
    workspace["values"] = {
        **original,
        "exp_name": "alice_delta_trial",
        "asset_id": "alice_delta_trial_stats",
        "action_semantics": "delta",
    }
    state = ui.start_job(workspace)
    assert state["creates_config"]
    assert "template.alice.alice_delta_trial" in state["module"]
    assert state["values"]["action_semantics"] == "delta"
    assert state["values"]["asset_id"] == "alice_delta_trial_stats"
    resolved = editor.config_from_source(Path(state["config_path"]).read_text(), state["module"], state["config_path"])
    assert resolved.data.extra_delta_transform
    assert resolved.policy_metadata["training_action_semantics"] == "delta_arm_joint_target_absolute_hand"
    record = json.loads((ui.job_directory(state["id"]) / "run.json").read_text())
    assert record["values"]["action_semantics"] == "delta"


def test_config_path_rejects_escape_and_symlink(workspace, tmp_path):
    with pytest.raises(ValueError, match="用户名"):
        editor.personal_path(BASE, "../bob", "trial")
    task = tmp_path / "training_configs/experiments/tomato_wuji/template"
    task.mkdir(parents=True)
    (task / "alice").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        editor.personal_path(BASE, "alice", "trial")


def test_gpu_validation(workspace, monkeypatch):
    config = editor.baseline_config(BASE)
    for selected in ([], ["missing"], ["GPU-test0", "GPU-test0"]):
        with pytest.raises(ValueError, match="GPU"):
            ui.validate_devices(selected, config)
    with pytest.raises(ValueError, match="整除"):
        ui.validate_devices(["GPU-test0"], replace(config, fsdp_devices=2))
    ui.validate_devices(["GPU-test0", "GPU-test1"], config, allow_busy=True)
    monkeypatch.setattr(ui, "gpu_inventory", lambda: {"gpus": [], "error": "nvidia-smi missing"})
    with pytest.raises(ValueError, match="nvidia-smi missing"):
        ui.validate_devices(["GPU-test0"], config)


def test_preflight_missing_and_existing_norm(workspace, monkeypatch):
    config, _ = ui.build_request(workspace)
    calls = []
    monkeypatch.setattr(
        run_training, "preflight", lambda config, require_norm=True: calls.append(require_norm) or {"frames": 2}
    )
    result = ui.preview(workspace, check_data=True)
    assert "自动计算" in result["statistics"]
    assert calls == [False]
    norm = Path(result["norm_dir"])
    norm.mkdir(parents=True)
    with pytest.raises(ValueError, match="不完整"):
        ui.preview(workspace, check_data=True)
    (norm / "norm_stats.json").write_text("{}")
    result = ui.preview(workspace, check_data=True)
    assert "可复用" in result["statistics"]
    assert calls[-2:] == [False, True]

    def invalid(config, *, require_norm=True):
        if require_norm:
            raise ValueError("statistics mismatch")
        return {}

    monkeypatch.setattr(run_training, "preflight", invalid)
    with pytest.raises(ValueError, match="mismatch"):
        ui.preview(workspace, check_data=True)


@pytest.mark.parametrize("memory_settings", [None, ("false", "0.70")])
def test_worker_sequences_cpu_preparation_and_gpu_training(workspace, fake_launch, monkeypatch, memory_settings):
    keys = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    if memory_settings is not None:
        for key, value in zip(keys, memory_settings, strict=True):
            monkeypatch.setenv(key, value)
    config, _ = ui.build_request(workspace)
    state = ui.start_job(workspace)
    monkeypatch.setattr(run_training, "load_experiment", lambda module: config)
    calls = []
    monkeypatch.setattr(
        ui, "run_stage", lambda directory, state, label, command, env: calls.append((label, command, env))
    )
    ui.worker(state["id"])
    assert [row[0] for row in calls] == ["校验数据", "计算归一化统计", "核对数据与统计", "训练中"]
    assert all(row[2]["JAX_PLATFORMS"] == "cpu" and row[2]["CUDA_VISIBLE_DEVICES"] == "" for row in calls[:-1])
    assert calls[-1][2]["JAX_PLATFORMS"] == "cuda"
    assert calls[-1][2]["CUDA_VISIBLE_DEVICES"] == "GPU-test0"
    assert calls[-1][1][-2:] == ["--wandb-mode", "offline"]
    expected = dict(zip(keys, memory_settings or ("true", "0.90"), strict=True))
    assert {key: calls[-1][2][key] for key in keys} == expected
    saved = ui.read_job(state["id"])
    assert saved["status"] == "completed"
    assert saved["training_environment"] == expected


def test_worker_failure_never_reaches_training(workspace, fake_launch, monkeypatch):
    config, _ = ui.build_request(workspace)
    state = ui.start_job(workspace)
    monkeypatch.setattr(run_training, "load_experiment", lambda module: config)
    calls = []

    def fail(directory, state, label, command, env):
        calls.append(label)
        raise ValueError("invalid dataset")

    monkeypatch.setattr(ui, "run_stage", fail)
    ui.worker(state["id"])
    assert calls == ["校验数据"]
    assert ui.read_job(state["id"])["status"] == "failed"


def test_resume_checks_config_and_keeps_original_parameters(workspace, fake_launch, monkeypatch):
    config, _ = ui.build_request(workspace)
    state = ui.start_job(workspace)
    state["status"] = "failed"
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    monkeypatch.setattr(run_training, "load_experiment", lambda module: config)
    restored = ui.resume_job({"id": state["id"]})
    assert restored["module"] == state["module"]
    assert restored["resume"] is False  # No checkpoint yet: retry preflight.
    restored["status"] = "failed"
    ui.atomic_json(ui.job_directory(restored["id"]) / "state.json", restored)
    config.checkpoint_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="尚无 checkpoint"):
        ui.resume_job({"id": state["id"]})
    (config.checkpoint_dir / "1000").mkdir()
    restored = ui.resume_job({"id": state["id"]})
    assert restored["resume"] is True
    Path(state["config_path"]).write_text("# modified")
    with pytest.raises(ValueError, match="配置文件已修改"):
        ui.resume_job({"id": state["id"]})


def test_shared_disk_foreign_host_not_marked_dead(workspace, fake_launch):
    state = ui.start_job(workspace)
    state.update(host="another-host", worker_start="different")
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    assert ui.read_job(state["id"])["status"] == "queued"
    with pytest.raises(ValueError, match="原训练服务器"):
        ui.stop_job(state["id"])
    state["host"] = socket.gethostname()
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    assert ui.read_job(state["id"])["status"] == "interrupted"


def test_run_stage_stops_owned_process_group(tmp_path):
    state = {"status": "queued"}
    pid_file = tmp_path / "child_pid"
    code = f"import os,time,pathlib; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"

    def request_stop():
        for _ in range(200):
            if pid_file.exists():
                (tmp_path / "stop").touch()
                return
            time.sleep(0.01)

    thread = threading.Thread(target=request_stop)
    thread.start()
    with pytest.raises(ui.StopRequestedError):
        ui.run_stage(tmp_path, state, "test", [sys.executable, "-c", code], os.environ.copy())
    thread.join()
    assert ui.process_identity(int(pid_file.read_text())) is None


def test_http_requires_token_and_json_for_mutations(workspace, monkeypatch):
    requests = []
    monkeypatch.setattr(ui, "start_job", lambda payload: requests.append(payload) or {"status": "queued"})
    server = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.handler_class("test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url, timeout=5) as response:
            assert b"app.js" in response.read()
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/api/baselines", timeout=5)
        assert error.value.code == 403
        headers = {"X-Training-Token": "test-token", "Content-Type": "application/json"}
        with urlopen(Request(url + "/api/start", data=b"{}", headers=headers), timeout=5) as response:
            assert json.load(response)["status"] == "queued"
        assert requests == [{}]
        with pytest.raises(HTTPError):
            urlopen(Request(url + "/api/start", data=b"{}", headers={"Content-Type": "application/json"}), timeout=5)
        assert requests == [{}]
        with urlopen(Request(url + "/api/baseline?id=" + BASE, headers=headers), timeout=5) as response:
            assert json.load(response)["values"]["batch_size"] == 8
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_group_detection_endpoint_requires_token_and_returns_errors(monkeypatch):
    def detect(payload):
        if payload["dataset_root"] == "/missing":
            raise ValueError("missing provenance")
        return {"groups": [{"name": "A", "episodes": [0, 2]}]}

    monkeypatch.setattr(ui, "detect_sampling_groups", detect)
    server = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.handler_class("test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/sampling-groups"
    headers = {"Content-Type": "application/json"}
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, data=b'{"dataset_root":"/data"}', headers=headers), timeout=5)
        assert error.value.code == 403
        headers["X-Training-Token"] = "test-token"
        with urlopen(Request(url, data=b'{"dataset_root":"/data"}', headers=headers), timeout=5) as response:
            assert json.load(response)["groups"][0]["episodes"] == [0, 2]
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, data=b'{"dataset_root":"/missing"}', headers=headers), timeout=5)
        assert error.value.code == 400
        assert json.load(error.value)["error"] == "missing provenance"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_group_detection_rejects_filtered_task_view(monkeypatch):
    config = editor.baseline_config(BASE)
    from openpi.training.config import TaskDataConfig
    from openpi.training.task_view import TaskView

    config = replace(config, data=replace(config.data, base_config=TaskDataConfig(task_view=TaskView())))
    monkeypatch.setattr(editor, "source_config", lambda _: config)
    with pytest.raises(ValueError, match="episode"):
        ui.detect_sampling_groups({"baseline": BASE, "dataset_root": "/unused"})


def test_living_training_child_keeps_gpu_reserved_after_supervisor_exit(workspace, fake_launch):
    state = ui.start_job(workspace)
    state.update(worker_start="dead-supervisor", child_pid=os.getpid(), child_start=ui.process_identity(os.getpid()))
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    assert ui.read_job(state["id"])["status"] == "orphaned"
    with pytest.raises(ValueError, match="预留"):
        ui.validate_devices(["GPU-test0"], editor.baseline_config(BASE))


def test_reuse_records_each_run_without_new_personal_file(workspace, fake_launch):
    workspace["values"] = {**editor.defaults(BASE)["values"], "exp_name": "reuse_one", "dataset_root": "/tmp/data_one"}
    first = ui.start_job(workspace)
    assert first["creates_config"] is False
    assert first["module"] == BASE
    assert not list((common.repository_root() / "training_configs/experiments").glob("*/template/*/*.py"))
    record = json.loads((ui.job_directory(first["id"]) / "run.json").read_text())
    assert record["values"]["dataset_root"] == "/tmp/data_one"
    first["status"] = "completed"
    ui.atomic_json(ui.job_directory(first["id"]) / "state.json", first)
    workspace["values"].update(exp_name="reuse_two", dataset_root="/tmp/data_two")
    second = ui.start_job(workspace)
    assert second["id"] != first["id"]
    assert second["checkpoint_dir"] != first["checkpoint_dir"]
    assert second["config_sha256"] == first["config_sha256"]
    assert ui.job_config(second).data.lerobot_datasets[0].root == "/tmp/data_two"
    assert record["values"]["exp_name"] == "reuse_one"


def test_personal_discovery_fresh_load_and_copy_preserve_custom_settings(workspace):
    path = editor.personal_path(BASE, "alice", "handwritten")
    path.parent.mkdir(parents=True)
    path.write_text(
        f"from {BASE} import build_config as base\n"
        "from dataclasses import replace\n"
        "def build_config():\n"
        "    c = base()\n"
        "    return replace(c, batch_size=4, model=replace(c.model, max_token_len=128))\n"
    )
    module = ".".join(path.relative_to(common.repository_root()).with_suffix("").parts)
    assert module in {entry["id"] for entry in editor.personal_catalog()}
    assert editor.defaults(module)["values"]["batch_size"] == 4
    path.write_text(path.read_text().replace("batch_size=4", "batch_size=2"))
    assert editor.defaults(module)["values"]["batch_size"] == 2
    values = editor.defaults(module)["values"]
    assert editor.changed_parameters(module, {**values, "exp_name": "next", "dataset_root": "/tmp/relocated"}) == []
    assert editor.changed_parameters(module, {**values, "peak_lr": 1e-5}) == ["peak_lr"]
    first = editor.personal_path(module, "alice", "copy_one")
    first.write_text(editor.personal_source(module, "alice", {**values, "peak_lr": 1e-5}))
    first_module = ".".join(first.relative_to(common.repository_root()).with_suffix("").parts)
    second_values = {**editor.defaults(first_module)["values"], "batch_size": 8}
    scope = {}
    exec(editor.personal_source(first_module, "alice", second_values), scope)
    copied = scope["build_config"]()
    assert copied.model.max_token_len == 128
    assert copied.batch_size == 8
    assert copied.lr_schedule.peak_lr == 1e-5
    assert editor.source_config(module).batch_size == 2


def test_run_override_matches_all_stages_and_resume(workspace, fake_launch, monkeypatch):
    workspace["values"] = {
        **editor.defaults(BASE)["values"],
        "exp_name": "run_override",
        "dataset_root": "/tmp/relocated",
    }
    state = ui.start_job(workspace)
    config = ui.job_config(state)
    assert config.exp_name == "run_override"
    assert config.policy_metadata["dataset_root"] == config.data.lerobot_datasets[0].root == "/tmp/relocated"
    calls = []
    monkeypatch.setattr(ui, "run_stage", lambda directory, state, label, command, env: calls.append(command))
    ui.worker(state["id"])
    assert len(calls) == 4
    assert all(
        "--ui-run" in command and str(ui.job_directory(state["id"]) / "run.json") in command for command in calls
    )
    state["status"] = "failed"
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    resumed = ui.resume_job({"id": state["id"]})
    assert ui.job_config(resumed) == config
    path = Path(resumed["config_path"])
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="does not match"):
        ui.job_config(resumed)


def test_source_change_after_form_load_is_rejected(workspace):
    workspace["source_sha256"] = editor.defaults(BASE)["source_sha256"]
    path = editor.source_path(BASE)
    path.write_text(path.read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="来源配置已修改"):
        ui.build_request(workspace)


def test_personal_catalog_skips_broken_and_external_files(workspace, tmp_path):
    directory = tmp_path / "training_configs/experiments/tomato_wuji/template/alice"
    directory.mkdir(parents=True)
    (directory / "broken.py").write_text("def broken(")
    (directory / "helper.py").write_text("VALUE = 1")
    (directory / "external.py").symlink_to(Path(__file__).resolve())
    assert editor.personal_catalog() == []


@pytest.mark.parametrize("trial", ["none", "t1", "t2", "t3"])
def test_gui_paper_features_save_reload_without_generator(tmp_path, trial):
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.pi0_delta_paper import PaperPi0Config
    from openpi.training.delta_paper import PaperSharpaData

    module = next(e["id"] for e in editor.baseline_catalog() if e["task"] == "tomato_sharpa" and "fullft" in e["id"])
    original = editor.source_config(module)
    values = {
        **editor.defaults(module)["values"],
        "paper_trial": trial,
        "action_semantics": "delta",
        "asset_id": f"gui_{trial}",
        "assets_dir": str(tmp_path),
        "exp_name": f"gui_{trial}",
        "dataset_root": "/data/source_a",
        "dataset_roots": ["/data/source_a", "/data/source_b"],
        "sources": ["/data/source_a", "/data/source_b"],
        "dataset_version": "gui_two_sources_v1",
    }
    config = editor.apply_overrides(original, values)
    is_paper = trial != "none"
    assert type(config.model) is (PaperPi0Config if is_paper else Pi0Config)
    assert isinstance(config.data, PaperSharpaData) == is_paper
    assert [spec.root for spec in config.data.lerobot_datasets] == values["dataset_roots"]
    assert config.policy_metadata["dataset_version"] == "gui_two_sources_v1"
    assert config.policy_metadata["source_datasets"] == values["sources"]
    assert config.policy_metadata["prompt_example"] == original.policy_metadata["prompt_example"]
    scope = {}
    exec(editor.personal_source(module, "alice", values), scope)
    restored = scope["build_config"]()
    assert restored.model == config.model
    assert restored.data == config.data
    assert editor.config_values(restored)["paper_trial"] == trial
    if is_paper:
        assert config.model.paper_mode == ("timestamp_zscore_arms" if trial in ("t1", "t3") else "global_quantile")
        assert config.model.paper_beta == (0.5 if trial in ("t2", "t3") else 0.0)
        assert config.model.paper_asset_dir == str(tmp_path / values["asset_id"])
        assert config.policy_metadata["paper_algorithm"] == config.data.paper_contract()
        assert "paper_trial" in editor.changed_parameters(module, values)
        for forbidden in ({"sampling_mode": "weighted_replacement"}, {"action_semantics": "absolute"}):
            with pytest.raises(ValueError, match="论文实验"):
                editor.apply_overrides(original, {**values, **forbidden})
        with pytest.raises(ValueError, match="asset_id"):
            editor.apply_overrides(original, {**values, "asset_id": original.data.assets.asset_id})
        # Disabling Paper creates an ordinary recipe with independent statistics.
        off = editor.apply_overrides(config, {**editor.config_values(config), "paper_trial": "none", "asset_id": "off"})
        assert type(off.model) is Pi0Config
        assert not isinstance(off.data, PaperSharpaData)
        assert "paper_algorithm" not in off.policy_metadata
    assert editor.source_config(module) == original


@pytest.mark.parametrize(("task", "method"), [("tomato_wuji", "fullft"), ("tomato_sharpa", "lora")])
def test_gui_rejects_unsupported_paper_recipes(task, method):
    module = next(e["id"] for e in editor.baseline_catalog() if e["task"] == task and method in e["id"])
    assert not editor.defaults(module)["protocol"]["paper_available"]
    with pytest.raises(ValueError, match="Sharpa"):
        editor.apply_overrides(
            editor.source_config(module),
            {**editor.defaults(module)["values"], "paper_trial": "t3", "asset_id": "new_paper"},
        )
