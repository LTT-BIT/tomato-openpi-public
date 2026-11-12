"""Keep the public training interface independent of upstream prefix conditioning."""

from dataclasses import fields
import inspect
from pathlib import Path

import pytest

from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_delta_paper import PaperPi0
from openpi.models.pi0_delta_paper import PaperPi0Config
from openpi.training.delta_paper_experiment import build_experiment
from scripts import create_delta_paper_experiments as generator
from scripts import training_ui_config as editor


def test_public_model_and_experiment_interfaces_exclude_delay_and_prefix():
    for cls in (Pi0Config, PaperPi0Config):
        assert "rtc_max_delay" not in {field.name for field in fields(cls)}
        with pytest.raises(TypeError):
            cls(rtc_max_delay=10)
    for cls in (Pi0, PaperPi0):
        parameters = inspect.signature(cls.sample_actions).parameters
        assert "prefix_actions" not in parameters
        assert "prefix_length" not in parameters
    for function in (build_experiment, generator.create, generator.source):
        assert "rtc_max_delay" not in inspect.signature(function).parameters
    assert PaperPi0.compute_loss is Pi0.compute_loss
    assert PaperPi0.sample_actions is Pi0.sample_actions


@pytest.mark.parametrize("trial", ["none", "t1", "t2", "t3"])
def test_public_ui_rejects_upstream_delay_settings(trial):
    module = next(e["id"] for e in editor.baseline_catalog() if e["task"] == "tomato_sharpa" and "fullft" in e["id"])
    defaults = editor.defaults(module)
    assert "rtc_max_delay" not in defaults["values"]
    assert "rtc_max_delay" not in defaults["protocol"]
    with pytest.raises(ValueError, match="不支持的参数"):
        editor.apply_overrides(
            editor.source_config(module),
            {**defaults["values"], "paper_trial": trial, "asset_id": "public_trial", "rtc_max_delay": 10},
        )


def test_public_web_form_and_generator_reject_delay(monkeypatch):
    web = (Path(__file__).resolve().parents[1] / "scripts/training_ui_web/app.js").read_text()
    assert "rtc_max_delay" not in web
    monkeypatch.setattr("sys.argv", ["create_delta_paper_experiments", "--owner", "test", "--rtc-max-delay", "10"])
    with pytest.raises(SystemExit) as error:
        generator.main()
    assert error.value.code == 2
