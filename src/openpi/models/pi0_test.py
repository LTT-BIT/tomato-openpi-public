import flax.nnx as nnx
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_independent_augmentation_preserves_flow_randomness(monkeypatch):
    """Inspect real loss preprocessing before the expensive network forward."""
    observed = []
    noise_samples = []
    time_samples = []
    random_normal = jax.random.normal
    random_beta = jax.random.beta

    def preprocess(rng, observation, *, train):
        observed.append(jax.random.key_data(rng))
        assert train
        return observation

    def normal(*args, **kwargs):
        result = random_normal(*args, **kwargs)
        noise_samples.append(result)
        return result

    def beta(*args, **kwargs):
        result = random_beta(*args, **kwargs)
        time_samples.append(result)
        return result

    class StopBeforeNetworkError(Exception):
        pass

    class StopBeforeNetwork:
        noise = pi0.Pi0.noise

        def embed_prefix(self, observation):
            del observation
            raise StopBeforeNetworkError

    monkeypatch.setattr(_model, "preprocess_observation", preprocess)
    monkeypatch.setattr(jax.random, "normal", normal)
    monkeypatch.setattr(jax.random, "beta", beta)
    cfg = _pi0_config.Pi0Config(action_dim=2, action_horizon=2)
    rng = jax.random.key(42)
    for augmentation_rng in (None, jax.random.key(42002), jax.random.key(43002)):
        with pytest.raises(StopBeforeNetworkError):
            pi0.Pi0.compute_loss(
                StopBeforeNetwork(),
                rng,
                cfg.fake_obs(2),
                cfg.fake_act(2),
                train=True,
                augmentation_rng=augmentation_rng,
            )
    np.testing.assert_array_equal(observed[0], jax.random.key_data(jax.random.split(rng, 3)[0]))
    np.testing.assert_array_equal(observed[1], jax.random.key_data(jax.random.key(42002)))
    np.testing.assert_array_equal(observed[2], jax.random.key_data(jax.random.key(43002)))
    for sample in noise_samples[1:]:
        np.testing.assert_array_equal(sample, noise_samples[0])
    for sample in time_samples[1:]:
        np.testing.assert_array_equal(sample, time_samples[0])
