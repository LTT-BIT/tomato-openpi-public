"""Full-horizon T3 optimization, sampling and checkpoint integration."""
# ruff: noqa: PLC0415, SLF001

from dataclasses import replace
import types

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_delta_paper_training import prepared as source_prepared
from test_delta_paper_training import shrink_network

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import delta_paper as paper
from openpi.shared import nnx_utils
from openpi.training.delta_paper_stats import compute
from scripts import training_ui_config as editor


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    return source_prepared.__wrapped__(tmp_path, monkeypatch, request)


def legacy_t3_loss(model, rng, observation, actions):
    """T3 reference from source 2ab68c8, with identical weights/inputs."""
    preprocess, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = model_lib.preprocess_observation(preprocess, observation, train=False)
    epsilon = model.noise(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x = time[..., None, None] * epsilon + (1 - time[..., None, None]) * actions
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(observation, x, time)
    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    (_, out), _ = model.PaliGemma.llm(
        [prefix, suffix],
        mask=make_attn_mask(mask, jnp.concatenate([prefix_ar, suffix_ar])),
        positions=jnp.cumsum(mask, axis=1) - 1,
        adarms_cond=[None, cond],
    )
    return jnp.mean((model.action_out_proj(out[:, -model.action_horizon :]) - (epsilon - actions)) ** 2, axis=-1)


@pytest.mark.parametrize("prepared", [50], indirect=True)
def test_real_50x58_t3_jit_adamw_checkpoint_and_reference_equivalence(prepared, monkeypatch, tmp_path):
    from openpi.training import checkpoints
    from openpi.training import data_loader
    from openpi.training import sharding
    from scripts import train

    build, _, _ = prepared
    compute(build(mode="timestamp_zscore_arms", beta=0.5), batch_size=3)
    base = build(mode="timestamp_zscore_arms", beta=0.5)
    # The parsed UI configuration feeds the full-horizon network.
    values = {**editor.config_values(base), "exp_name": "public_t3"}
    config = editor.apply_overrides(base, values)
    config = replace(config, model=replace(config.model, dtype="float32", max_token_len=4), batch_size=3, num_workers=0)
    shrink_network(monkeypatch)
    model = config.model.create(jax.random.key(0))
    initial = nnx.state(model)
    assert "paper_noise_cholesky" not in initial.filter(config.trainable_filter)
    base_weights = {k: v for k, v in initial.to_pure_dict().items() if k != "paper_noise_cholesky"}
    restore_params = model_lib.restore_params
    monkeypatch.setattr(model_lib, "restore_params", lambda *a, **kw: base_weights)
    monkeypatch.setattr("openpi.training.weight_loaders.download.maybe_download", lambda path: path)
    mesh = sharding.make_mesh(1)
    state, _ = train.init_train_state(config, jax.random.key(42), mesh, resume=False)
    loader = data_loader.create_data_loader(config, shuffle=True, num_batches=1)
    obs, actions = next(iter(loader))
    assert actions.shape == (3, 50, 58)
    assert config.model.paper_beta == 0.5
    updated, metrics = jax.jit(lambda s: train.train_step(config, jax.random.key(42), s, (obs, actions)))(state)
    assert int(updated.step) == 1
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["grad_norm"])
    assert metrics["grad_norm"] > 0
    np.testing.assert_array_equal(
        updated.params["paper_noise_cholesky"].value, state.params["paper_noise_cholesky"].value
    )
    assert not np.array_equal(
        updated.params["action_out_proj"]["kernel"].value, state.params["action_out_proj"]["kernel"].value
    )
    trained = nnx.merge(updated.model_def, updated.params)
    # Compare against an independent reference of the original T3 full-window loss.
    np.testing.assert_array_equal(
        trained.compute_loss(jax.random.key(5), obs, actions), legacy_t3_loss(trained, jax.random.key(5), obs, actions)
    )
    sample = nnx_utils.module_jit(trained.sample_actions)
    expected = sample(jax.random.key(3), obs, num_steps=2)
    assert expected.shape == (3, 50, 58)
    assert np.isfinite(expected).all()
    # Batch one is supported with the actual 58D network.
    one_obs = jax.tree.map(lambda x: x[:1] if x is not None else None, obs)
    one = sample(jax.random.key(3), one_obs, num_steps=2)
    assert one.shape == (1, 50, 58)
    assert np.isfinite(one).all()
    view = config.data.create(config.assets_dirs, config.model)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "saved", keep_period=10, overwrite=False, resume=False
    )
    checkpoints.save_state(manager, updated, types.SimpleNamespace(data_config=lambda: view), step=1)
    manager.wait_until_finished()
    manager.close()
    asset = tmp_path / "saved/1/assets" / view.asset_id
    source = config.data.asset_directory(config.assets_dirs)
    for name in ("norm_stats.json", "norm_metadata.json", paper.BUNDLE, paper.MANIFEST):
        assert (asset / name).read_bytes() == (source / name).read_bytes()
    params = restore_params(tmp_path / "saved/1/params", restore_type=np.ndarray)
    restored = config.model.load(params)
    np.testing.assert_array_equal(
        nnx.state(restored)["paper_noise_cholesky"].value, updated.params["paper_noise_cholesky"].value
    )
    restored_sample = nnx_utils.module_jit(restored.sample_actions)
    np.testing.assert_array_equal(restored_sample(jax.random.key(3), obs, num_steps=2), expected)
    # The same loader can initialize a NEW training run from a T3 checkpoint, retaining its pinned prior.
    monkeypatch.setattr(model_lib, "restore_params", lambda *a, **kw: params)
    new_weights = replace(config.weight_loader, params_path="fixture_t3_params").load(initial.to_pure_dict())
    np.testing.assert_array_equal(new_weights["paper_noise_cholesky"], initial["paper_noise_cholesky"].value)
    np.testing.assert_array_equal(new_weights["action_out_proj"]["kernel"], params["action_out_proj"]["kernel"])
