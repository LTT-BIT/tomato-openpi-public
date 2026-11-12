"""Correlated noise prior; full-window training and sampling inherit Pi0."""

from dataclasses import dataclass

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.shared import delta_paper as paper
from openpi.training.weight_loaders import PartialCheckpointWeightLoader


@dataclass(frozen=True)
class PaperPi0Config(Pi0Config):
    paper_asset_dir: str = ""
    paper_asset_sha256: str | None = None
    paper_mode: str = "global_quantile"
    paper_beta: float = 0.0
    paper_std_floor: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if (
            not self.pi05
            or self.action_dim != 58
            or self.action_horizon <= 0
            or self.paper_mode not in paper.MODES
            or not 0 <= self.paper_beta < 1
        ):
            raise ValueError("Paper model requires Pi0.5, 58D actions and valid mode/beta")

    def create(self, rng):
        return PaperPi0(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path):
        raise ValueError("Delta paper experiments are JAX-only; PyTorch is not implemented")


@dataclass(frozen=True)
class PaperWeightLoader(PartialCheckpointWeightLoader):
    """Retain initialized nontrainable correlation state when loading base weights."""

    def load(self, params):
        base = {key: value for key, value in params.items() if key != "paper_noise_cholesky"}
        result = super().load(base)
        if "paper_noise_cholesky" in params:
            result["paper_noise_cholesky"] = params["paper_noise_cholesky"]
        return result


class PaperPi0(Pi0):
    def __init__(self, config, *, rngs):
        if not config.paper_asset_sha256:
            raise ValueError("Compute matching paper statistics and reload configuration before model creation")
        _, arrays = paper.load_bundle(
            config.paper_asset_dir,
            mode=config.paper_mode,
            beta=config.paper_beta,
            horizon=config.action_horizon,
            std_floor=config.paper_std_floor,
            expected_sha256=config.paper_asset_sha256,
        )
        super().__init__(config, rngs=rngs)
        self.paper_beta = config.paper_beta
        if self.paper_beta:
            # Variable rather than Param: checkpointed and device-resident, excluded from optimizer.
            self.paper_noise_cholesky = nnx.Variable(jnp.asarray(arrays["noise_cholesky"]))

    def noise(self, rng, shape):
        value = jax.random.normal(rng, shape)
        if not self.paper_beta:
            return value
        flat = value.reshape(*shape[:-2], self.action_horizon * self.action_dim)
        return (flat @ self.paper_noise_cholesky.value.T).reshape(shape)
