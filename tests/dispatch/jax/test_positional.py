from functools import partial

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("jax")

from pytensor_ml.layers.norm import RMSNorm
from pytensor_ml.layers.positional import rotary_embedding
from tests.dispatch.jax.test_basic import compare_jax_and_py

floatX = pytensor.config.floatX
assert_close = partial(np.testing.assert_allclose, atol=1e-5, rtol=1e-4)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "JAX Positional")))


# Neither op registers a jax_funcify of its own: both are SymbolicOps whose inner graphs are built
# from ops the backend already knows, so pytensor's OpFromGraph fallback inlines them. These tests
# hold that claim to account -- a primitive the backend cannot lower is not a usable primitive.
@pytest.mark.parametrize("pairing", ["half", "adjacent"])
def test_rotary_embedding_matches_py(pairing, rng):
    x_np = rng.normal(size=(2, 3, 5, 8)).astype(floatX)
    x = pt.tensor("x", shape=x_np.shape)
    positions = pt.lvector("positions")

    out = rotary_embedding(x, positions, pairing=pairing)
    compare_jax_and_py([x, positions], out, [x_np, np.arange(5)], assert_fn=assert_close)


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
def test_rms_norm_matches_py(affine, rng):
    X_np = rng.normal(size=(8, 6)).astype(floatX)
    X = pt.tensor("X", shape=(None, 6))

    rms_norm = RMSNorm("rms", n_in=6, affine=affine)
    if affine:
        rms_norm.scale.set_value(rng.normal(size=6).astype(floatX))

    compare_jax_and_py([X], rms_norm(X), [X_np], assert_fn=assert_close)
