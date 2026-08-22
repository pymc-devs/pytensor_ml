from functools import partial

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("jax")

from pytensor_ml.layers import Conv1D
from pytensor_ml.layers.conv import ConvLayer, ConvLayerGrad
from tests.dispatch.jax.test_basic import compare_jax_and_py

floatX = pytensor.config.floatX
assert_close = partial(np.testing.assert_allclose, atol=1e-5, rtol=1e-4)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "jax conv")))


@pytest.mark.parametrize(
    "stride, dilation",
    [(1, 1), (2, 1), (1, 2), (3, 2)],
    ids=["plain", "strided", "dilated", "both"],
)
def test_the_conv_op_matches_the_graph_it_replaces(stride, dilation, rng):
    """The dispatch has to agree with the gather-and-Dot it stands in for, at every combination of
    stride and dilation the op carries in its props."""
    X_np = rng.normal(size=(2, 16, 3)).astype(floatX)
    W_np = rng.normal(size=(3, 3, 4)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    W = pt.tensor("W", shape=W_np.shape)

    out = ConvLayer((3,), (stride,), (dilation,))(X, W)
    compare_jax_and_py([X, W], out, [X_np, W_np], assert_fn=assert_close)


def test_the_conv_op_adds_its_bias(rng):
    """The bias is a third input to the op rather than an Elemwise outside it, so the dispatch is what
    has to add it; dropping it would go unnoticed by a test that never passes one."""
    X_np = rng.normal(size=(2, 12, 3)).astype(floatX)
    W_np = rng.normal(size=(4, 3, 5)).astype(floatX)
    b_np = rng.normal(size=(5,)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    W = pt.tensor("W", shape=W_np.shape)
    b = pt.tensor("b", shape=b_np.shape)

    out = ConvLayer((4,), (1,), (1,))(X, W, b)
    compare_jax_and_py([X, W, b], out, [X_np, W_np, b_np], assert_fn=assert_close)


def test_a_conv1d_layer_matches_end_to_end(rng):
    """Through the layer rather than the op, so the padding the layer applies ahead of the op is in the
    graph too -- the dispatch assumes what reaches it is already padded."""
    X_np = rng.normal(size=(2, 10, 3)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    layer = Conv1D("conv", in_channels=3, out_channels=6, kernel_size=3, padding="same")
    layer.W.set_value(rng.normal(size=(3, 3, 6)).astype(floatX))
    layer.b.set_value(rng.normal(size=(6,)).astype(floatX))

    compare_jax_and_py([X], layer(X), [X_np], assert_fn=assert_close)


@pytest.mark.parametrize("spatial", [32, None], ids=["static_length", "dynamic_length"])
def test_an_undeclared_length_reaches_jax_only_through_the_dispatch(spatial, rng):
    """The gather builds its window indices from ``X.shape``, and JAX refuses a non-constant ``arange``.
    A declared spatial extent lets `fast_run` fold the shape away, so the graph alone would compile; an
    undeclared one does not fold, and the layer reaches JAX only because the forward and the pullback are
    both ops with dispatches of their own.

    Written against ``mode="JAX"`` rather than the stripped query the other tests here use, since that
    query drops `fast_run` and would hide the folding half of this entirely."""
    X = pt.tensor("X", shape=(8, spatial, 3), dtype=floatX)
    layer = Conv1D("conv", in_channels=3, out_channels=4, kernel_size=3)
    layer.W.set_value(rng.normal(size=(3, 3, 4)).astype(floatX))
    layer.b.set_value(rng.normal(size=(4,)).astype(floatX))
    out = layer(X)
    X_np = rng.normal(size=(8, 32, 3)).astype(floatX)

    assert_close(np.asarray(pytensor.function([X], out, mode="JAX")(X_np)), out.eval({X: X_np}))

    gradient = pt.grad((out**2).sum(), layer.W)
    assert_close(
        np.asarray(pytensor.function([X], gradient, mode="JAX")(X_np)),
        np.asarray(gradient.eval({X: X_np})),
    )


@pytest.mark.parametrize("spatial", [24, None], ids=["static_length", "dynamic_length"])
def test_the_gradient_dispatch_matches_the_graph(spatial, rng):
    """The pullback is its own op so a backend can differentiate its own convolution instead of running
    the gather's scatter-add. Checked against the same gradient on the default backend, which computes
    it from the graph -- two engines differentiating the same forward."""
    X = pt.tensor("X", shape=(4, spatial, 3), dtype=floatX)
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=3)
    layer.W.set_value(rng.normal(size=(3, 3, 5)).astype(floatX))
    layer.b.set_value(rng.normal(size=(5,)).astype(floatX))
    cost = (layer(X) ** 2).sum()
    X_np = rng.normal(size=(4, 24, 3)).astype(floatX)

    for wrt in (layer.W, layer.b, X):
        gradient = pt.grad(cost, wrt)
        reference = pytensor.function([X], gradient)(X_np)
        assert_close(
            np.asarray(pytensor.function([X], gradient, mode="JAX")(X_np)), np.asarray(reference)
        )


@pytest.mark.parametrize("wrt", ["W", "X"], ids=["kernel_only", "input_only"])
def test_a_lone_gradient_dispatches_and_matches_the_graph(wrt, rng):
    """`ConvLayer.pullback` asks for both gradients; the rewrite drops whichever has no clients. That
    reaches a different branch of the dispatch than the pair does -- one that differentiates toward a
    single input and returns a bare array rather than a tuple -- so it is checked for itself."""
    X = pt.tensor("X", shape=(4, 24, 3), dtype=floatX)
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=3)
    layer.W.set_value(rng.normal(size=(3, 3, 5)).astype(floatX))
    layer.b.set_value(rng.normal(size=(5,)).astype(floatX))
    cost = (layer(X) ** 2).sum()
    X_np = rng.normal(size=(4, 24, 3)).astype(floatX)

    gradient = pt.grad(cost, X if wrt == "X" else layer.W)
    fn = pytensor.function([X], gradient, mode="JAX")
    grad_op = next(
        n.op for n in fn.maker.fgraph.apply_nodes if type(n.op).__name__ == "ConvLayerGrad"
    )

    assert (grad_op.compute_dX, grad_op.compute_dW) == (wrt == "X", wrt == "W")
    assert_close(np.asarray(fn(X_np)), np.asarray(pytensor.function([X], gradient)(X_np)))


def test_the_conv_op_matches_the_graph_over_two_spatial_axes(rng):
    """Rank 2 is where a backend's layout convention can disagree with ours about which spatial axis is
    which, and a square kernel at unit stride hides it. Rectangular, with a different stride and
    dilation per axis, is the shape that does not."""
    X_np = rng.normal(size=(2, 12, 14, 3)).astype(floatX)
    W_np = rng.normal(size=(2, 3, 3, 5)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    W = pt.tensor("W", shape=W_np.shape)

    out = ConvLayer((2, 3), (2, 1), (1, 2))(X, W)
    compare_jax_and_py([X, W], out, [X_np, W_np], assert_fn=assert_close)


@pytest.mark.parametrize("stride", [1, 2], ids=["plain", "strided"])
def test_a_transposed_convolutions_gradient_matches_the_graph(stride, rng):
    """A transposed convolution is `ConvLayerGrad` run forward, so its own gradient is what a decoder
    trains on. The pullback has to reach ops this backend dispatches -- the inherited `OpFromGraph` one
    reaches the patch gather, which has no jax kernel and so fails to convert at all. Stride rides
    on the op's props, so the dispatch has to read it back rather than assume it."""
    X_shape = (2, 7, 7, 3)
    W_np = rng.normal(size=(3, 3, 3, 4)).astype(floatX)
    geometry = ((3, 3), (stride, stride), (1, 1))
    cotangent_shape = ConvLayer(*geometry)(
        pt.zeros(X_shape, dtype=floatX), pt.tensor(shape=W_np.shape)
    ).type.shape
    cotangent_np = rng.normal(size=cotangent_shape).astype(floatX)

    W = pt.tensor("W", shape=W_np.shape, dtype=floatX)
    cotangent = pt.tensor("cotangent", shape=cotangent_shape, dtype=floatX)
    out = ConvLayerGrad(*geometry, compute_dW=False)(pt.zeros(X_shape, dtype=floatX), W, cotangent)
    gradients = pt.grad((out**2).sum(), [W, cotangent])

    dispatched = pytensor.function([W, cotangent], gradients, mode="JAX")(W_np, cotangent_np)
    reference = pytensor.function([W, cotangent], gradients)(W_np, cotangent_np)
    for got, expected in zip(dispatched, reference):
        assert_close(np.asarray(got), np.asarray(expected))
