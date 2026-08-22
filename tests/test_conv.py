import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import get_default_mode
from pytensor.gradient import verify_grad
from scipy.signal import correlate

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm, Conv1D, Conv2D, Flatten, Input, Linear, MaxPool2D
from pytensor_ml.layers.conv import (
    Col2Im,
    ConvLayer,
    ConvLayerGrad,
    Im2Col,
    _extract_patches,
    _scatter_patches,
)
from pytensor_ml.loss import SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optim import adam
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import OneInitializer, ZeroInitializer, fans

floatX = pytensor.config.floatX

# The reference sums channel contributions in a different order than the Dot, so the gap tracks
# the precision.
ATOL = 1e-6 if floatX == "float64" else 1e-4


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml conv")))


def test_patches_over_one_spatial_axis_are_the_windows_a_kernel_visits():
    """The windows written out by hand. A sequence 0..5 with width 3 and unit stride visits four of
    them, and each row of the result is the three consecutive values the kernel would multiply."""
    X = pt.tensor("X", shape=(None, None, 1))
    patches = _extract_patches(X, kernel_size=(3,), stride=(1,), dilation=(1,))

    X_np = np.arange(6, dtype=floatX).reshape(1, 6, 1)
    got = patches.eval({X: X_np})

    assert got.shape == (1, 4, 3, 1)
    expected = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=floatX)
    np.testing.assert_array_equal(got[0, :, :, 0], expected)


def test_patches_step_by_the_stride_and_skip_by_the_dilation():
    """Stride moves the window's start, dilation spreads its taps, and the two are independent. Read
    off 0..7: stride 2 starts at 0, 2, 4; dilation 2 takes every other element within a window."""
    X = pt.tensor("X", shape=(None, None, 1))
    X_np = np.arange(8, dtype=floatX).reshape(1, 8, 1)

    strided = _extract_patches(X, kernel_size=(2,), stride=(3,), dilation=(1,)).eval({X: X_np})
    np.testing.assert_array_equal(strided[0, :, :, 0], np.array([[0, 1], [3, 4], [6, 7]]))

    dilated = _extract_patches(X, kernel_size=(3,), stride=(1,), dilation=(2,)).eval({X: X_np})
    np.testing.assert_array_equal(
        dilated[0, :, :, 0], np.array([[0, 2, 4], [1, 3, 5], [2, 4, 6], [3, 5, 7]])
    )


def test_patches_over_two_spatial_axes_pair_every_row_window_with_every_column_window():
    """The 2-D case is the 1-D case on each axis and the product across them, which is what the
    broadcast between the two index arrays has to produce. A 3x3 image with a 2x2 window visits four
    positions, and the corner ones are the top-left and bottom-right 2x2 blocks."""
    X = pt.tensor("X", shape=(None, None, None, 1))
    patches = _extract_patches(X, kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1))

    X_np = np.arange(9, dtype=floatX).reshape(1, 3, 3, 1)
    got = patches.eval({X: X_np})

    assert got.shape == (1, 2, 2, 2, 2, 1)
    np.testing.assert_array_equal(got[0, 0, 0, :, :, 0], np.array([[0, 1], [3, 4]]))
    np.testing.assert_array_equal(got[0, 1, 1, :, :, 0], np.array([[4, 5], [7, 8]]))


def test_patches_take_a_different_extent_per_spatial_axis(rng):
    """Nothing forces the window to be square, and an implementation that broadcast one axis's index
    array over both would agree with a square kernel and disagree here."""
    X = pt.tensor("X", shape=(None, None, None, None))
    patches = _extract_patches(X, kernel_size=(2, 3), stride=(1, 1), dilation=(1, 1))

    X_np = rng.normal(size=(2, 5, 7, 3)).astype(floatX)
    got = patches.eval({X: X_np})

    assert got.shape == (2, 4, 5, 2, 3, 3)
    np.testing.assert_allclose(got[1, 2, 3], X_np[1, 2:4, 3:6], atol=1e-12)


def test_patches_carry_every_batch_and_channel_axis_through(rng):
    """The gather touches the spatial axes alone, so a batch element and a channel are along for the
    ride. Checked against numpy slicing at a window chosen off the diagonal."""
    X = pt.tensor("X", shape=(None, None, None, None))
    patches = _extract_patches(X, kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))

    X_np = rng.normal(size=(4, 9, 9, 5)).astype(floatX)
    got = patches.eval({X: X_np})

    assert got.shape == (4, 4, 4, 3, 3, 5)
    for batch, row, col in ((0, 0, 3), (3, 2, 1)):
        np.testing.assert_allclose(
            got[batch, row, col],
            X_np[batch, 2 * row : 2 * row + 3, 2 * col : 2 * col + 3],
            atol=1e-12,
        )


def correlate_reference(X_np, W_np, stride=1, dilation=1):
    """A convolution written with scipy, one channel pair at a time, as the independent reference.

    ``scipy.signal.correlate`` is not another path through pytensor, so it cannot agree with a bug the
    implementation and a pytensor-based reference would share. It has no notion of stride or dilation,
    so dilation is spelled by zero-stuffing the kernel and stride by subsampling the result -- both
    definitions rather than reimplementations of what the layer does. ``stride`` and ``dilation`` each
    take a scalar or one value per spatial axis.
    """
    *kernel, in_channels, out_channels = W_np.shape
    strides = (stride,) * len(kernel) if isinstance(stride, int) else tuple(stride)
    dilations = (dilation,) * len(kernel) if isinstance(dilation, int) else tuple(dilation)
    spans = tuple(spacing * (extent - 1) + 1 for extent, spacing in zip(kernel, dilations))
    if any(spacing != 1 for spacing in dilations):
        stuffed = np.zeros((*spans, in_channels, out_channels), dtype=W_np.dtype)
        stuffed[
            (*(slice(None, None, spacing) for spacing in dilations), slice(None), slice(None))
        ] = W_np
        W_np = stuffed
    outputs = []
    for image in X_np:
        planes = []
        for out_channel in range(out_channels):
            total = None
            for in_channel in range(in_channels):
                term = correlate(
                    image[..., in_channel], W_np[..., in_channel, out_channel], mode="valid"
                )
                total = term if total is None else total + term
            planes.append(total)
        outputs.append(np.stack(planes, axis=-1))
    assert outputs[0].ndim == len(kernel) + 1
    stacked = np.stack(outputs)
    return stacked[(slice(None), *(slice(None, None, step) for step in strides), slice(None))]


@pytest.mark.parametrize(
    "spatial, kernel_size",
    [((9,), (3,)), ((9, 9), (3, 3)), ((7, 9), (2, 4))],
    ids=["1d", "2d_square", "2d_rectangular"],
)
def test_the_conv_op_correlates_like_scipy(spatial, kernel_size, rng):
    """The load-bearing correctness test. Every channel pair, summed over input channels, against
    scipy's correlation rather than against another pytensor graph."""
    in_channels, out_channels = 3, 4
    X = pt.tensor("X", shape=(None, *(None for _ in spatial), in_channels))
    W = pt.tensor("W", shape=(*kernel_size, in_channels, out_channels))
    op = ConvLayer(
        kernel_size=kernel_size, stride=(1,) * len(spatial), dilation=(1,) * len(spatial)
    )

    X_np = rng.normal(size=(2, *spatial, in_channels)).astype(floatX)
    W_np = rng.normal(size=(*kernel_size, in_channels, out_channels)).astype(floatX)

    np.testing.assert_allclose(
        op(X, W).eval({X: X_np, W: W_np}), correlate_reference(X_np, W_np), atol=ATOL
    )


def test_the_conv_op_does_not_flip_its_kernel(rng):
    """Correlation, not convolution. An asymmetric kernel is the only thing that tells them apart, and
    a flipped implementation would still agree with scipy's ``convolve``, so pin the convention."""
    X = pt.tensor("X", shape=(None, None, 1))
    W = pt.tensor("W", shape=(2, 1, 1))
    op = ConvLayer(kernel_size=(2,), stride=(1,), dilation=(1,))

    X_np = np.array([[[1.0], [0.0], [0.0]]], dtype=floatX)
    W_np = np.array([[[1.0]], [[10.0]]], dtype=floatX)

    # The kernel's first tap lands on the input's first element; flipped, the 10 would land there.
    np.testing.assert_allclose(op(X, W).eval({X: X_np, W: W_np})[0, :, 0], [1.0, 0.0], atol=ATOL)


def test_the_conv_op_adds_an_optional_bias(rng):
    """The bias is a third input rather than a separate Elemwise outside the op, so that a backend
    kernel that fuses it has something to fuse."""
    X = pt.tensor("X", shape=(None, None, 2))
    W = pt.tensor("W", shape=(3, 2, 4))
    b = pt.tensor("b", shape=(4,))
    op = ConvLayer(kernel_size=(3,), stride=(1,), dilation=(1,))

    X_np = rng.normal(size=(2, 8, 2)).astype(floatX)
    W_np = rng.normal(size=(3, 2, 4)).astype(floatX)
    b_np = rng.normal(size=(4,)).astype(floatX)

    without = op(X, W).eval({X: X_np, W: W_np})
    with_bias = op(X, W, b).eval({X: X_np, W: W_np, b: b_np})
    np.testing.assert_allclose(with_bias, without + b_np, atol=ATOL)


def test_the_conv_op_takes_a_gradient_matching_finite_differences(rng):
    """Overlapping windows make the patch gather's pullback a scatter-add, and pytensor will produce a
    gradient for an advanced-indexing expression whether or not the accumulation is right. Checked
    against finite differences in float64, where the step size is meaningful."""
    with pytensor.config.change_flags(floatX="float64"):
        X = pt.tensor("X", shape=(None, None, None, 2), dtype="float64")
        W = pt.tensor("W", shape=(2, 2, 2, 3), dtype="float64")
        op = ConvLayer(kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1))

        X_np = rng.normal(size=(2, 5, 5, 2))
        W_np = rng.normal(size=(2, 2, 2, 3))
        cost = (op(X, W) ** 2).sum()

        for wrt, value in ((X, X_np), (W, W_np)):
            analytic = pt.grad(cost, wrt).eval({X: X_np, W: W_np})
            numeric = np.zeros_like(value)
            step = 1e-6
            flat = numeric.reshape(-1)
            for index in range(flat.size):
                nudged = value.copy().reshape(-1)
                nudged[index] += step
                up = cost.eval({X: X_np, W: W_np} | {wrt: nudged.reshape(value.shape)})
                nudged[index] -= 2 * step
                down = cost.eval({X: X_np, W: W_np} | {wrt: nudged.reshape(value.shape)})
                flat[index] = (up - down) / (2 * step)
            np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_conv1d_correlates_like_scipy_and_adds_its_bias(rng):
    """The layer is padding, parameters and a call. Checked against the reference rather than against
    the op, so a layer that padded the wrong axis or transposed its kernel would show up."""
    X = pt.tensor("X", shape=(None, None, 3))
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=4)
    out = layer(X)

    W_np = rng.normal(size=(4, 3, 5)).astype(floatX)
    b_np = rng.normal(size=(5,)).astype(floatX)
    layer.W.set_value(W_np)
    layer.b.set_value(b_np)
    X_np = rng.normal(size=(2, 11, 3)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), correlate_reference(X_np, W_np) + b_np, atol=ATOL
    )


@pytest.mark.parametrize(
    "kernel_size, padding, expected",
    [(3, "valid", 8), (3, "same", 10), (4, "same", 10), (3, 2, 12)],
    ids=["valid", "same_odd", "same_even", "explicit"],
)
def test_conv1d_pads_to_the_length_it_promises(kernel_size, padding, expected, rng):
    """`same` is the one that has to hold at even kernel sizes, where the padding cannot be symmetric
    and the output length is the thing the name is a claim about."""
    X = pt.tensor("X", shape=(None, None, 2))
    layer = Conv1D("conv", in_channels=2, out_channels=3, kernel_size=kernel_size, padding=padding)

    got = layer(X).eval({X: rng.normal(size=(2, 10, 2)).astype(floatX)})
    assert got.shape == (2, expected, 3)


def test_conv1d_pads_with_the_mode_it_is_given():
    """A padding mode reaches `pt.pad`, so an edge-padded input repeats its boundary rather than
    fading to zero. With a summing kernel over a constant input, zero padding shows at the edges and
    edge padding does not."""
    X = pt.tensor("X", shape=(None, None, 1))
    X_np = np.ones((1, 6, 1), dtype=floatX)
    ones = np.ones((3, 1, 1), dtype=floatX)

    zero_padded = Conv1D("c", 1, 1, 3, padding="same", bias=False)
    zero_padded.W.set_value(ones)
    edge_padded = Conv1D("c", 1, 1, 3, padding="same", padding_mode="edge", bias=False)
    edge_padded.W.set_value(ones)

    np.testing.assert_allclose(
        zero_padded(X).eval({X: X_np})[0, :, 0], [2, 3, 3, 3, 3, 2], atol=ATOL
    )
    np.testing.assert_allclose(
        edge_padded(X).eval({X: X_np})[0, :, 0], [3, 3, 3, 3, 3, 3], atol=ATOL
    )


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_conv1d_bias_is_optional(bias, rng):
    """Dropping the bias drops the parameter as well as the term; an unused one would hand the
    optimizer moment state to carry for a weight that never moves."""
    X = pt.tensor("X", shape=(None, None, 2))
    layer = Conv1D("conv", in_channels=2, out_channels=3, kernel_size=2, bias=bias)
    out = layer(X)

    W_np = rng.normal(size=(2, 2, 3)).astype(floatX)
    layer.W.set_value(W_np)
    shift = np.zeros(3, dtype=floatX)
    if bias:
        shift = rng.normal(size=(3,)).astype(floatX)
        layer.b.set_value(shift)
    X_np = rng.normal(size=(2, 7, 2)).astype(floatX)

    assert set(collect_trainable_params(out)) == ({layer.W, layer.b} if bias else {layer.W})
    np.testing.assert_allclose(
        out.eval({X: X_np}), correlate_reference(X_np, W_np) + shift, atol=ATOL
    )


def test_conv1d_draws_its_kernel_with_the_receptive_field_in_the_fans():
    """A conv kernel is stored input-channels-first so `fans` reads it correctly: fan_in is
    `in_channels * kernel_size`, not two kernel extents. A kernel-first layout would give a draw scaled
    for the wrong fans and nothing else here would notice."""
    layer = Conv1D("conv", in_channels=8, out_channels=16, kernel_size=5)

    assert layer.W.get_value().shape == (5, 8, 16)
    assert fans(layer.W.get_value().shape) == (8 * 5, 16 * 5)
    assert layer.W.get_value().std() == pytest.approx(np.sqrt(2.0 / (40 + 80)), rel=0.15)


def test_conv1d_forwards_its_initializers_to_its_parameters():
    """Two keyword-only arguments, and one dropped on the floor leaves a parameter silently at its
    default draw."""
    layer = Conv1D(
        "conv",
        in_channels=2,
        out_channels=3,
        kernel_size=2,
        weight_initializer=ZeroInitializer(),
        bias_initializer=OneInitializer(),
    )

    np.testing.assert_array_equal(layer.W.get_value(), np.zeros((2, 2, 3)))
    np.testing.assert_array_equal(layer.b.get_value(), np.ones(3))


def test_conv1d_rejects_an_input_of_the_wrong_rank():
    """A 1-D convolution takes (batch, time, channels); handing it a bare sequence is the natural
    mistake and the op would report it from inside the gather."""
    layer = Conv1D("conv", in_channels=2, out_channels=3, kernel_size=2)

    with pytest.raises(ValueError, match="needs a 3-dimensional input; got a 2-dimensional one"):
        layer(pt.tensor("X", shape=(None, 2)))


def test_conv1d_trains_end_to_end(rng):
    """Gradients survive the gather, the matmul and the training machinery, and the parameters move.

    The head pools over time rather than taking the last position, because a `valid` convolution's last
    output sees only the final window -- a target summing the whole sequence would be unlearnable by
    construction, and the test would be measuring nothing."""
    X = Input("X", shape=(None, 12, 2))
    features = Conv1D("conv", in_channels=2, out_channels=4, kernel_size=3, padding="same")(X)
    y = Linear("head", 4, 1)(features.sum(axis=1))
    model = Model(X, y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError(), ndim_out=2)

    X_np = rng.normal(size=(32, 12, 2)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


def test_conv1d_rejects_a_kernel_wider_than_the_input():
    """A window that does not fit yields no windows at all, and every downstream shape then carries a
    zero axis, so the graph computes an empty answer instead of failing. Caught wherever the input's
    length is known when the graph is built."""
    layer = Conv1D("conv", in_channels=1, out_channels=1, kernel_size=10)

    with pytest.raises(ValueError, match="at least 10 elements along spatial axis 0"):
        layer(pt.tensor("X", shape=(None, 5, 1)))

    # Dilation stretches the span, so a kernel that fits undilated need not fit dilated.
    dilated = Conv1D("conv", in_channels=1, out_channels=1, kernel_size=3, dilation=4)
    with pytest.raises(ValueError, match="at least 9 elements"):
        dilated(pt.tensor("X", shape=(None, 8, 1)))

    # Padding is counted, so the same kernel fits once there is enough of it. Checked by evaluating:
    # the gather does not propagate a static spatial size, so the built graph's type says None here.
    padded = Conv1D("conv", in_channels=1, out_channels=1, kernel_size=10, padding=3)
    X = pt.tensor("X", shape=(None, 5, 1))
    assert padded(X).eval({X: np.zeros((2, 5, 1), dtype=floatX)}).shape == (2, 2, 1)


def test_conv1d_rejects_negative_padding():
    """Padding adds elements. A negative amount reaches `pt.pad` as nonsense rather than quietly
    trimming, and the caller almost certainly meant to slice the input."""
    with pytest.raises(ValueError, match="cannot be negative"):
        Conv1D("conv", in_channels=1, out_channels=1, kernel_size=3, padding=-1)


@pytest.mark.parametrize("stride", [1, 2, 3], ids=["stride1", "stride2", "stride3"])
@pytest.mark.parametrize("length", [10, 11], ids=["even", "odd"])
def test_conv1d_same_padding_holds_at_any_stride(stride, length, rng):
    """`same` is a claim about the output length, and torch and keras both define it as
    ceil(input / stride) rather than only holding at unit stride."""
    X = pt.tensor("X", shape=(None, None, 1))
    layer = Conv1D(
        "conv", in_channels=1, out_channels=1, kernel_size=3, stride=stride, padding="same"
    )

    got = layer(X).eval({X: rng.normal(size=(2, length, 1)).astype(floatX)})
    assert got.shape == (2, -(-length // stride), 1)


@pytest.mark.parametrize(
    "stride, dilation",
    [(2, 1), (3, 1), (1, 2), (1, 3), (2, 2)],
    ids=["stride2", "stride3", "dilation2", "dilation3", "both"],
)
def test_conv1d_strides_and_dilates_like_scipy(stride, dilation, rng):
    """Stride and dilation are checked at the helper against hand-written windows, but nothing until
    here checks that they survive the trip from the layer's constructor through the op to the gather.
    A layer that swapped the two, or dropped either, would pass every other test in this file."""
    X = pt.tensor("X", shape=(None, None, 3))
    layer = Conv1D(
        "conv", in_channels=3, out_channels=4, kernel_size=3, stride=stride, dilation=dilation
    )

    W_np = rng.normal(size=(3, 3, 4)).astype(floatX)
    b_np = rng.normal(size=(4,)).astype(floatX)
    layer.W.set_value(W_np)
    layer.b.set_value(b_np)
    X_np = rng.normal(size=(2, 20, 3)).astype(floatX)

    np.testing.assert_allclose(
        layer(X).eval({X: X_np}),
        correlate_reference(X_np, W_np, stride=stride, dilation=dilation) + b_np,
        atol=ATOL,
    )


def test_conv1d_rejects_a_per_axis_argument_of_the_wrong_length():
    """`kernel_size`, `stride` and `dilation` each take a scalar or one value per spatial axis, and
    handing a 1-D convolution a pair is the natural mistake when moving code over from 2-D."""
    with pytest.raises(
        ValueError, match="kernel_size must be an int or one value per spatial axis"
    ):
        Conv1D("conv", in_channels=1, out_channels=1, kernel_size=(3, 3))

    with pytest.raises(ValueError, match="stride must be an int"):
        Conv1D("conv", in_channels=1, out_channels=1, kernel_size=3, stride=(1, 1))


def test_a_kernel_of_one_is_a_linear_layer_applied_per_step(rng):
    """A one-tap convolution is a matrix applied at every position, so it has to agree with `Linear` on
    the same weights. That pins the two conventions together: the kernel's trailing axes are the same
    ``(in, out)`` a weight matrix is, so copying one into the other needs a new axis and nothing else."""
    X = pt.tensor("X", shape=(None, None, 4))
    linear = Linear("dense", 4, 6)
    conv = Conv1D("conv", in_channels=4, out_channels=6, kernel_size=1)

    W_np = rng.normal(size=(4, 6)).astype(floatX)
    b_np = rng.normal(size=(6,)).astype(floatX)
    linear.W.set_value(W_np)
    linear.b.set_value(b_np)
    conv.W.set_value(W_np[None])
    conv.b.set_value(b_np)

    assert fans(conv.W.get_value().shape) == fans(linear.W.get_value().shape)

    X_np = rng.normal(size=(3, 7, 4)).astype(floatX)
    np.testing.assert_allclose(conv(X).eval({X: X_np}), linear(X).eval({X: X_np}), atol=ATOL)


@pytest.mark.parametrize(
    "spatial, kernel_size, stride, dilation",
    [
        ((11,), (3,), (1,), (1,)),
        ((11,), (3,), (2,), (2,)),
        ((7, 9), (2, 3), (2, 1), (1, 2)),
        ((6, 6, 6), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
    ],
    ids=["1d", "1d_strided_dilated", "2d_asymmetric", "3d"],
)
def test_im2col_matches_the_reference_gather_at_every_rank(
    spatial, kernel_size, stride, dilation, rng
):
    """The op stands in for the advanced-indexing gather, so it has to agree with it exactly, at every
    rank and not only the ranks a layer happens to build. The 3-D case is the regression: a dispatch
    that handles some ranks and refuses the rest does not fall back, it fails to compile."""
    X = pt.tensor("X", shape=(2, *spatial, 3))
    X_np = rng.normal(size=(2, *spatial, 3)).astype(floatX)

    reference = _extract_patches(X, kernel_size, stride, dilation).eval({X: X_np})
    got = pytensor.function([X], Im2Col(kernel_size, stride, dilation)(X))(X_np)

    assert got.shape == reference.shape
    np.testing.assert_array_equal(got, reference)


def test_im2col_infers_the_shape_it_produces(rng):
    """`infer_shape` is what lets the rest of the graph reason about the gather without running it, so
    a formula that drifts from `perform` would mis-shape everything downstream and only fail later."""
    X = pt.tensor("X", shape=(None, None, 3))
    patches = Im2Col((3,), (2,), (2,))(X)
    X_np = rng.normal(size=(2, 13, 3)).astype(floatX)

    inferred = pytensor.function([X], patches.shape)(X_np)
    np.testing.assert_array_equal(inferred, pytensor.function([X], patches)(X_np).shape)


@pytest.mark.parametrize(
    "spatial, kernel_size, stride, dilation",
    [
        ((11,), (3,), (1,), (1,)),
        ((11,), (3,), (2,), (2,)),
        ((12,), (3,), (5,), (1,)),
        ((7, 9), (2, 3), (2, 1), (1, 2)),
        ((6, 6, 6), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
    ],
    ids=["1d", "1d_strided_dilated", "1d_untouched_tail", "2d_asymmetric", "3d"],
)
def test_col2im_matches_the_reference_scatter_at_every_rank(
    spatial, kernel_size, stride, dilation, rng
):
    """The op stands in for the `inc_subtensor` scatter, so it has to agree with it exactly. The
    untouched-tail case is the one a scatter can get wrong on its own terms: a stride that overshoots
    leaves positions no window reaches, and those have to stay zero rather than pick up a neighbor."""
    X = pt.tensor("X", shape=(2, *spatial, 3))
    X_np = rng.normal(size=(2, *spatial, 3)).astype(floatX)
    patches_np = pytensor.function([X], Im2Col(kernel_size, stride, dilation)(X))(X_np)

    cotangent = pt.tensor("cotangent", shape=patches_np.shape)
    reference = pytensor.function(
        [cotangent, X],
        _scatter_patches(cotangent, X, kernel_size, stride, dilation),
        on_unused_input="ignore",
    )(patches_np, X_np)
    got = pytensor.function(
        [cotangent], Col2Im(kernel_size, stride, dilation)(cotangent, *spatial)
    )(patches_np)

    assert got.shape == reference.shape
    np.testing.assert_allclose(got, reference, atol=ATOL)


def test_col2im_keeps_a_spatial_extent_it_is_given_statically():
    """The extents arrive as inputs rather than as props, so a known one has to survive as a static
    shape -- otherwise every backward pass loses the shape its forward had."""
    cotangent = pt.tensor("cotangent", shape=(2, 9, 3, 3))
    length = pt.scalar("length", dtype="int64")

    assert Col2Im((3,), (1,), (1,))(cotangent, 11).type.shape == (2, 11, 3)
    assert Col2Im((3,), (1,), (1,))(cotangent, length).type.shape == (2, None, 3)


def test_col2im_gathers_the_cotangent_it_scattered(rng):
    """A scatter-add's pullback is the gather that reverses it, so seeding the output with any
    cotangent has to come back as that cotangent gathered into the windows that reached it."""
    patches = pt.tensor("patches", shape=(2, 9, 3, 3))
    scattered = Col2Im((3,), (1,), (1,))(patches, 11)
    seed = pt.tensor("seed", shape=(2, 11, 3))
    seed_np = rng.normal(size=(2, 11, 3)).astype(floatX)

    pulled_back = pt.grad(cost=None, wrt=patches, known_grads={scattered: seed})
    gathered = Im2Col((3,), (1,), (1,))(seed)

    np.testing.assert_allclose(
        pulled_back.eval({seed: seed_np}), gathered.eval({seed: seed_np}), atol=ATOL
    )


@pytest.mark.parametrize("view", ["transposed", "reversed"])
def test_im2col_accepts_an_input_it_cannot_assume_is_contiguous(view, rng):
    """`TensorType` carries no layout, so a kernel is typed against any layout whatever the data turns
    out to be. One that quietly needs a contiguous buffer does not fall back to `perform` when it does
    not get one -- it fails to compile, for every caller."""
    X_np = rng.normal(size=(2, 3, 11)).astype(floatX)
    X = pt.tensor("X", shape=(2, 3, 11))
    source = X.transpose(0, 2, 1) if view == "transposed" else X[:, :, ::-1].transpose(0, 2, 1)

    got = pytensor.function([X], Im2Col((3,), (1,), (1,))(source))(X_np)
    reference = _extract_patches(source, (3,), (1,), (1,)).eval({X: X_np})

    np.testing.assert_allclose(got, reference, atol=ATOL)


def test_col2im_needs_one_extent_per_spatial_axis():
    """The extents are positional, so a caller passing the wrong number of them would otherwise build
    a node whose rank silently disagrees with the kernel's."""
    patches = pt.tensor("patches", shape=(2, 9, 3, 3))

    with pytest.raises(ValueError, match="needs that many extents"):
        Col2Im((3, 3), (1, 1), (1, 1))(patches, 11)


def test_the_gather_backward_runs_when_the_spatial_extent_is_only_known_at_runtime(rng):
    """`Im2Col.pullback` hands the input's spatial extents to the scatter as ordinary inputs, so an
    extent nobody knows until the graph runs has to arrive as a value rather than as a constant folded
    into the node."""
    X_np = rng.normal(size=(2, 11, 3)).astype(floatX)
    seed_np = rng.normal(size=(2, 9, 3, 3)).astype(floatX)
    seed = pt.tensor("seed", shape=(2, 9, 3, 3))

    gradients = []
    for spatial in (None, 11):
        X = pt.tensor("X", shape=(2, spatial, 3))
        patches = Im2Col((3,), (1,), (1,))(X)
        pulled_back = pt.grad(cost=None, wrt=X, known_grads={patches: seed})
        gradients.append(pytensor.function([X, seed], pulled_back)(X_np, seed_np))

    np.testing.assert_allclose(*gradients, atol=ATOL)


def test_conv2d_correlates_like_scipy_and_adds_its_bias(rng):
    """The layer at rank 2, against the reference rather than against the op. A rectangular kernel is
    what separates the two spatial axes: a layer that transposed them would still pass at rank 1 and
    with a square kernel."""
    X = pt.tensor("X", shape=(None, None, None, 3))
    layer = Conv2D("conv", in_channels=3, out_channels=5, kernel_size=(2, 4))

    W_np = rng.normal(size=(2, 4, 3, 5)).astype(floatX)
    b_np = rng.normal(size=(5,)).astype(floatX)
    layer.W.set_value(W_np)
    layer.b.set_value(b_np)
    X_np = rng.normal(size=(2, 9, 11, 3)).astype(floatX)

    np.testing.assert_allclose(
        layer(X).eval({X: X_np}), correlate_reference(X_np, W_np) + b_np, atol=ATOL
    )


@pytest.mark.parametrize(
    "stride, dilation",
    [((2, 1), (1, 1)), ((1, 3), (1, 1)), ((1, 1), (2, 1)), ((1, 1), (1, 3)), ((3, 2), (1, 2))],
    ids=["stride_h", "stride_w", "dilate_h", "dilate_w", "both_asymmetric"],
)
def test_conv2d_strides_and_dilates_per_axis_like_scipy(stride, dilation, rng):
    """Each axis carries its own stride and dilation, and only an asymmetric setting can catch the two
    being swapped or one of them broadcast over both axes."""
    X = pt.tensor("X", shape=(None, None, None, 3))
    layer = Conv2D(
        "conv",
        in_channels=3,
        out_channels=4,
        kernel_size=(2, 3),
        stride=stride,
        dilation=dilation,
        bias=False,
    )

    W_np = rng.normal(size=(2, 3, 3, 4)).astype(floatX)
    layer.W.set_value(W_np)
    X_np = rng.normal(size=(2, 14, 16, 3)).astype(floatX)

    np.testing.assert_allclose(
        layer(X).eval({X: X_np}),
        correlate_reference(X_np, W_np, stride=stride, dilation=dilation),
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "kernel_size, spatial, expected",
    [
        ((3, 3), (8, 10), (8, 10)),
        ((2, 4), (8, 10), (8, 10)),
        ((2, 5), (7, 7), (7, 7)),
    ],
    ids=["both_odd", "both_even", "mixed_parity"],
)
def test_conv2d_same_padding_holds_each_axis_independently(kernel_size, spatial, expected, rng):
    """`same` has no symmetric answer at an even extent, and rank 2 is where one axis can be even while
    the other is odd -- the case that catches a padding amount computed once and reused."""
    X = pt.tensor("X", shape=(None, None, None, 3))
    layer = Conv2D("conv", in_channels=3, out_channels=2, kernel_size=kernel_size, padding="same")
    X_np = rng.normal(size=(2, *spatial, 3)).astype(floatX)

    assert layer(X).eval({X: X_np}).shape == (2, *expected, 2)


def test_conv1d_matches_a_conv2d_whose_second_axis_is_degenerate(rng):
    """Both layers are `_ConvNd` with a different `n_spatial`, and this is what pins that: a 2-D
    convolution over a width-1 image with a width-1 kernel is a 1-D convolution, so the two have to
    agree to the last bit rather than merely have matching shapes."""
    W_np = rng.normal(size=(3, 4, 5)).astype(floatX)
    b_np = rng.normal(size=(5,)).astype(floatX)
    X_np = rng.normal(size=(2, 12, 4)).astype(floatX)

    one_d = Conv1D("one_d", in_channels=4, out_channels=5, kernel_size=3, stride=2)
    one_d.W.set_value(W_np)
    one_d.b.set_value(b_np)

    two_d = Conv2D("two_d", in_channels=4, out_channels=5, kernel_size=(3, 1), stride=(2, 1))
    two_d.W.set_value(W_np[:, None])
    two_d.b.set_value(b_np)

    X1 = pt.tensor("X1", shape=(None, None, 4))
    X2 = pt.tensor("X2", shape=(None, None, None, 4))
    np.testing.assert_allclose(
        one_d(X1).eval({X1: X_np}),
        two_d(X2).eval({X2: X_np[:, :, None]})[:, :, 0],
        atol=ATOL,
    )


def test_conv2d_draws_its_kernel_with_both_extents_in_the_fans():
    """`fans` takes the two trailing dimensions as the features and everything before them as the
    receptive field, so at rank 2 the field is the product of both extents."""
    layer = Conv2D("conv", in_channels=8, out_channels=16, kernel_size=(3, 5))

    assert layer.W.get_value().shape == (3, 5, 8, 16)
    assert fans(layer.W.get_value().shape) == (8 * 15, 16 * 15)


def test_conv2d_same_padding_puts_the_odd_element_after_on_each_axis():
    """At an even extent `same` cannot pad symmetrically, and the extra element goes after rather than
    before. Nothing about the output shape distinguishes the two, so only the values pin it -- and a
    mixed-parity kernel puts the asymmetry on one axis while the other stays symmetric."""
    X = pt.tensor("X", shape=(None, None, None, 1))
    X_np = np.ones((1, 4, 4, 1), dtype=floatX)

    layer = Conv2D(
        "conv", in_channels=1, out_channels=1, kernel_size=(2, 3), padding="same", bias=False
    )
    layer.W.set_value(np.ones((2, 3, 1, 1), dtype=floatX))

    # Summing over a constant input counts how many real elements each window saw. Height pads (0, 1),
    # so only the last row is short; width pads (1, 1), so the first and last columns are.
    np.testing.assert_allclose(
        layer(X).eval({X: X_np})[0, :, :, 0],
        [[4, 6, 6, 4], [4, 6, 6, 4], [4, 6, 6, 4], [2, 3, 3, 2]],
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "spatial, expected",
    [((28, 28), (32, 24, 24, 16)), ((None, 28), (32, None, 24, 16))],
    ids=["static", "one_axis_dynamic"],
)
def test_a_conv_stack_keeps_the_output_shape_it_can_work_out(spatial, expected):
    """A layer downstream of a convolution has to size itself from the graph, so the extents the input
    does know have to survive the op. They reach the output only through reshapes that can fold their
    targets, which a product over a shape slice cannot."""
    X = pt.tensor("X", shape=(32, *spatial, 3))
    first = Conv2D("c1", in_channels=3, out_channels=8, kernel_size=3)(X)
    second = Conv2D("c2", in_channels=8, out_channels=16, kernel_size=3)(first)

    assert second.type.shape == expected


def test_a_convolutional_network_trains_end_to_end(rng):
    """The step the whole plan is for: convolution, pooling, spatial batch norm and a dense head, in
    one graph, learning. Each piece is tested alone elsewhere -- this is the only test that says they
    compose, and that gradients survive every op boundary between them."""
    X = Input("X", shape=(None, 8, 8, 2))
    features = Conv2D("conv", in_channels=2, out_channels=4, kernel_size=3, padding="same")(X)
    normalized = BatchNorm("norm", n_in=4)(ReLU()(features))
    pooled = MaxPool2D("pool", kernel_size=2)(normalized)
    y = Linear("head", 4 * 4 * 4, 1)(Flatten(pooled))

    model = Model(X, y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError(), ndim_out=2)

    X_np = rng.normal(size=(32, 8, 8, 2)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2, 3))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


@pytest.mark.parametrize(
    "wanted, expected",
    [("dX", (True, False)), ("dW", (False, True)), ("both", (True, True))],
    ids=["input_only", "kernel_only", "both"],
)
def test_the_pullback_computes_only_the_gradients_something_reads(wanted, expected, rng):
    """`ConvLayer.pullback` asks for both gradients because only the graph knows which are wanted, and
    only once it is built. A rewrite then drops whichever has no clients -- the input gradient for the
    first convolution of a network, the kernel gradient for a transposed one -- and the numbers it
    leaves behind have to be the ones the un-rewritten graph would have produced."""
    X = pt.tensor("X", shape=(4, 24, 3))
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=3)
    layer.W.set_value(rng.normal(size=(3, 3, 5)).astype(floatX))
    layer.b.set_value(rng.normal(size=(5,)).astype(floatX))
    cost = (layer(X) ** 2).sum()
    targets = {
        "dX": [pt.grad(cost, X)],
        "dW": [pt.grad(cost, layer.W)],
        "both": pt.grad(cost, [X, layer.W]),
    }[wanted]
    X_np = rng.normal(size=(4, 24, 3)).astype(floatX)

    fn = pytensor.function([X], targets)
    grad_op = next(
        node.op for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, ConvLayerGrad)
    )
    assert (grad_op.compute_dX, grad_op.compute_dW) == expected

    unrewritten = get_default_mode().excluding("drop_unused_input_grad", "drop_unused_kernel_grad")
    for dropped, kept in zip(pytensor.function([X], targets, mode=unrewritten)(X_np), fn(X_np)):
        np.testing.assert_allclose(kept, dropped, atol=ATOL)


@pytest.mark.parametrize("dropped", ["dX", "dW"], ids=["without_dX", "without_dW"])
def test_dropping_one_gradient_leaves_the_other_unchanged(dropped, rng):
    """The rewrite is only safe if a lowered op returns the same numbers as the pair it replaces, so
    this compares them directly rather than trusting that fewer outputs means the same arithmetic."""
    X_np = rng.normal(size=(2, 6, 6, 3)).astype(floatX)
    W_np = rng.normal(size=(3, 3, 3, 4)).astype(floatX)
    V_np = rng.normal(size=(2, 4, 4, 4)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    W = pt.tensor("W", shape=W_np.shape)
    cotangent = pt.tensor("cotangent", shape=V_np.shape)

    geometry = ((3, 3), (1, 1), (1, 1))
    keeping_dX = dropped == "dW"
    both = ConvLayerGrad(*geometry)(X, W, cotangent)
    alone = ConvLayerGrad(*geometry, compute_dX=keeping_dX, compute_dW=not keeping_dX)(
        X, W, cotangent
    )

    kept = both[0] if keeping_dX else both[1]
    values = pytensor.function([X, W, cotangent], [kept, alone])(X_np, W_np, V_np)
    np.testing.assert_allclose(*values, atol=ATOL)


def test_the_pullback_must_return_some_gradient():
    """Both flags false describes an op with no outputs, which would build and then fail somewhere
    downstream rather than where the mistake was made."""
    with pytest.raises(ValueError, match="must return at least one gradient"):
        ConvLayerGrad((3,), (1,), (1,), compute_dX=False, compute_dW=False)


@pytest.mark.parametrize(
    "flags",
    [{"compute_dW": False}, {"compute_dX": False}, {}],
    ids=["dX_only", "dW_only", "both"],
)
def test_differentiating_a_conv_gradient_leaves_only_dispatchable_ops(flags):
    """A transposed convolution is `ConvLayerGrad` run forward, so training one differentiates through
    it. Inheriting `OpFromGraph`'s pullback would wrap the gather in an anonymous op carrying no props
    and registered against no type, which every backend but numba refuses outright. Each combination of
    flags builds a different set of terms, so each is checked."""
    geometry = ((3, 3), (1, 1), (1, 1))
    X = pt.tensor("X", shape=(2, 6, 6, 3))
    W = pt.tensor("W", shape=(3, 3, 3, 4))
    cotangent = pt.tensor("cotangent", shape=(2, 4, 4, 4))

    outputs = ConvLayerGrad(*geometry, **flags)(X, W, cotangent, return_list=True)
    cost = sum((out**2).sum() for out in outputs)
    gradients = pt.grad(cost, [X, W, cotangent], disconnected_inputs="ignore")
    fn = pytensor.function([X, W, cotangent], gradients)

    convolutions = {ConvLayer, ConvLayerGrad}
    leftover = [
        node.op
        for node in fn.maker.fgraph.apply_nodes
        if isinstance(node.op, OpFromGraph) and type(node.op) not in convolutions
    ]
    assert not leftover, f"undispatchable ops survived the pullback: {leftover}"


@pytest.mark.parametrize(
    "flags",
    [{"compute_dW": False}, {"compute_dX": False}, {}],
    ids=["dX_only", "dW_only", "both"],
)
@pytest.mark.parametrize(
    "stride, dilation", [(1, 1), (2, 1), (1, 2)], ids=["plain", "strided", "dilated"]
)
def test_the_pullback_of_the_pullback_matches_finite_differences(flags, stride, dilation, rng):
    """The closed forms the pullback uses are adjoint identities rather than a differentiated graph,
    so they are checked numerically. Every input is perturbed, including the one each output
    ignores."""
    geometry = ((3, 3), (stride, stride), (dilation, dilation))
    X_np = rng.normal(size=(2, 7, 7, 3)).astype(floatX)
    W_np = rng.normal(size=(3, 3, 3, 4)).astype(floatX)
    cotangent_shape = ConvLayer(*geometry)(
        pt.zeros(X_np.shape, dtype=floatX), pt.tensor(shape=W_np.shape)
    ).type.shape
    cotangent_np = rng.normal(size=cotangent_shape).astype(floatX)
    op = ConvLayerGrad(*geometry, **flags)

    def summed_outputs(X, W, cotangent):
        return sum((out**2).sum() for out in op(X, W, cotangent, return_list=True))

    verify_grad(summed_outputs, [X_np, W_np, cotangent_np], rng=np.random.default_rng(0))
