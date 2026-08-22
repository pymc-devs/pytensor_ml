from collections.abc import Sequence
from functools import reduce
from operator import add, mul

import numpy as np
import pytensor.tensor as pt

from numpy.lib.stride_tricks import sliding_window_view
from pytensor.gradient import disconnected_type
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.pad import PadMode
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import Layer, LayerOp, UnaryLayerOp
from pytensor_ml.params import trainable_parameter
from pytensor_ml.state import Initializer, XavierNormalInitializer, ZeroInitializer


def _max_over_taps(patches: TensorVariable, axis: int) -> TensorVariable:
    """
    Take each window's largest tap, routing the whole gradient to the one tap that won it.

    Reducing with :func:`pytensor.tensor.max` would instead give the full cotangent to *every* tap tied
    for the maximum, which returns more gradient than the window received -- and a window of ties is
    routine rather than exotic, since max pooling usually follows a rectifier that clamps whole windows
    to zero. Selecting through an ``argmax`` picks one tap, and conserves the gradient the way every
    backend's own pooling does.
    """
    winner = pt.argmax(patches, axis=axis)
    pattern: list[int | str] = ["x"] * patches.ndim
    pattern[axis] = 0
    taps = pt.arange(patches.shape[axis]).dimshuffle(*pattern)
    # Selected with `switch` rather than multiplied by the mask: an unselected tap may be the -inf a
    # padded position carries, and -inf times zero is nan rather than nothing.
    chosen = pt.switch(pt.eq(pt.expand_dims(winner, axis), taps), patches, 0)
    return chosen.sum(axis=axis)


# Each reduction and the value that pads without changing it: -inf loses every comparison to a real
# element, and zero is what an average counts padded positions as, matching torch's count_include_pad.
_REDUCTIONS = {"max": _max_over_taps, "mean": pt.mean}
_PADDING_IDENTITY = {"max": -np.inf, "mean": 0.0}


def _check_reduction(reduction: str) -> None:
    """Reject a reduction neither pooling op knows, where a lookup would otherwise raise a KeyError."""
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {sorted(_REDUCTIONS)}, but got {reduction!r}.")


def _window_span(extent: int, spacing: int) -> int:
    """How far one window reaches along an axis, once dilation has spread its taps."""
    return spacing * (extent - 1) + 1


def _window_indices(
    X: TensorVariable, kernel_size: Sequence[int], stride: Sequence[int], dilation: Sequence[int]
) -> list[TensorVariable]:
    """One advanced index per spatial axis, carrying that axis's windows and its taps.

    Each broadcasts against the others, so indexing with all of them at once gives windows-then-taps in
    order. Shared by the gather's reference graph and by the scatter that reverses it.
    """
    n_spatial = len(kernel_size)
    indices = []
    for axis, (extent, step, spacing) in enumerate(zip(kernel_size, stride, dilation)):
        span = _window_span(extent, spacing)
        starts = pt.arange(0, X.shape[1 + axis] - span + 1, step)
        window = starts[:, None] + pt.arange(extent)[None, :] * spacing

        pattern: list[int | str] = ["x"] * (2 * n_spatial)
        pattern[axis] = 0
        pattern[n_spatial + axis] = 1
        indices.append(window.dimshuffle(*pattern))
    return indices


def _scatter_patches(
    cotangent: TensorVariable,
    X: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> TensorVariable:
    """Add each window's cotangent back at the position it was gathered from.

    Windows overlap, so a position reached by several of them accumulates all of their contributions --
    which is why this is a scatter-add rather than an assignment.
    """
    indices = _window_indices(X, kernel_size, stride, dilation)
    zeros = pt.zeros(X.shape, dtype=cotangent.dtype)
    return pt.inc_subtensor(zeros[(slice(None), *indices, slice(None))], cotangent)


class Im2Col(Op):
    """
    Gather every window a kernel visits, as one node a backend can put a real copy behind.

    The equivalent advanced-indexing graph does scalar index arithmetic per element; the copy this
    describes is one contiguous channel-row per ``(batch, window, tap)``, which is the difference between
    a few GB/s and memory bandwidth. It sits inside :class:`ConvLayer`'s inner graph, so a backend that
    dispatches the convolution itself never reaches it.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        One entry per spatial axis.
    """

    __props__ = ("kernel_size", "stride", "dilation")

    def __init__(
        self,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        dilation: Sequence[int],
    ):
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.dilation = tuple(dilation)

    def __call__(self, *inputs, **kwargs) -> TensorVariable:
        """Narrow the single output to a tensor, which ``Op.__call__`` cannot know it is."""
        out = super().__call__(*inputs, **kwargs)
        assert isinstance(out, TensorVariable), "Im2Col produces one output"
        return out

    def make_node(self, X):
        X = pt.as_tensor(X)
        n_spatial = len(self.kernel_size)
        spatial = [
            None
            if X.type.shape[1 + axis] is None
            else (
                X.type.shape[1 + axis] - _window_span(self.kernel_size[axis], self.dilation[axis])
            )
            // self.stride[axis]
            + 1
            for axis in range(n_spatial)
        ]
        out_type = pt.tensor(
            dtype=X.type.dtype,
            shape=(X.type.shape[0], *spatial, *self.kernel_size, X.type.shape[-1]),
        )
        return Apply(self, [X], [out_type])

    def perform(self, node, inputs, outputs):
        (X,) = inputs
        n_spatial = len(self.kernel_size)
        spans = tuple(
            _window_span(self.kernel_size[axis], self.dilation[axis]) for axis in range(n_spatial)
        )

        # (batch, *windows, channels, *spans), a view over X with nothing copied yet
        view = sliding_window_view(X, spans, axis=tuple(range(1, 1 + n_spatial)))
        view = view[
            (
                slice(None),
                *(slice(None, None, step) for step in self.stride),
                slice(None),
                *(slice(None, None, spacing) for spacing in self.dilation),
            )
        ]
        # channels trail the taps in our layout, so move that axis past them and make it contiguous
        order = (
            0,
            *range(1, 1 + n_spatial),
            *range(2 + n_spatial, 2 + 2 * n_spatial),
            1 + n_spatial,
        )
        outputs[0][0] = np.ascontiguousarray(view.transpose(order))

    def infer_shape(self, node, input_shapes):
        [(batch, *spatial, channels)] = input_shapes
        windows = [
            (spatial[axis] - _window_span(self.kernel_size[axis], self.dilation[axis]))
            // self.stride[axis]
            + 1
            for axis in range(len(self.kernel_size))
        ]
        return [[batch, *windows, *self.kernel_size, channels]]

    def pullback(self, inputs, outputs, cotangents):
        """Scatter each window's cotangent back where it was gathered from."""
        (X,) = inputs
        (cotangent,) = cotangents
        spatial = [X.shape[1 + axis] for axis in range(len(self.kernel_size))]
        return [Col2Im(self.kernel_size, self.stride, self.dilation)(cotangent, *spatial)]


class Col2Im(Op):
    """
    Add every window's contribution back at the position it was gathered from.

    The pullback of :class:`Im2Col`, and like it one node a backend can put a real kernel behind rather
    than an indexing graph. Windows overlap, so a position several of them reached accumulates all of
    their contributions, which is what makes this a scatter-add rather than an assignment.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        One entry per spatial axis, describing the gather being reversed.
    """

    __props__ = ("kernel_size", "stride", "dilation")

    def __init__(
        self,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        dilation: Sequence[int],
    ):
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.dilation = tuple(dilation)

    def __call__(self, *inputs, **kwargs) -> TensorVariable:
        """Narrow the single output to a tensor, which ``Op.__call__`` cannot know it is."""
        out = super().__call__(*inputs, **kwargs)
        assert isinstance(out, TensorVariable), "Col2Im produces one output"
        return out

    def make_node(self, patches, *spatial):
        """Take the spatial extents one scalar at a time, so a known one stays known statically."""
        if len(spatial) != len(self.kernel_size):
            raise ValueError(
                f"Col2Im over {len(self.kernel_size)} spatial axes needs that many extents, got "
                f"{len(spatial)}"
            )
        patches = pt.as_tensor(patches)
        spatial = [pt.as_tensor(length, dtype="int64") for length in spatial]

        lengths = [
            get_scalar_constant_value(length, raise_not_constant=False) for length in spatial
        ]
        out_type = pt.tensor(
            dtype=patches.type.dtype,
            shape=(
                patches.type.shape[0],
                *(None if isinstance(length, Variable) else int(length) for length in lengths),
                patches.type.shape[-1],
            ),
        )
        return Apply(self, [patches, *spatial], [out_type])

    def perform(self, node, inputs, outputs):
        patches, *lengths = inputs
        spatial = tuple(int(length) for length in lengths)
        n_spatial = len(self.kernel_size)

        # One index array per spatial axis, broadcasting to windows-then-taps as the gather's do
        indices = []
        for axis in range(n_spatial):
            span = _window_span(self.kernel_size[axis], self.dilation[axis])
            starts = np.arange(0, spatial[axis] - span + 1, self.stride[axis])
            window = starts[:, None] + np.arange(self.kernel_size[axis]) * self.dilation[axis]
            pattern = [1] * (2 * n_spatial)
            pattern[axis], pattern[n_spatial + axis] = window.shape
            indices.append(window.reshape(pattern))

        out = np.zeros((patches.shape[0], *spatial, patches.shape[-1]), dtype=patches.dtype)
        np.add.at(out, (slice(None), *indices, slice(None)), patches)
        outputs[0][0] = out

    def infer_shape(self, node, input_shapes):
        [(batch, *_, channels)] = input_shapes
        return [[batch, *node.inputs[1:], channels]]

    def connection_pattern(self, node):
        """Only the patches reach the output; the spatial extents give it its shape."""
        return [[True], *([False] for _ in self.kernel_size)]

    def pullback(self, inputs, outputs, cotangents):
        """Gather each position's cotangent into every window that reached it."""
        (cotangent,) = cotangents
        gathered = Im2Col(self.kernel_size, self.stride, self.dilation)(cotangent)
        return [gathered, *(disconnected_type() for _ in self.kernel_size)]


def _extract_patches(
    X: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> TensorVariable:
    """
    Gather every window a kernel visits, for an input with any number of spatial axes.

    Parameters
    ----------
    X : TensorVariable
        Input of shape ``(batch, *spatial, channels)``.
    kernel_size : sequence of int
        Window extent along each spatial axis. Its length is the number of spatial axes.
    stride : sequence of int
        Step between windows along each spatial axis.
    dilation : sequence of int
        Spacing between the taps of one window along each spatial axis.

    Returns
    -------
    TensorVariable
        Shape ``(batch, *out_spatial, *kernel_size, channels)``, where ``out_spatial`` counts the
        windows that fit.
    """
    indices = _window_indices(X, kernel_size, stride, dilation)
    return X[(slice(None), *indices, slice(None))]


class ConvLayer(UnaryLayerOp):
    """
    Cross-correlate an input with a kernel, over any number of spatial axes.

    Marks the convolution as one node so a backend with a real kernel can be dispatched to it, the way
    :class:`~pytensor_ml.layers.attention.AttentionLayer` is. The graph inside gathers every window and
    contracts them against the kernel with a single ``Dot``, whose reduction runs over
    ``prod(kernel_size) * in_channels`` -- deep enough to reach BLAS, which is what a convolution over
    single-channel signal ops cannot offer.

    Correlation, not convolution: the kernel is not flipped, matching torch, keras and flax.
    """

    __props__ = ("kernel_size", "stride", "dilation")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        super().__init__(**kwargs)

    def build_inner_graph(self, X, W, *bias):
        """
        Correlate ``X`` of shape ``(batch, *spatial, in_channels)`` with ``W`` of shape
        ``(*kernel_size, in_channels, out_channels)``, optionally adding ``bias``.
        """
        out = _correlate(X, W, self.kernel_size, self.stride, self.dilation)
        if bias:
            out = out + bias[0]
        return [out]

    def pullback(self, inputs, outputs, cotangents):
        """
        Differentiate through one :class:`ConvLayerGrad` rather than through the inner graph.

        The default would inline the pullback into an anonymous ``OpFromGraph``, which no backend can
        dispatch against, so the whole backward pass would run as the gather and its scatter whatever
        kernels were available. ``compute_dX`` starts True and a rewrite lowers it where the input
        gradient turns out to be unused; see :func:`~pytensor_ml.rewriting.conv.drop_unused_input_grad`.
        """
        X, W, *bias = inputs
        (cotangent,) = cotangents

        dX, dW = ConvLayerGrad(self.kernel_size, self.stride, self.dilation, compute_dX=True)(
            X, W, cotangent
        )
        if not bias:
            return [dX, dW]
        # One bias per output channel, so its gradient sums the cotangent over everything else.
        summed = cotangent.sum(axis=tuple(range(cotangent.ndim - 1)))
        return [dX, dW, summed]


class ConvLayerGrad(LayerOp):
    """
    The pullback of :class:`ConvLayer`, as a node a backend can dispatch against.

    Both gradients are convolutions in their own right -- the input's is a full convolution of the
    cotangent with the flipped kernel, the kernel's a correlation of the input with the cotangent -- so a
    backend with a convolution kernel has one for each. The dispatches reach them by differentiating
    that backend's own convolution rather than by spelling either out.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        The forward convolution being differentiated.
    compute_dX, compute_dW : bool, optional
        Which gradients to return, in that order. Dropping one is right wherever nothing consumes it:
        the first convolution of a network needs no input gradient, and a transposed convolution needs
        no kernel gradient. At least one must be asked for. Both default to ``True``.
    """

    __props__ = ("kernel_size", "stride", "dilation", "compute_dX", "compute_dW")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        compute_dX: bool = True,
        compute_dW: bool = True,
        **kwargs,
    ):
        if not (compute_dX or compute_dW):
            raise ValueError(
                "ConvLayerGrad must return at least one gradient, but both compute_dX and compute_dW "
                "are False."
            )
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.compute_dX = compute_dX
        self.compute_dW = compute_dW
        super().__init__(**kwargs)

    def build_inner_graph(self, X, W, cotangent):
        """Differentiate the forward correlation, which is what every dispatch also does."""
        out = _correlate(X, W, self.kernel_size, self.stride, self.dilation)
        wrt = [
            variable for variable, wanted in ((X, self.compute_dX), (W, self.compute_dW)) if wanted
        ]
        return list(pt.grad(cost=None, wrt=wrt, known_grads={out: cotangent}))

    def pullback(self, inputs, outputs, cotangents):
        r"""
        Differentiate through convolutions rather than through the inner graph.

        The default would inline an anonymous ``OpFromGraph`` around the gather, which no backend can
        dispatch against and which only numba can run at all. Each output is linear in two of the three
        inputs and the adjoint of a correlation is the other output, so writing :math:`C` for the
        forward correlation, :math:`T` for the input gradient and :math:`K` for the kernel gradient,

        .. math::

            \langle T(V, W), G \rangle = \langle V, C(G, W) \rangle = \langle W, K(G, V) \rangle

            \langle K(X, V), H \rangle = \langle V, C(X, H) \rangle = \langle X, T(V, H) \rangle

        Each gradient is therefore a :class:`ConvLayer` or a one-sided :class:`ConvLayerGrad`. The input
        gradient does not read ``X`` and the kernel gradient does not read ``W``, so each is
        disconnected from the input the other consumes when its own output is not computed.
        """
        X, W, cotangent = inputs
        geometry = (self.kernel_size, self.stride, self.dilation)
        # One cotangent arrives per output, in output order, so the flags that chose the outputs
        # also say which cotangent belongs to which gradient.
        arriving = iter(cotangents)
        dX_bar = next(arriving) if self.compute_dX else None
        dW_bar = next(arriving) if self.compute_dW else None

        # The cotangent is the one input both outputs read, so its gradient collects a term from each.
        cotangent_terms = []
        if dX_bar is not None:
            cotangent_terms.append(ConvLayer(*geometry)(dX_bar, W))
        if dW_bar is not None:
            cotangent_terms.append(ConvLayer(*geometry)(X, dW_bar))

        dX = (
            disconnected_type()
            if dW_bar is None
            else ConvLayerGrad(*geometry, compute_dW=False)(X, dW_bar, cotangent)
        )
        dW = (
            disconnected_type()
            if dX_bar is None
            else ConvLayerGrad(*geometry, compute_dX=False)(dX_bar, W, cotangent)
        )
        return [dX, dW, reduce(add, cotangent_terms)]


def _correlate(
    X: TensorVariable,
    W: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> TensorVariable:
    """Cross-correlate ``(batch, *spatial, in_channels)`` with ``(*kernel_size, in_channels, out_channels)``."""
    patches = Im2Col(kernel_size, stride, dilation)(X)
    n_spatial = len(kernel_size)
    # Both reshape targets are taken an axis at a time rather than from a slice of the shape vector.
    # `Shape_i` folds to a constant wherever the extent is known statically, a sliced `Subtensor` does
    # not, and `Reshape` keeps a static output shape only for the entries it can fold.
    kept = [patches.shape[axis] for axis in range(1 + n_spatial)]
    contracted = reduce(mul, (patches.shape[axis] for axis in range(1 + n_spatial, patches.ndim)))

    # Collapsed to a 2-D matmul rather than contracting the 4-D patch tensor directly, because `pt.grad`
    # reads this graph as written: leaving the batch and window axes in place makes the kernel's gradient
    # a batched contraction, one matmul per window row over an intermediate larger than the patch buffer.
    flat = patches.reshape((-1, contracted))
    out = flat @ W.reshape((-1, W.shape[-1]))
    return out.reshape((*kept, W.shape[-1]))


class PoolLayer(UnaryLayerOp):
    """
    Reduce each window a kernel would visit, instead of contracting it against one.

    The same windows :class:`ConvLayer` correlates over, with ``max`` or ``mean`` in place of the matmul,
    so a backend that gathers windows well serves both.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        One entry per spatial axis.
    reduction : {"max", "mean"}
        Which reduction each window collapses to.
    """

    __props__ = ("kernel_size", "stride", "dilation", "reduction")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        reduction: str,
        **kwargs,
    ):
        _check_reduction(reduction)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.reduction = reduction
        super().__init__(**kwargs)

    def build_inner_graph(self, X):
        """Collapse the tap axes of ``(batch, *out_spatial, *kernel_size, channels)``."""
        return [_pool(X, self.kernel_size, self.stride, self.dilation, self.reduction)]

    def pullback(self, inputs, outputs, cotangents):
        """
        Differentiate through one :class:`PoolLayerGrad` rather than through the inner graph.

        The default would wrap the pullback in an anonymous ``OpFromGraph``, which registers against no
        type, so the backward pass would gather every window and reduce it again on every backend.
        """
        (X,) = inputs
        (cotangent,) = cotangents
        grad_op = PoolLayerGrad(self.kernel_size, self.stride, self.dilation, self.reduction)
        return [grad_op(X, cotangent)]


class PoolLayerGrad(UnaryLayerOp):
    """
    The pullback of :class:`PoolLayer`, as a node a backend can dispatch against.

    A max pool routes each window's cotangent to the position that won it and a mean pool splits it
    evenly, so a backend with its own pooling primitive has both by differentiating that primitive rather
    than by spelling either out.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        The forward pooling being differentiated.
    reduction : {"max", "mean"}
        Which reduction the forward collapsed each window to.
    """

    __props__ = ("kernel_size", "stride", "dilation", "reduction")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        reduction: str,
        **kwargs,
    ):
        _check_reduction(reduction)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.reduction = reduction
        super().__init__(**kwargs)

    def build_inner_graph(self, X, cotangent):
        """Differentiate the forward reduction, which is what every dispatch also does."""
        out = _pool(X, self.kernel_size, self.stride, self.dilation, self.reduction)
        [gradient] = pt.grad(cost=None, wrt=[X], known_grads={out: cotangent})
        return [gradient]


def _pool(
    X: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    reduction: str,
) -> TensorVariable:
    """Reduce every window of ``(batch, *spatial, channels)`` to one value per channel."""
    patches = Im2Col(kernel_size, stride, dilation)(X)
    n_spatial = len(kernel_size)
    kept = [patches.shape[axis] for axis in range(1 + n_spatial)]
    # One tap axis rather than one per spatial dimension, so a single reduction covers the window.
    merged = patches.reshape((*kept, -1, patches.shape[-1]))
    return _REDUCTIONS[reduction](merged, axis=1 + n_spatial)


def _resolve_padding(
    padding: str | int | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """
    Turn a padding argument into an explicit ``(before, after)`` pair per spatial axis.

    ``"same"`` pads by the kernel's span less one, which leaves the output spatial size at
    ``ceil(input / stride)`` -- the input's own size at unit stride. That total is odd whenever the span
    is even, and the extra element goes after rather than before, matching torch and keras.

    Parameters
    ----------
    padding : {"valid", "same"}, int, or sequence of int
        No padding, enough to leave the output at ``ceil(input / stride)``, or an explicit amount
        applied to both sides of every axis or of each axis in turn.
    kernel_size : sequence of int
        Window extent along each spatial axis.
    dilation : sequence of int
        Spacing between taps, which stretches the span the padding has to cover.
    """
    if padding == "valid":
        return tuple((0, 0) for _ in kernel_size)
    if padding == "same":
        pads = []
        for extent, spacing in zip(kernel_size, dilation):
            total = spacing * (extent - 1)
            pads.append((total // 2, total - total // 2))
        return tuple(pads)
    if isinstance(padding, str):
        raise ValueError(
            f"padding must be 'valid', 'same', or an explicit number of elements, but got {padding!r}."
        )

    amounts = [padding] * len(kernel_size) if isinstance(padding, int) else list(padding)
    if any(amount < 0 for amount in amounts):
        raise ValueError(
            f"Padding adds elements, so it cannot be negative; got {padding!r}. To use less of the "
            "input than it has, take a slice of it before convolving."
        )
    if len(amounts) != len(kernel_size):
        raise ValueError(
            f"A convolution over {len(kernel_size)} spatial axes needs one padding amount per axis, but "
            f"got {len(amounts)}."
        )
    return tuple((amount, amount) for amount in amounts)


def _pad_spatial(
    X: TensorVariable,
    padding: tuple[tuple[int, int], ...],
    mode: PadMode,
    constant_value: float = 0.0,
) -> TensorVariable:
    """Pad the spatial axes of ``(batch, *spatial, channels)``, leaving batch and channels alone."""
    if not any(before or after for before, after in padding):
        return X
    pad_width = [(0, 0), *padding, (0, 0)]
    # Spelled out per branch rather than unpacked from a dict: only `constant` takes a fill value, and
    # `pad` is overloaded per mode, so a `**kwargs` call cannot be matched against those overloads.
    if mode == "constant":
        return pt.pad(X, pad_width, mode="constant", constant_values=constant_value)
    return pt.pad(X, pad_width, mode=mode)


def _as_spatial_tuple(value: int | Sequence[int], n_spatial: int, name: str) -> tuple[int, ...]:
    """Broadcast a scalar argument across the spatial axes, or check one already given per axis."""
    if isinstance(value, int):
        return (value,) * n_spatial
    given = tuple(value)
    if len(given) != n_spatial:
        raise ValueError(
            f"{name} must be an int or one value per spatial axis, but got {len(given)} values for "
            f"{n_spatial} spatial axes."
        )
    return given


def _check_input_rank(X: TensorVariable, name: str, n_spatial: int) -> None:
    """Reject an input whose rank is not batch, one axis per spatial dimension, then channels."""
    if X.ndim != n_spatial + 2:
        raise ValueError(
            f"{name} takes an input of shape (batch, {', '.join(['spatial'] * n_spatial)}, channels), "
            f"so it needs a {n_spatial + 2}-dimensional input; got a {X.ndim}-dimensional one."
        )


def _check_input_covers_a_window(
    X: TensorVariable,
    name: str,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> None:
    """
    Reject a padded input too short for one window, where its length is known at build time.

    Such an input yields no windows at all, so every downstream shape has a zero axis and the graph
    computes an empty answer rather than failing. Only a statically known spatial size can be checked; a
    symbolic one still reaches the loop and comes back empty.
    """
    for axis, (extent, spacing, (before, after)) in enumerate(zip(kernel_size, dilation, padding)):
        length = X.type.shape[1 + axis]
        if length is None:
            continue
        span = _window_span(extent, spacing)
        if length + before + after < span:
            raise ValueError(
                f"{name} needs at least {span} elements along spatial axis {axis} to place one window, "
                f"but the input has {length} there and the padding adds {before + after}. Shorten the "
                "kernel, lower the dilation, or pad more."
            )


class _ConvNd(Layer):
    """
    Everything a convolution does that does not depend on how many spatial axes it has.

    Subclasses set :attr:`n_spatial`; see :class:`Conv1D` for the arguments, which are shared.
    """

    n_spatial: int

    def __init__(
        self,
        name: str | None,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] = "valid",
        *,
        padding_mode: PadMode = "constant",
        bias: bool = True,
        weight_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = name if name else type(self).__name__
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _as_spatial_tuple(kernel_size, self.n_spatial, "kernel_size")
        self.stride = _as_spatial_tuple(stride, self.n_spatial, "stride")
        self.dilation = _as_spatial_tuple(dilation, self.n_spatial, "dilation")
        self.padding = _resolve_padding(padding, self.kernel_size, self.dilation)
        self.padding_mode: PadMode = padding_mode
        self.bias = bias

        # Receptive field, then input channels, then output: the layout flax and keras use, and the one
        # `fans` reads, since it takes the two trailing dimensions as the features.
        self.W = trainable_parameter(
            f"{self.name}_W",
            (*self.kernel_size, in_channels, out_channels),
            weight_initializer,
            XavierNormalInitializer(),
        )
        if bias:
            self.b = trainable_parameter(
                f"{self.name}_b", (out_channels,), bias_initializer, ZeroInitializer()
            )

    def __call__(self, X: pt.TensorLike) -> TensorVariable:
        """
        Correlate ``X`` of shape ``(batch, *spatial, in_channels)`` with the layer's kernel.

        Returns
        -------
        TensorVariable
            Shape ``(batch, *out_spatial, out_channels)``.
        """
        X = pt.as_tensor(X)
        _check_input_rank(X, self.name, self.n_spatial)
        _check_input_covers_a_window(X, self.name, self.kernel_size, self.dilation, self.padding)

        padded = _pad_spatial(X, self.padding, self.padding_mode)
        op = ConvLayer(self.kernel_size, self.stride, self.dilation)
        parameters = (self.W, self.b) if self.bias else (self.W,)

        out = op(padded, *parameters)
        out.name = f"{self.name}_output"
        return out


class Conv1D(_ConvNd):
    r"""
    Cross-correlate a sequence with a learned kernel, over one spatial axis.

    Takes ``(batch, time, in_channels)`` and returns ``(batch, out_time, out_channels)``, so it stacks
    with the recurrent layers without a transpose between them. Each output position is the kernel
    contracted against the window of the input beneath it:

    .. math::

        y_{t,o} = b_o + \sum_{c} \sum_{j} x_{t s + j d,\,c} \, W_{j,c,o},

    with :math:`s` the stride and :math:`d` the dilation. The kernel is not flipped, so this is
    correlation in the signal-processing sense and convolution in the sense every ML framework means.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to the class name when None.
    in_channels : int
        Size of the input's channel axis.
    out_channels : int
        Size of the output's channel axis, one per kernel.
    kernel_size : int
        Window extent along the time axis.
    stride : int, optional
        Step between windows. Default is 1.
    dilation : int, optional
        Spacing between the kernel's taps, which widens the receptive field without adding parameters.
        Default is 1.
    padding : {"valid", "same"} or int, optional
        No padding, enough to leave the output length equal to :math:`\lceil \text{time} / s \rceil`, or
        an explicit number of elements on each side. Default is "valid".
    padding_mode : str, optional
        How padded elements are filled, passed to :func:`pytensor.tensor.pad`. Default is "constant",
        which pads with zeros; "reflect", "symmetric" and "edge" are the other useful ones.
    bias : bool, optional
        Add the learned shift :math:`b`, one per output channel. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W` is drawn. Xavier normal when omitted, whose fans count the receptive field.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.
    """

    n_spatial = 1


class Conv2D(_ConvNd):
    r"""
    Cross-correlate an image with a learned kernel, over two spatial axes.

    Takes ``(batch, height, width, in_channels)`` and returns ``(batch, out_height, out_width,
    out_channels)``, keeping the channels-last layout the rest of the library uses. Each output position
    is the kernel contracted against the window of the input beneath it:

    .. math::

        y_{h,w,o} = b_o + \sum_{c} \sum_{i,j}
            x_{h s_0 + i d_0,\; w s_1 + j d_1,\; c} \, W_{i,j,c,o},

    with :math:`s` the stride and :math:`d` the dilation, one of each per axis. The kernel is not
    flipped, so this is correlation in the signal-processing sense and convolution in the sense every
    ML framework means.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to the class name when None.
    in_channels : int
        Size of the input's channel axis.
    out_channels : int
        Size of the output's channel axis, one per kernel.
    kernel_size : int or tuple of int
        Window extent, shared by both axes or given as ``(height, width)``.
    stride : int or tuple of int, optional
        Step between windows, shared or per axis. Default is 1.
    dilation : int or tuple of int, optional
        Spacing between the kernel's taps, which widens the receptive field without adding parameters.
        Shared or per axis. Default is 1.
    padding : {"valid", "same"}, int, or tuple of int, optional
        No padding, enough to leave each output extent at :math:`\lceil \text{extent} / s \rceil`, or
        an explicit number of elements on each side, shared or per axis. Default is "valid".
    padding_mode : str, optional
        How padded elements are filled, passed to :func:`pytensor.tensor.pad`. Default is "constant",
        which pads with zeros; "reflect", "symmetric" and "edge" are the other useful ones.
    bias : bool, optional
        Add the learned shift :math:`b`, one per output channel. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W` is drawn. Xavier normal when omitted, whose fans count the receptive field.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.
    """

    n_spatial = 2


class _PoolNd(Layer):
    """
    Everything pooling does that does not depend on how many spatial axes it has.

    Subclasses set :attr:`n_spatial` and :attr:`reduction`; see :class:`MaxPool2D` for the arguments,
    which are shared.
    """

    n_spatial: int
    reduction: str

    def __init__(
        self,
        name: str | None = None,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] | None = None,
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] = "valid",
    ):
        self.name = name if name else type(self).__name__
        self.kernel_size = _as_spatial_tuple(kernel_size, self.n_spatial, "kernel_size")
        self.stride = (
            self.kernel_size
            if stride is None
            else _as_spatial_tuple(stride, self.n_spatial, "stride")
        )
        self.dilation = _as_spatial_tuple(dilation, self.n_spatial, "dilation")
        self.padding = _resolve_padding(padding, self.kernel_size, self.dilation)

    def __call__(self, X: pt.TensorLike) -> TensorVariable:
        """
        Reduce each window of ``X``, of shape ``(batch, *spatial, channels)``.

        Returns
        -------
        TensorVariable
            Shape ``(batch, *out_spatial, channels)``, with the channel axis untouched.
        """
        X = pt.as_tensor(X)
        _check_input_rank(X, self.name, self.n_spatial)
        _check_input_covers_a_window(X, self.name, self.kernel_size, self.dilation, self.padding)

        padded = _pad_spatial(X, self.padding, "constant", _PADDING_IDENTITY[self.reduction])
        out = PoolLayer(self.kernel_size, self.stride, self.dilation, self.reduction)(padded)
        out.name = f"{self.name}_output"
        return out


class MaxPool2D(_PoolNd):
    """
    Downsample by taking the largest activation in each window, over two spatial axes.

    Takes ``(batch, height, width, channels)`` and returns ``(batch, out_height, out_width, channels)``.
    Padding fills with :math:`-\\infty` so a padded position never wins a window, which zero-filling
    would do wherever every real activation is negative. Where a window ties, the whole gradient goes
    to the earliest tap; a backend that pools with its own kernel may instead split the gradient
    between the tied taps, but every backend returns exactly the gradient the window received.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer. Defaults to the class name when None.
    kernel_size : int or tuple of int, optional
        Window extent, shared by both axes or given as ``(height, width)``. Default is 2.
    stride : int or tuple of int, optional
        Step between windows. Defaults to ``kernel_size``, so windows tile without overlapping --
        the opposite of the convolution layers, which step by one.
    dilation : int or tuple of int, optional
        Spacing between the positions a window covers. Default is 1.
    padding : {"valid", "same"}, int, or tuple of int, optional
        No padding, enough to leave each output extent at :math:`\\lceil \text{extent} / s \rceil`, or
        an explicit number of elements on each side. Default is "valid".
    """

    n_spatial = 2
    reduction = "max"


class MaxPool1D(_PoolNd):
    """Downsample a sequence by taking the largest activation in each window; see :class:`MaxPool2D`."""

    n_spatial = 1
    reduction = "max"


class AvgPool2D(_PoolNd):
    """
    Downsample by averaging each window, over two spatial axes.

    Takes ``(batch, height, width, channels)`` and returns ``(batch, out_height, out_width, channels)``.
    Padded positions count toward the average as zeros, matching torch's ``count_include_pad``. See
    :class:`MaxPool2D` for the arguments, which are shared.
    """

    n_spatial = 2
    reduction = "mean"


class AvgPool1D(_PoolNd):
    """Downsample a sequence by averaging each window; see :class:`AvgPool2D`."""

    n_spatial = 1
    reduction = "mean"
