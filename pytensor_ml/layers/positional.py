from typing import Literal, get_args

import pytensor.tensor as pt

from pytensor.tensor.type import float_dtypes
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import PositionalLayer, UnaryLayerOp

Pairing = Literal["half", "adjacent"]
Scaling = Literal["none", "linear", "ntk"]

_PAIRINGS = get_args(Pairing)
_SCALINGS = get_args(Scaling)


def _validate_options(pairing: str, scaling: str, scaling_factor: float) -> None:
    """Reject unknown options where they are given, so a typo cannot silently select the default
    behavior inside the inner graph."""
    if pairing not in _PAIRINGS:
        raise ValueError(f"pairing must be one of {_PAIRINGS}, got {pairing!r}")
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {_SCALINGS}, got {scaling!r}")
    if scaling != "none" and scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")


def _head_dim(x: TensorVariable) -> int | TensorVariable:
    """Size of the rotated feature axis, as a Python int when ``x`` declares one.

    A static size lets the frequency ladder fold to a constant and keeps the output's static shape;
    a symbolic one works too, so it is a preference rather than a requirement.
    """
    if x.type.dtype not in float_dtypes:
        raise ValueError(f"RotaryEmbedding needs a floating-point input, got dtype {x.type.dtype}.")

    static_size = x.type.shape[-1]
    if static_size is None:
        return x.shape[-1]
    if static_size % 2:
        raise ValueError(
            f"RotaryEmbedding rotates channel pairs, so head_dim must be even, got {static_size}."
        )

    return static_size


def _add_head_axes(
    angles: TensorVariable, x: TensorVariable, position_ids: TensorVariable
) -> TensorVariable:
    """Unsqueeze ``angles`` so its sequence axis lines up with ``x``'s.

    ``position_ids`` indexes tokens, so its last axis is the sequence and any leading axes are batch
    axes. ``x`` carries head axes between the two, and the number of them follows from the two ranks --
    which is why the caller does not pass an axis to unsqueeze.
    """
    n_head_axes = (x.type.ndim - 1) - position_ids.type.ndim
    if n_head_axes < 0:
        raise ValueError(
            f"position_ids has {position_ids.type.ndim} dimensions, more than the "
            f"{x.type.ndim - 1} non-feature dimensions of x."
        )
    if n_head_axes == 0:
        return angles

    return angles[(Ellipsis, *(None,) * n_head_axes, slice(None), slice(None))]


def _inverse_frequencies(
    head_dim: int | TensorVariable, dtype: str, base: float, scaling: str, scaling_factor: float
) -> TensorVariable:
    r"""
    Angular frequencies :math:`\theta_i = \mathrm{base}^{-2i/d}`, one per rotated pair.

    Returns
    -------
    inverse_frequencies : TensorVariable
        Shape ``(head_dim // 2,)``, dtype ``dtype``. Folds to a constant when ``head_dim`` is static.
    """
    if scaling == "ntk":
        if isinstance(head_dim, int) and head_dim <= 2:
            raise ValueError(
                f"NTK scaling rescales the base by scaling_factor ** (d / (d - 2)), which is "
                f"undefined for head_dim <= 2; got {head_dim}."
            )
        # Stretches the frequency ladder itself rather than the positions, leaving the highest
        # frequencies nearly untouched. Only the static form is offered: the dynamic variant rescales
        # the base from the running sequence length, which is not known at graph-build time.
        base = base * scaling_factor ** (head_dim / (head_dim - 2))

    exponent = pt.arange(0, head_dim, 2, dtype=dtype) / head_dim
    inverse_frequencies = base**-exponent

    if scaling == "linear":
        # Dividing the frequencies is algebraically the same as dividing the positions, since the
        # angle is bilinear in the two.
        inverse_frequencies = inverse_frequencies / scaling_factor

    inverse_frequencies = inverse_frequencies.astype(dtype)
    inverse_frequencies.name = "inverse_frequencies"

    return inverse_frequencies


def _split_pairs(
    x: TensorVariable, pairing: str, head_dim: int | TensorVariable
) -> tuple[TensorVariable, TensorVariable]:
    """Split the feature axis into the two members of every rotated pair.

    The conventions are a permutation of the feature axis apart and are not interchangeable: weights
    trained under one produce nonsense under the other.

    ``"half"`` pairs channel ``i`` with ``i + d/2``, as HuggingFace's ``rotate_half`` does.
    ``"adjacent"`` pairs ``2i`` with ``2i + 1``, as the original RoFormer paper and torchtune do. Both
    are cited from :func:`rotary_embedding`.
    """
    if pairing == "half":
        half = head_dim // 2
        return x[..., :half], x[..., half:]

    return x[..., 0::2], x[..., 1::2]


def _join_pairs(
    x: TensorVariable, first: TensorVariable, second: TensorVariable, pairing: str
) -> TensorVariable:
    """Reassemble the feature axis, inverting :func:`_split_pairs` for the same ``pairing``."""
    if pairing == "half":
        return pt.concatenate([first, second], axis=-1)

    # Every channel is overwritten -- the even ones by `first`, the odd ones by `second` -- and both
    # were read before either write, so nothing of x survives into the result.
    return x[..., 0::2].set(first)[..., 1::2].set(second)


class RotaryEmbeddingLayer(UnaryLayerOp):
    __props__ = ("base", "pairing", "scaling", "scaling_factor")

    def build_inner_graph(self, x, position_ids):
        _validate_options(self.pairing, self.scaling, self.scaling_factor)
        head_dim = _head_dim(x)
        dtype = x.type.dtype

        inverse_frequencies = _inverse_frequencies(
            head_dim, dtype, self.base, self.scaling, self.scaling_factor
        )
        angles = position_ids[..., None].astype(dtype) * inverse_frequencies
        angles = _add_head_axes(angles, x, position_ids)
        cos, sin = pt.cos(angles), pt.sin(angles)

        first, second = _split_pairs(x, self.pairing, head_dim)
        rotated = _join_pairs(
            x, first * cos - second * sin, second * cos + first * sin, self.pairing
        )
        rotated.name = "rotary_embedding"

        return [rotated]


def rotary_embedding(
    x: pt.TensorLike,
    position_ids: pt.TensorLike,
    *,
    base: float = 10_000.0,
    pairing: Pairing = "half",
    scaling: Scaling = "none",
    scaling_factor: float = 1.0,
) -> TensorVariable:
    r"""
    Rotary position embedding (RoPE) applied to the trailing feature axis.

    Rotate each two-dimensional subspace of the feature axis by an angle proportional to the token's
    position:

    .. math::

        \begin{pmatrix} x'_a \\ x'_b \end{pmatrix} =
        \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\
                        \sin m\theta_i & \cos m\theta_i \end{pmatrix}
        \begin{pmatrix} x_a \\ x_b \end{pmatrix},
        \qquad \theta_i = \mathrm{base}^{-2i/d},

    where :math:`m` is the position and :math:`(a, b)` is the :math:`i`-th channel pair [1]_. The
    rotation is orthogonal, so the dot product of a rotated query and a rotated key depends only on
    the difference of their positions. That is what makes it usable for incremental decoding: a token
    rotated by its absolute position keeps the same relationship to every earlier token however late
    it is computed.

    Apply to queries and keys before
    :func:`~pytensor_ml.layers.attention.scaled_dot_product_attention`, never to values.

    Parameters
    ----------
    x : TensorLike
        Tensor whose last axis is rotated, typically queries or keys of shape
        ``(..., n_head, seq, head_dim)``. ``head_dim`` must be even.
    position_ids : TensorLike
        Integer position of each token, shape ``(..., seq)``. ``x``'s head axes are unsqueezed in, so
        ``(seq,)`` and ``(batch, seq)`` both align with ``(batch, n_head, seq, head_dim)``. Explicit
        rather than an implied ``0..seq-1``, so one graph serves a full sequence and a single decode
        step appended to a cached prefix.
    base : float, optional
        Geometric base of the frequency ladder. Default 10000.0.
    pairing : str, optional
        Which channels form each rotated pair, ``"half"`` (default) or ``"adjacent"``. Must match the
        convention the weights were trained with; see :func:`_split_pairs`.
    scaling : str, optional
        Context-extension scheme: ``"none"`` (default), ``"linear"`` for position interpolation [2]_,
        or ``"ntk"`` for static NTK-aware scaling.
    scaling_factor : float, optional
        Extension factor for the scaled variants, ignored when ``scaling="none"``. Default 1.0, the
        identity for both.

    Returns
    -------
    rotated : TensorVariable
        ``x`` with its last axis rotated, same shape and dtype.

    References
    ----------
    .. [1] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
           Transformer with Rotary Position Embedding. arXiv:2104.09864. https://arxiv.org/abs/2104.09864.
    .. [2] Chen, S., Wong, S., Chen, L., & Tian, Y. (2023). Extending Context Window of Large Language
           Models via Positional Interpolation. arXiv:2306.15595. https://arxiv.org/abs/2306.15595.
    .. [3] https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py#L109-L113
    .. [4] https://github.com/meta-pytorch/torchtune/blob/v0.6.1/torchtune/modules/position_embeddings.py#L99-L113
    """
    _validate_options(pairing, scaling, scaling_factor)

    x = pt.as_tensor(x)
    position_ids = pt.as_tensor(position_ids)

    rotated = RotaryEmbeddingLayer(
        name="RotaryEmbedding",
        base=base,
        pairing=pairing,
        scaling=scaling,
        scaling_factor=scaling_factor,
    )(x, position_ids)
    rotated.name = "rotary_embedding_output"

    return rotated


class RotaryEmbedding(PositionalLayer):
    r"""
    Rotary position embeddings as a configured layer.

    Holds the frequency configuration so queries and keys are rotated identically -- they must be,
    since attention compares them. The layer has no parameters of its own.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's output. Defaults to "RotaryEmbedding" when None.
    base : float, optional
        Geometric base of the frequency ladder. Default 10000.0.
    pairing : str, optional
        ``"half"`` (default) or ``"adjacent"``.
    scaling : str, optional
        ``"none"`` (default), ``"linear"``, or ``"ntk"``.
    scaling_factor : float, optional
        Extension factor for the scaled variants. Default 1.0.

    References
    ----------
    .. [1] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
           Transformer with Rotary Position Embedding. arXiv:2104.09864. https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self,
        name: str | None = None,
        base: float = 10_000.0,
        pairing: Pairing = "half",
        scaling: Scaling = "none",
        scaling_factor: float = 1.0,
    ):
        _validate_options(pairing, scaling, scaling_factor)

        self.name = name if name else "RotaryEmbedding"
        self.base = base
        self.pairing = pairing
        self.scaling = scaling
        self.scaling_factor = scaling_factor

    def __call__(self, x: pt.TensorLike, position_ids: pt.TensorLike) -> TensorVariable:
        rotated = rotary_embedding(
            x,
            position_ids,
            base=self.base,
            pairing=self.pairing,
            scaling=self.scaling,
            scaling_factor=self.scaling_factor,
        )
        rotated.name = f"{self.name}_output"

        return rotated


__all__ = [
    "RotaryEmbedding",
    "rotary_embedding",
]
