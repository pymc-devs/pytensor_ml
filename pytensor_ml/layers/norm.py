import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.base import Layer, LayerOp, UnaryLayerOp
from pytensor_ml.params import NonTrainableParameter, TrainableParameter, non_trainable, trainable


def _standardize(X, epsilon, axis, keepdims=False):
    """
    Center and scale ``X`` over ``axis``, returning the statistics used.

    Notes
    -----
    The variance is biased (``ddof=0``), matching :class:`torch.nn.BatchNorm1d` and
    :class:`torch.nn.LayerNorm`; pretrained weights assume it. Do not "correct" it to the unbiased
    estimator -- that would diverge from every pretrained model.

    Returns
    -------
    X_standardized : TensorVariable
        ``X`` centered and scaled to unit variance.
    mu : TensorVariable
        Mean of ``X`` over ``axis``.
    sigma_sq : TensorVariable
        Biased variance of ``X`` over ``axis``.
    """
    mu = X.mean(axis=axis, keepdims=keepdims)
    sigma_sq = X.var(axis=axis, keepdims=keepdims)
    return (X - mu) / pt.sqrt(sigma_sq + epsilon), mu, sigma_sq


def _affine_input_count(affine: bool) -> int:
    """Number of inputs the learned affine transform contributes, which every norm op places directly
    after ``X``. Both the graph builders and :meth:`BatchNormLayer.update_map` index around it."""
    return 2 if affine else 0


def _split_affine(affine: bool, rest: tuple) -> tuple[tuple, tuple]:
    """
    Split an op's inputs after ``X`` into the affine pair and whatever that op expects behind it.

    Returns
    -------
    affine_params : tuple
        ``(loc, scale)``, or empty when the layer has no affine transform.
    extras : tuple
        The remaining inputs, such as batch norm's running statistics.
    """
    n_affine = _affine_input_count(affine)
    return rest[:n_affine], rest[n_affine:]


def _rescale(X_normalized, affine_params: tuple):
    if not affine_params:
        return X_normalized

    loc, scale = affine_params
    return X_normalized * scale + loc


def _resolve_n_in(name: str, n_in: int | None, X: pt.TensorVariable | None) -> int | None:
    """Resolve the feature-axis size, or ``None`` while the layer is still waiting for an input to
    infer it from."""
    if X is None:
        return n_in

    inferred = X.type.shape[-1]
    if inferred is None:
        raise ValueError(
            f"{name} cannot infer n_in from an input whose last dimension is unknown. Pass n_in "
            f"explicitly, or give the input a static feature dimension."
        )
    return inferred


def _affine_parameters(name: str, n_in: int) -> tuple[TrainableParameter, TrainableParameter]:
    """Build the learned shift and scale. Returns them in the ``(loc, scale)`` order that every norm
    op unpacks its inputs in, so the two cannot drift apart."""
    loc = trainable(np.zeros(n_in, dtype=config.floatX), f"{name}_loc")
    scale = trainable(np.ones(n_in, dtype=config.floatX), f"{name}_scale")
    return loc, scale


class BatchNormLayer(LayerOp):
    __props__ = ("n_in", "epsilon", "momentum", "affine")

    def update_map(self):
        # Outputs 1 and 2 are the new running mean and variance; they write back to the running
        # statistics they were computed from, which follow X and the affine pair.
        running_mean_index = 1 + _affine_input_count(self.affine)
        return {1: running_mean_index, 2: running_mean_index + 1}

    def build_inner_graph(self, X, *rest):
        affine_params, (running_mean, running_var) = _split_affine(self.affine, rest)
        X_normalized, mu, sigma_sq = _standardize(X, self.epsilon, axis=0)

        new_running_mean = self.momentum * mu + (1 - self.momentum) * running_mean
        new_running_var = self.momentum * sigma_sq + (1 - self.momentum) * running_var

        return [_rescale(X_normalized, affine_params), new_running_mean, new_running_var]


class NoRunningStatsBatchNormLayer(LayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        # Reports the batch statistics to match BatchNormLayer's arity; declaring no update_map is what
        # keeps them from being written back to anything.
        affine_params, _ = _split_affine(self.affine, rest)
        X_normalized, mu, sigma_sq = _standardize(X, self.epsilon, axis=0)

        return [_rescale(X_normalized, affine_params), mu, sigma_sq]


class PredictionBatchNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        affine_params, (running_mean, running_var) = _split_affine(self.affine, rest)
        X_normalized = (X - running_mean) / pt.sqrt(running_var + self.epsilon)

        return [_rescale(X_normalized, affine_params)]


class BatchNorm2D(Layer):
    r"""
    Batch normalization over the batch axis.

    Standardize each feature across the batch, then optionally apply a learned affine transform:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta,

    where the mean and (biased) variance are taken over the batch (first) axis. During training the
    batch statistics are used and the running mean and variance are updated toward them as
    :math:`(1 - m)\,r + m\,b` from each batch statistic :math:`b`.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Default is "BatchNorm".
    n_in : int, optional
        Size of the feature axis. Inferred from the input's last dimension on the first call when
        omitted.
    epsilon : float, optional
        Constant :math:`\epsilon` added to the variance for numerical stability. Default is 1e-5.
    momentum : float, optional
        Weight :math:`m` of the current batch statistic in the running-average update. Default is
        0.1.
    affine : bool, optional
        Apply the learned scale :math:`\gamma` and shift :math:`\beta`. Default is True.
    track_running_stats : bool, optional
        Maintain running mean and variance for use at prediction time. Default is True.

    Notes
    -----
    Batch normalization is not symmetric between training and prediction, so inference needs
    special handling. A plain forward pass -- calling the layer, or compiling with
    :func:`function` -- normalizes with the *current batch's* mean and variance, making each
    output depend on the other samples in the batch. That is what you want while training, but
    wrong at inference, where an example must normalize the same way no matter what it happens to
    be batched with. Compile prediction graphs with :func:`compile_predict`, which applies
    :func:`rewrite_for_prediction` to substitute the accumulated running statistics for the batch
    statistics.
    """

    def __init__(
        self,
        name: str | None = None,
        n_in: int | None = None,
        epsilon: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        self.name = name if name else "BatchNorm"
        self.n_in = n_in
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.scale: TrainableParameter | None = None
        self.loc: TrainableParameter | None = None
        self.running_mean: NonTrainableParameter | None = None
        self.running_var: NonTrainableParameter | None = None
        self.new_running_mean: pt.TensorVariable | None = None
        self.new_running_var: pt.TensorVariable | None = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorVariable | None):
        if self.initialized:
            return

        n_in = _resolve_n_in(self.name, self.n_in, X)
        if n_in is None:
            return

        if self.affine:
            self.loc, self.scale = _affine_parameters(self.name, n_in)

        if self.track_running_stats:
            zeros = np.zeros(n_in, dtype=config.floatX)
            ones = np.ones(n_in, dtype=config.floatX)
            self.running_mean = non_trainable(zeros, f"{self.name}_running_mean")
            self.running_var = non_trainable(ones, f"{self.name}_running_var")

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        inputs = [X]

        self._initialize_params(X)

        if self.affine:
            assert self.scale is not None and self.loc is not None
            inputs.extend([self.loc, self.scale])

        if self.track_running_stats:
            assert self.running_mean is not None and self.running_var is not None
            inputs.extend([self.running_mean, self.running_var])
            batch_norm_op: LayerOp = BatchNormLayer(
                name=self.name,
                n_in=self.n_in,
                epsilon=self.epsilon,
                momentum=self.momentum,
                affine=self.affine,
            )
        else:
            batch_norm_op = NoRunningStatsBatchNormLayer(
                name=self.name,
                n_in=self.n_in,
                epsilon=self.epsilon,
                affine=self.affine,
            )

        X_transformed, batch_mean, batch_var = batch_norm_op(*inputs)
        if self.track_running_stats:
            self.new_running_mean, self.new_running_var = batch_mean, batch_var

        X_transformed.name = f"{self.name}_output"

        return X_transformed


class LayerNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        affine_params, _ = _split_affine(self.affine, rest)
        X_normalized, _, _ = _standardize(X, self.epsilon, axis=-1, keepdims=True)

        return [_rescale(X_normalized, affine_params)]


class LayerNorm(Layer):
    r"""
    Layer normalization over the last (feature) axis.

    Standardize each sample independently across its features, then optionally apply a learned
    affine transform:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta,

    where the mean and (biased) variance are taken over the last axis. The statistics depend only on
    the current sample, so there are no running statistics and no train/eval distinction.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Default is "LayerNorm".
    n_in : int, optional
        Size of the normalized feature axis. Inferred from the input's last dimension on the first
        call when omitted.
    epsilon : float, optional
        Constant :math:`\epsilon` added to the variance for numerical stability. Default is 1e-5.
    affine : bool, optional
        Apply the learned scale :math:`\gamma` and shift :math:`\beta`. Default is True.
    """

    def __init__(
        self,
        name: str | None = None,
        n_in: int | None = None,
        epsilon: float = 1e-5,
        affine: bool = True,
    ):
        self.name = name if name else "LayerNorm"
        self.n_in = n_in
        self.epsilon = epsilon
        self.affine = affine

        self.scale: TrainableParameter | None = None
        self.loc: TrainableParameter | None = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorVariable | None):
        if self.initialized:
            return

        n_in = _resolve_n_in(self.name, self.n_in, X)
        if n_in is None:
            return

        if self.affine:
            self.loc, self.scale = _affine_parameters(self.name, n_in)

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        self._initialize_params(X)

        inputs = [X]
        if self.affine:
            assert self.scale is not None and self.loc is not None
            inputs.extend([self.loc, self.scale])

        X_transformed = LayerNormLayer(
            name=self.name,
            n_in=self.n_in,
            epsilon=self.epsilon,
            affine=self.affine,
        )(*inputs)
        X_transformed.name = f"{self.name}_output"

        return X_transformed


class RMSNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        epsilon = pt.constant(self.epsilon, dtype=X.type.dtype)
        mean_square = pt.mean(pt.square(X), axis=-1, keepdims=True)
        X_normalized = X / pt.sqrt(mean_square + epsilon)
        if not self.affine:
            return [X_normalized]

        (scale,) = rest
        return [X_normalized * scale]


class RMSNorm(Layer):
    r"""
    Root-mean-square layer normalization over the last (feature) axis.

    Divide each sample by the root mean square of its own features, then optionally apply a learned
    scale:

    .. math::

        y = \frac{x}{\sqrt{\frac{1}{n} \sum_i x_i^2 + \epsilon}} \cdot \gamma.

    Unlike :class:`LayerNorm` there is no mean subtraction and no learned shift; the scale is the only
    parameter. This is the normalization used by the Llama, Gemma and Qwen decoder families.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Default is "RMSNorm".
    n_in : int, optional
        Size of the normalized feature axis. Inferred from the input's last dimension on the first
        call when omitted.
    epsilon : float, optional
        Constant :math:`\epsilon` added to the mean square for numerical stability. Default is 1e-6.
    affine : bool, optional
        Apply the learned scale :math:`\gamma`. Default is True.

    References
    ----------
    .. [1] Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
           arXiv:1910.07467. https://arxiv.org/abs/1910.07467.
    """

    def __init__(
        self,
        name: str | None = None,
        n_in: int | None = None,
        epsilon: float = 1e-6,
        affine: bool = True,
    ):
        self.name = name if name else "RMSNorm"
        self.n_in = n_in
        self.epsilon = epsilon
        self.affine = affine

        self.scale: TrainableParameter | None = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorVariable | None):
        if self.initialized:
            return

        n_in = _resolve_n_in(self.name, self.n_in, X)
        if n_in is None:
            return

        if self.affine:
            self.scale = trainable(np.ones(n_in, dtype=config.floatX), f"{self.name}_scale")

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        self._initialize_params(X)

        inputs = [X]
        if self.affine:
            assert self.scale is not None
            inputs.append(self.scale)

        X_transformed = RMSNormLayer(
            name=self.name,
            n_in=self.n_in,
            epsilon=self.epsilon,
            affine=self.affine,
        )(*inputs)
        X_transformed.name = f"{self.name}_output"

        return X_transformed
