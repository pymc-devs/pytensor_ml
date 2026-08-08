from collections.abc import Callable, Sequence

from pytensor_ml.optim.base import Parameter, Schedule, Transform, Updates, counter, state_for


def trace(decay: float = 0.9, nesterov: bool = False) -> Transform:
    r"""
    Accumulate steps into a velocity buffer (classical or Nesterov momentum).

    Operating in step space (:math:`s = \text{updates}[p] - p`), the velocity is
    :math:`v \leftarrow \rho v + s`, and the new step is :math:`v` (classical) or :math:`s + \rho v`
    (Nesterov lookahead).

    Parameters
    ----------
    decay : float
        Momentum coefficient :math:`\rho`. Default 0.9.
    nesterov : bool
        Apply the Nesterov lookahead correction. Default False.

    Returns
    -------
    Transform
        A transform that folds momentum into the updates dict.
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        next_updates = dict(updates)
        for parameter in parameters:
            step = updates[parameter] - parameter
            velocity = state_for(parameter, "trace/velocity")
            new_velocity = decay * velocity + step
            next_updates[velocity] = new_velocity
            next_updates[parameter] = parameter + (
                step + decay * new_velocity if nesterov else new_velocity
            )
        return next_updates

    return transform


def scale(factor: float) -> Transform:
    """
    Scale each step by a constant factor.

    Typically the terminal transform in a chain, used to apply the learning rate after a unit-rate base rule.

    Parameters
    ----------
    factor : float
        Multiplier applied to every step.

    Returns
    -------
    Transform
        A transform that rescales the updates dict.
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        next_updates = dict(updates)
        for parameter in parameters:
            next_updates[parameter] = parameter + factor * (updates[parameter] - parameter)
        return next_updates

    return transform


def scale_by_schedule(schedule_fn: Schedule) -> Transform:
    """
    Scale each step by a learning rate produced from an owned step counter.

    The counter starts at zero and increments by one each call. ``schedule_fn`` receives it symbolically and
    must return a scalar learning rate, so the schedule lives on the graph and stays synchronized with the
    optimizer's own time step.

    Parameters
    ----------
    schedule_fn : callable
        Maps the symbolic step counter (a TensorVariable) to a scalar learning rate.

    Returns
    -------
    Transform
        A transform that rescales the updates dict by the scheduled rate and advances the counter.
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        step_count = counter("schedule/step_count")
        learning_rate = schedule_fn(step_count)
        next_updates = dict(updates)
        for parameter in parameters:
            next_updates[parameter] = parameter + learning_rate * (updates[parameter] - parameter)
        next_updates[step_count] = step_count + 1
        return next_updates

    return transform


def add_weight_decay(
    weight_decay: float = 0.01,
    mask: Callable[[Parameter], bool] | None = None,
) -> Transform:
    r"""
    Subtract a decoupled weight-decay term :math:`\lambda p` from each step.

    Place this before a terminal :func:`scale` so the final update is
    :math:`p \leftarrow p + \eta (s - \lambda p)`, giving decay that scales with the learning rate but is
    decoupled from any adaptive step rescaling earlier in the chain.

    Parameters
    ----------
    weight_decay : float
        Decay coefficient :math:`\lambda`. Default 0.01.
    mask : callable, optional
        Predicate ``(parameter) -> bool`` selecting which parameters receive decay. Decay is applied to every
        parameter when omitted.

    Returns
    -------
    Transform
        A transform that folds weight decay into the updates dict.
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        next_updates = dict(updates)
        for parameter in parameters:
            step = updates[parameter] - parameter
            decayed_step = (
                step - weight_decay * parameter if (mask is None or mask(parameter)) else step
            )
            next_updates[parameter] = parameter + decayed_step
        return next_updates

    return transform
