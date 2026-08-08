import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import Schedule


def cosine_annealing(
    learning_rate: float,
    total_steps: int,
    min_learning_rate: float = 0.0,
) -> Schedule:
    r"""
    Anneal the learning rate from ``learning_rate`` to ``min_learning_rate`` along a half cosine.

    At step :math:`t` of :math:`T` total steps,

    .. math::

        \eta_t = \eta_{\min} + \frac{1}{2} (\eta_0 - \eta_{\min})
                 \left(1 + \cos\left(\pi \frac{\min(t, T)}{T}\right)\right)

    so the rate leaves :math:`\eta_0` slowly, falls fastest at the midpoint, and flattens into
    :math:`\eta_{\min}`, where it stays for any step past :math:`T`.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at step zero.
    total_steps : int
        Number of steps :math:`T` over which the rate reaches its floor. Must be at least one.
    min_learning_rate : float, optional
        Floor :math:`\eta_{\min}` reached at step ``total_steps``. Default 0.0.

    Returns
    -------
    Schedule
        A callable mapping the symbolic step count to a scalar learning rate, for
        :func:`~pytensor_ml.optim.transform.scale_by_schedule`.

    Examples
    --------
    Chain the schedule onto a unit-rate base rule, so the schedule alone sets the step size:

    .. code-block:: python

        from pytensor_ml.optim import adam, chain, cosine_annealing, scale_by_schedule

        rule = chain(adam(learning_rate=1.0), scale_by_schedule(cosine_annealing(3e-4, 10_000)))
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}.")

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(min_learning_rate, dtype=floatX)
        step_limit = np.asarray(total_steps, dtype=floatX)

        progress = pt.minimum(step_count.astype(floatX), step_limit) / step_limit
        cosine_factor = 0.5 * (1.0 + pt.cos(np.pi * progress))
        return final_rate + (initial_rate - final_rate) * cosine_factor

    return schedule
