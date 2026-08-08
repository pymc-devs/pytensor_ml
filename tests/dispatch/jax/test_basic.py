from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import pytensor
import pytest

from pytensor.compile.mode import JAX, Mode
from pytensor.graph.basic import Variable
from pytensor.link.jax.linker import JAXLinker

jax = pytest.importorskip("jax")

# The JAX mode's own optimizer, not a hand-built ``include=["jax"]`` query. A narrower query skips
# canonicalize, so ops that reach the backend only after being rewritten -- SplitDims and JoinDims
# become Reshape, for instance -- raise "No JAX conversion" here while compiling fine in real use.
jax_mode = Mode(linker=JAXLinker(), optimizer=JAX._optimizer)
py_mode = Mode(linker="py", optimizer=None)


def compare_jax_and_py(
    graph_inputs: Iterable[Variable],
    graph_outputs: Variable | Iterable[Variable],
    test_inputs: Iterable,
    *,
    assert_fn: Callable | None = None,
    must_be_device_array: bool = True,
    jax_mode=jax_mode,
    py_mode=py_mode,
):
    """Compile a graph on JAX and on the python linker, run both, and assert they agree.

    Parameters
    ----------
    graph_inputs : iterable of Variable
        Symbolic root inputs to the graph.
    graph_outputs : Variable or iterable of Variable
        Symbolic outputs of the graph.
    test_inputs : iterable
        Numerical inputs to evaluate the function on.
    assert_fn : callable, optional
        Equality check between the JAX and python results. Defaults to
        ``np.testing.assert_allclose`` with ``rtol=1e-4``.
    must_be_device_array : bool
        Assert the JAX result is a ``jax.Array``, confirming it was computed by JAX. Default True.

    Returns
    -------
    tuple
        The compiled JAX function and its result.
    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    if any(inp.owner is not None for inp in graph_inputs):
        raise ValueError("Inputs must be root variables")

    pytensor_jax_fn = pytensor.function(graph_inputs, graph_outputs, mode=jax_mode)
    jax_res = pytensor_jax_fn(*test_inputs)

    if must_be_device_array:
        if isinstance(jax_res, list):
            assert all(isinstance(res, jax.Array) for res in jax_res)
        else:
            assert isinstance(jax_res, jax.Array)

    pytensor_py_fn = pytensor.function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    if isinstance(graph_outputs, list | tuple):
        for j, p in zip(jax_res, py_res, strict=True):
            assert_fn(j, p)
    else:
        assert_fn(jax_res, py_res)

    return pytensor_jax_fn, jax_res
