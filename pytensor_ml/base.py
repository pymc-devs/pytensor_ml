from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pytensor.tensor as pt

from pytensor.compile.builders import SymbolicOp
from pytensor.tensor.variable import TensorVariable


class Layer(ABC):
    """Base class for the objects that build layer graphs. Defined here, not in ``pytensor_ml.layers``, so
    that ``pytensor_ml.activations`` can subclass it without a circular import."""

    @abstractmethod
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable: ...


class PositionalLayer(ABC):
    """Base class for layers that consume token positions alongside their input.

    A sibling of :class:`Layer` rather than a subclass: narrowing ``Layer.__call__`` from one tensor to
    two is not a valid override, so a subclass would need a type-checker escape at every level.
    """

    @abstractmethod
    def __call__(self, x: pt.TensorLike, position_ids: pt.TensorLike) -> pt.TensorVariable: ...


class LayerOp(SymbolicOp):
    """Base class for the library's neural-network ops.

    A ``SymbolicOp`` is an ``OpFromGraph`` whose inner graph is rebuilt from its ``__props__`` and input
    types by :meth:`build_inner_graph`, so equal props with equal inputs yield an identical op. Basing the
    layers on it (rather than plain ``OpFromGraph``) is what lets the numba backend optimize each inner
    graph: ``SymbolicOp`` restores the fgraph-aware ``__eq__``/``__hash__`` that a props-carrying
    ``OpFromGraph`` would otherwise lose, so the ``ofg_inner_graph`` rewrite keeps its optimized inner
    graph instead of discarding it as unchanged.
    """

    __props__: tuple[str, ...] = ()


class UnaryLayerOp(LayerOp):
    """A ``LayerOp`` with exactly one output, typed as such.

    ``SymbolicOp.__call__`` is annotated ``Variable | list[Variable]`` because an op may produce many
    outputs; a unary layer op produces one, so narrow the result to ``TensorVariable``. The ``isinstance``
    guard narrows for the type checker without a cast and asserts the invariant at runtime.
    """

    def __call__(self, *inputs, **kwargs) -> TensorVariable:
        out = super().__call__(*inputs, **kwargs)
        assert isinstance(out, TensorVariable), f"{type(self).__name__} produced multiple outputs"
        return out


@runtime_checkable
class StatefulOp(Protocol):
    """An op that writes some of its outputs back to shared variables it takes as inputs, the way batch
    norm updates its running statistics. Defining :meth:`update_map` is what marks an op stateful -- the
    check is structural, so there is nothing to register."""

    def update_map(self) -> dict[int, int]:
        """Map each output index to the index of the input that output updates."""


__all__ = ["Layer", "LayerOp", "PositionalLayer", "StatefulOp", "UnaryLayerOp"]
