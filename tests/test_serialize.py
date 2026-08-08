import json

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.traversal import ancestors
from pytensor.graph.type import Type
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.type import TensorType

from pytensor_ml.activations import GELU, LeakyReLU, ReLU, Sigmoid, Softmax, SoftPlus, Swish, Tanh
from pytensor_ml.json_serialize import (
    deserialize_graph,
    op_from_json,
    prop_to_json,
    register_type,
    serialize_graph,
    type_from_json,
    type_to_json,
)
from pytensor_ml.layers import (
    BatchNorm2D,
    Concatenate,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    Sequential,
    Squeeze,
)
from pytensor_ml.layers.attention import scaled_dot_product_attention
from pytensor_ml.layers.positional import rotary_embedding
from pytensor_ml.pytensorf import collect_shared_variables, collect_trainable_params
from pytensor_ml.serialize.base import _TYPE_FROM_JSON, _TYPE_TO_JSON

floatX = pytensor.config.floatX

ALL_ACTIVATIONS = [
    ReLU(),
    LeakyReLU(),
    Tanh(),
    Sigmoid(),
    SoftPlus(),
    Softmax(),
    GELU(approximate=False),
    GELU(approximate=True),
    Swish(),
    Swish(beta=1.5),
]


def assert_outputs_roundtrip(data_inputs, outputs, data_values):
    """Serialize the graph of ``outputs`` to JSON and back, and check the rebuilt graph computes the same."""
    output_list = outputs if isinstance(outputs, list) else [outputs]
    shared = collect_shared_variables(output_list)
    # Plain literals default to float64, which does not fit a float32 graph.
    data_values = [
        np.asarray(value, dtype=data_input.type.dtype)
        for data_input, value in zip(data_inputs, data_values)
    ]

    # allow_nan=False enforces strict, portable JSON: inf/nan must go through sentinels, not the
    # non-standard Infinity/NaN tokens a lenient parser would emit.
    blob = json.dumps(serialize_graph([*data_inputs, *shared], output_list), allow_nan=False)
    rebuilt_inputs, rebuilt_outputs = deserialize_graph(json.loads(blob))

    original = pytensor.function(data_inputs, output_list)  # shared inputs captured
    restored = pytensor.function(rebuilt_inputs, rebuilt_outputs)  # every input explicit
    feed = [*data_values, *(s.get_value() for s in shared)]
    for got, want in zip(restored(*feed), original(*data_values)):
        np.testing.assert_allclose(got, want, rtol=1e-6)


def initialized_network(*layers, seed=0):
    rng = np.random.default_rng(seed)
    X = pt.matrix("X")
    output = Sequential(*layers)(X)
    for parameter in collect_trainable_params(output):
        value = rng.normal(size=parameter.get_value().shape)
        parameter.set_value(value.astype(parameter.type.dtype))
    return X, output


def _activation_id(activation):
    if isinstance(activation, GELU) and activation.approximate:
        return "GELU_tanh"
    if isinstance(activation, Swish):
        return f"Swish_beta{activation.beta}"
    return type(activation).__name__


@pytest.mark.parametrize("activation", ALL_ACTIVATIONS, ids=_activation_id)
def test_every_activation_roundtrips(activation):
    X, output = initialized_network(Linear("fc1", 4, 8), activation, Linear("fc2", 8, 4))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_linear_bias_variants_roundtrip(bias):
    X, output = initialized_network(Linear("fc", 4, 3, bias=bias))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"affine": False}, {"track_running_stats": False}],
    ids=["default", "no_affine", "no_running_stats"],
)
def test_batchnorm_variants_roundtrip(kwargs):
    X, output = initialized_network(Linear("fc", 4, 4), BatchNorm2D("bn", n_in=4, **kwargs))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(8, 4))])


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
def test_layernorm_roundtrips(affine):
    X, output = initialized_network(Linear("fc", 4, 6), LayerNorm("ln", n_in=6, affine=affine))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(8, 4))])


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
def test_rmsnorm_roundtrips(affine):
    X, output = initialized_network(Linear("fc", 4, 6), RMSNorm("rms", n_in=6, affine=affine))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(8, 4))])


@pytest.mark.parametrize("pairing", ["half", "adjacent"])
@pytest.mark.parametrize(
    "scaling, scaling_factor",
    [("none", 1.0), ("linear", 4.0), ("ntk", 4.0)],
    ids=["unscaled", "linear", "ntk"],
)
def test_rotary_embedding_roundtrips(pairing, scaling, scaling_factor):
    # Every option is a prop, so the rebuilt op must reproduce the same frequencies; a dropped prop
    # would silently fall back to the default and still evaluate.
    x = pt.tensor("x", shape=(2, 3, 5, 8))
    positions = pt.lvector("positions")
    output = rotary_embedding(
        x, positions, base=500.0, pairing=pairing, scaling=scaling, scaling_factor=scaling_factor
    )
    values = [np.random.default_rng(0).normal(size=(2, 3, 5, 8)), np.arange(5)]
    assert_outputs_roundtrip([x, positions], output, values)


def test_squeeze_roundtrips():
    X = pt.matrix("X")
    assert_outputs_roundtrip(
        [X], Squeeze(X[:, :1], axis=1), [np.random.default_rng(0).normal(size=(5, 4))]
    )


def test_concatenate_roundtrips():
    X = pt.matrix("X")
    assert_outputs_roundtrip(
        [X], Concatenate([X, X], axis=1), [np.random.default_rng(0).normal(size=(5, 4))]
    )


def test_specify_shape_with_unknown_dim_roundtrips():
    # An unspecified (None) dim in specify_shape is a NoneTypeT constant -- exactly what the grad and
    # canonicalize rewrites inside compile_train bake into a real model's graph.
    X = pt.matrix("X")
    output = pt.specify_shape(X, (None, 4))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(0).normal(size=(5, 4))])


def test_embedding_roundtrips():
    ids = pt.lmatrix("ids")
    embedding = Embedding("emb", n_embeddings=8, n_features=5)
    embedding.W.set_value(np.random.default_rng(0).normal(size=(8, 5)).astype(floatX))
    assert_outputs_roundtrip([ids], embedding(ids), [np.array([[1, 2, 3], [4, 0, 7]])])


@pytest.mark.parametrize("scale", [None, 0.5], ids=["default_scale", "custom_scale"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_attention_roundtrips(is_causal, scale):
    # The causal branch bakes a -inf constant into the graph, exercising the non-finite codec path.
    rng = np.random.default_rng(0)
    q, k, v = (pt.tensor(name, shape=(2, 2, 4, 3)) for name in "qkv")
    output = scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    values = [rng.normal(size=(2, 2, 4, 3)).astype(floatX) for _ in range(3)]
    assert_outputs_roundtrip([q, k, v], output, values)


def test_multi_output_network_roundtrips():
    X = pt.matrix("X")
    output = [Linear("head_a", 4, 2)(X), Linear("head_b", 4, 3)(X)]
    for parameter in collect_trainable_params(output):
        value = np.random.default_rng(0).normal(size=parameter.get_value().shape)
        parameter.set_value(value.astype(parameter.type.dtype))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


def test_dropout_graph_with_rng_roundtrips():
    # Dropout introduces an RNG (RandomGeneratorType) shared variable and a bernoulli RandomVariable; the
    # bernoulli must survive reconstruction, not collapse to an identity pass-through.
    X = pt.matrix("X")
    output = Dropout(p=0.5, random_state=0)(X)
    blob = json.dumps(serialize_graph([X, *collect_shared_variables(output)], [output]))
    _, rebuilt_outputs = deserialize_graph(json.loads(blob))
    assert any(
        node.owner and isinstance(node.owner.op, RandomVariable)
        for node in ancestors(rebuilt_outputs)
    )


def test_scan_recurrent_loop_roundtrips():
    rng = np.random.default_rng(0)
    sequence = pt.matrix("sequence")
    # floatX: the recurrence must not upcast against the float32 initial state.
    W_rec = pytensor.shared(rng.normal(size=(3, 3)).astype(floatX), name="W_rec")

    def step(x_t, hidden):
        return pt.tanh(x_t + hidden @ W_rec)

    hidden_seq = pytensor.scan(
        step, sequences=sequence, outputs_info=pt.zeros(3), return_updates=False
    )
    assert_outputs_roundtrip([sequence], hidden_seq[-1], [rng.normal(size=(6, 3))])


def test_unregistered_type_raises_loudly():
    class UnregisteredType(Type):
        def filter(self, data, strict=False, allow_downcast=None):
            return data

    with pytest.raises(TypeError, match="Unserializable type"):
        type_to_json(UnregisteredType())


@pytest.fixture
def isolated_type_registry():
    """Undo any type registration a test performs. The registries are module-level and lookup is
    isinstance-based, so a leaked entry would shadow a built-in for every test that ran afterwards."""
    encoders, decoders = list(_TYPE_TO_JSON), dict(_TYPE_FROM_JSON)
    yield
    _TYPE_TO_JSON[:] = encoders
    _TYPE_FROM_JSON.clear()
    _TYPE_FROM_JSON.update(decoders)


def test_register_type_takes_precedence_over_a_registered_supertype(isolated_type_registry):
    class CountingType(TensorType):
        pass

    register_type(
        "counting",
        CountingType,
        lambda graph_type: {"kind": "counting", "dtype": graph_type.dtype},
        lambda type_dict: CountingType(type_dict["dtype"], shape=()),
    )

    # TensorType is registered at import time and CountingType subclasses it, so this picks the new handler
    # only because the newest registration wins; appending would fall through to the tensor encoder.
    encoded = type_to_json(CountingType("float64", shape=()))

    assert encoded == {"kind": "counting", "dtype": "float64"}
    assert isinstance(type_from_json(encoded), CountingType)


# A config captured from a previous release, kept verbatim. An op's serialized "type" is its class's import
# path, so moving a layer op to another module silently stops old saved models from loading; this is the only
# test that would notice. Regenerate it only alongside a deliberate GRAPH_FORMAT_VERSION bump.
PREVIOUSLY_SERIALIZED_LINEAR = """
{"inputs": [{"kind": "tensor", "dtype": "float64", "shape": [2, 3]},
            {"kind": "tensor", "dtype": "float64", "shape": [2]},
            {"kind": "tensor", "dtype": "float64", "shape": [3, 2]}],
 "nodes": [{"op": {"family": "leaf", "type": "pytensor_ml.layers.LinearLayer",
                   "props": {"n_in": 3, "n_out": 2, "bias": true}},
            "inputs": [{"input": 0}, {"input": 2}, {"input": 1}],
            "outputs": [{"kind": "tensor", "dtype": "float64", "shape": [2, 2]}]}],
 "outputs": [{"node": 0, "out": 0}]}
"""


def test_previously_serialized_graph_still_deserializes():
    inputs, outputs = deserialize_graph(json.loads(PREVIOUSLY_SERIALIZED_LINEAR))

    X_values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    bias = np.array([1.0, -1.0])
    weight = np.arange(6, dtype="float64").reshape(3, 2)
    result = pytensor.function(inputs, outputs)(X_values, bias, weight)

    np.testing.assert_allclose(result[0], X_values @ weight + bias)


def test_unregistered_scalar_op_raises_loudly():
    with pytest.raises(NotImplementedError, match="Unregistered scalar op"):
        op_from_json({"family": "scalar", "type": "NoSuchScalarOp"})


def test_non_json_native_prop_raises_loudly():
    # The "no silent drop" guarantee: a prop the codec can't represent must raise, not vanish.
    with pytest.raises(TypeError, match="Unserializable op prop"):
        prop_to_json(object())
