import importlib

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.replace import vectorize_graph

import pytensor_ml.layers

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import (
    BatchNorm2D,
    Dropout,
    Embedding,
    Input,
    LayerNorm,
    Linear,
    RMSNorm,
    Sequential,
)
from pytensor_ml.pytensorf import (
    collect_non_trainable_updates,
    collect_trainable_params,
    rewrite_for_prediction,
)

floatX = pytensor.config.floatX

# The numpy references below sum in a different order than the graph, so the gap tracks the precision.
ATOL = 1e-6 if floatX == "float64" else 1e-5


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml layers")))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_linear_layer(bias, rng):
    X = pt.tensor("X", shape=(None, 6))
    linear = Linear(name="Linear_1", n_in=6, n_out=3, bias=bias)
    out = linear(X)

    X_in, *weights = out.owner.inputs
    [X_out] = out.owner.outputs

    assert out.owner.op.name == "Linear_1[(?,6) -> (?,3)]"

    expected_names = ["Linear_1_W", "Linear_1_b"] if bias else ["Linear_1_W"]
    assert [w.name for w in weights] == expected_names

    assert X_out.name == "Linear_1_output"

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    W_np = rng.normal(size=(6, 3)).astype(floatX)
    b_np = rng.normal(size=(3,)).astype(floatX)

    linear.W.set_value(W_np)
    if bias:
        linear.b.set_value(b_np)

    res = out.eval({X: X_np})
    expected = X_np @ W_np + b_np if bias else X_np @ W_np
    np.testing.assert_allclose(res, expected)


def test_sequential(rng):
    linear1 = Linear(name="Linear_1", n_in=6, n_out=3)
    linear2 = Linear(name="Linear_2", n_in=3, n_out=1)
    mlp = Sequential(linear1, linear2)

    X = pt.tensor("X", shape=(None, 6))
    out = mlp(X)
    assert out.type.shape == (None, 1)

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    W1_np = rng.normal(size=(6, 3)).astype(floatX)
    b1_np = rng.normal(size=(3,)).astype(floatX)
    W2_np = rng.normal(size=(3, 1)).astype(floatX)
    b2_np = rng.normal(size=(1,)).astype(floatX)

    linear1.W.set_value(W1_np)
    linear1.b.set_value(b1_np)
    linear2.W.set_value(W2_np)
    linear2.b.set_value(b2_np)

    f = pytensor.function([X], out)
    res = f(X_np)

    np.testing.assert_allclose(res, (X_np @ W1_np + b1_np) @ W2_np + b2_np)


def test_dropout(rng):
    X = pt.tensor("X", shape=(None, 6))
    dropout = Dropout(name="Dropout_1", p=1.0)
    out = dropout(X)

    X_np = rng.normal(size=(10, 6)).astype(floatX)

    res = out.eval({X: X_np})
    np.testing.assert_allclose(res, np.zeros_like(X_np))


def test_invalid_dropout_p_raises():
    with pytest.raises(
        ValueError, match=r"Dropout probability has to be between 0 and 1, but got -0\.1"
    ):
        Dropout(name=None, p=-0.1)

    with pytest.raises(
        ValueError, match=r"Dropout probability has to be between 0 and 1, but got 1\.1"
    ):
        Dropout(name=None, p=1.1)


def test_embedding_forward(rng):
    n_embeddings, n_features = 10, 4
    embedding = Embedding("emb", n_embeddings, n_features)
    W_np = rng.normal(size=(n_embeddings, n_features)).astype(floatX)
    embedding.W.set_value(W_np)

    ids = Input("ids", (2, 3), dtype="int64")  # a batch of index rows
    out = embedding(ids)
    assert out.name == "emb_output"

    ids_np = np.array([[1, 2, 3], [4, 5, 6]])
    res = out.eval({ids: ids_np})
    np.testing.assert_allclose(res, W_np[ids_np])
    assert res.shape == (2, 3, n_features)


def test_embedding_table_is_trainable(rng):
    # The OpFromGraph marker must pass the gradient through to the selected rows -- and only
    # those rows -- so the table trains; the integer indices are non-differentiable.
    embedding = Embedding("emb", n_embeddings=6, n_features=3)
    embedding.W.set_value(rng.normal(size=(6, 3)).astype(floatX))
    ids = pt.lvector("ids")
    grad_fn = pytensor.function([ids], pytensor.grad((embedding(ids) ** 2).sum(), embedding.W))

    grad = grad_fn(np.array([1, 1, 4]))
    selected = np.zeros(6, dtype=bool)
    selected[[1, 4]] = True
    assert np.any(grad[selected] != 0)
    assert np.all(grad[~selected] == 0)


@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_batch_norm_2d_forward(n_in, rng):
    X = pt.tensor("X", shape=(None, 6))
    batch_norm = BatchNorm2D(name="BatchNorm_1", n_in=n_in)
    out = batch_norm(X)

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    scale_np = rng.normal(size=(6,)).astype(floatX) ** 2
    loc_np = rng.normal(size=(6,)).astype(floatX) ** 2
    batch_norm.scale.set_value(scale_np)
    batch_norm.loc.set_value(loc_np)

    res = out.eval({X: X_np})
    mean_np = X_np.mean(axis=0)
    var_np = X_np.var(axis=0)
    expected = (X_np - mean_np) / np.sqrt(var_np + batch_norm.epsilon) * scale_np + loc_np

    np.testing.assert_allclose(res, expected, rtol=1e-5)


# Rank > 2 is the transformer case (batch, seq, d_model): the last axis is normalized and the affine
# parameters broadcast over every leading dimension.
@pytest.mark.parametrize("batch_shape", [(10,), (2, 4)], ids=["2d", "3d"])
@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_layer_norm_forward(n_in, batch_shape, rng):
    X = pt.tensor("X", shape=(*(None,) * len(batch_shape), 6))
    layer_norm = LayerNorm(name="LayerNorm_1", n_in=n_in)
    out = layer_norm(X)
    assert out.name == "LayerNorm_1_output"

    X_np = rng.normal(size=(*batch_shape, 6)).astype(floatX)
    scale_np = rng.normal(size=(6,)).astype(floatX)
    loc_np = rng.normal(size=(6,)).astype(floatX)
    layer_norm.scale.set_value(scale_np)
    layer_norm.loc.set_value(loc_np)

    res = out.eval({X: X_np})
    mean_np = X_np.mean(axis=-1, keepdims=True)
    var_np = X_np.var(axis=-1, keepdims=True)
    expected = (X_np - mean_np) / np.sqrt(var_np + layer_norm.epsilon) * scale_np + loc_np
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_layer_norm_prediction_matches_training(rng):
    # LayerNorm normalizes over per-sample statistics, identical in train and eval, so unlike
    # BatchNorm it needs no prediction rewrite: rewrite_for_prediction leaves its output unchanged.
    X = pt.tensor("X", shape=(None, 6))
    layer_norm = LayerNorm("ln", n_in=6)
    out = layer_norm(X)
    layer_norm.scale.set_value(rng.normal(size=6).astype(floatX))
    layer_norm.loc.set_value(rng.normal(size=6).astype(floatX))

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    np.testing.assert_allclose(
        rewrite_for_prediction(out).eval({X: X_np}), out.eval({X: X_np}), rtol=1e-6
    )


def test_vectorize_graph_batches_independent_predictions(rng):
    # A model built for a single sample must vectorize over a batch through the OpFromGraph-based
    # layers (Linear, LayerNorm); the batched result must match looping the single-sample graph.
    x = pt.vector("x", shape=(4,))
    net = Sequential(Linear("fc1", 4, 8), ReLU(), LayerNorm("ln", n_in=8), Linear("fc2", 8, 3))
    out = net(x)
    for parameter in collect_trainable_params(out):
        parameter.set_value(rng.normal(size=parameter.get_value().shape).astype(floatX))

    X = pt.matrix("X", shape=(None, 4))
    f_single = pytensor.function([x], out)
    f_batch = pytensor.function([X], vectorize_graph(out, {x: X}))

    X_np = rng.normal(size=(5, 4)).astype(floatX)
    np.testing.assert_allclose(f_batch(X_np), np.stack([f_single(row) for row in X_np]), rtol=1e-5)


def test_layer_norm_no_affine_standardizes_each_row(rng):
    X = pt.tensor("X", shape=(None, 8))
    out = LayerNorm(name="LayerNorm_1", n_in=8, affine=False)(X)

    X_np = rng.normal(loc=3.0, scale=2.0, size=(10, 8)).astype(floatX)
    res = out.eval({X: X_np})

    np.testing.assert_allclose(res.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(res.var(axis=-1), 1.0, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [(10,), (2, 4)], ids=["2d", "3d"])
@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_rms_norm_forward(n_in, batch_shape, rng):
    X = pt.tensor("X", shape=(*(None,) * len(batch_shape), 6))
    rms_norm = RMSNorm(name="RMSNorm_1", n_in=n_in)
    out = rms_norm(X)
    assert out.name == "RMSNorm_1_output"

    X_np = rng.normal(size=(*batch_shape, 6)).astype(floatX)
    scale_np = rng.normal(size=(6,)).astype(floatX)
    rms_norm.scale.set_value(scale_np)

    res = out.eval({X: X_np})
    mean_square_np = np.square(X_np).mean(axis=-1, keepdims=True)
    expected = X_np / np.sqrt(mean_square_np + rms_norm.epsilon) * scale_np
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_rms_norm_has_no_shift_parameter():
    # Scale-only by definition, matching torch.nn.RMSNorm and flax/tinygrad's RMSNorm. A `loc` here
    # would mean the layer had quietly become a LayerNorm.
    rms_norm = RMSNorm(name="RMSNorm_1", n_in=6)
    assert not hasattr(rms_norm, "loc")
    assert collect_trainable_params(rms_norm(pt.tensor("X", shape=(None, 6)))) == [rms_norm.scale]


def test_rms_norm_no_affine_gives_unit_root_mean_square(rng):
    X = pt.tensor("X", shape=(None, 8))
    out = RMSNorm(name="RMSNorm_1", n_in=8, affine=False)(X)

    X_np = rng.normal(loc=3.0, scale=2.0, size=(10, 8)).astype(floatX)
    res = out.eval({X: X_np})

    np.testing.assert_allclose(np.sqrt(np.square(res).mean(axis=-1)), 1.0, rtol=1e-3)


def test_rms_norm_does_not_center_its_input(rng):
    # The one thing that separates RMSNorm from LayerNorm: the mean survives. On input with a large
    # offset, LayerNorm removes it and RMSNorm does not, so the two must disagree.
    X = pt.tensor("X", shape=(None, 8))
    X_np = rng.normal(loc=5.0, scale=1.0, size=(10, 8)).astype(floatX)

    rms = RMSNorm(name="rms", n_in=8, affine=False)(X).eval({X: X_np})
    layer = LayerNorm(name="ln", n_in=8, affine=False)(X).eval({X: X_np})

    # LayerNorm removes the offset outright; RMSNorm only rescales it, so a clearly non-zero mean
    # survives. A stray mean subtraction in RMSNorm would drive this to zero.
    np.testing.assert_allclose(layer.mean(axis=-1), 0.0, atol=1e-5)
    assert np.all(rms.mean(axis=-1) > 0.5)
    assert not np.allclose(rms, layer)


def test_rms_norm_prediction_matches_training(rng):
    # Per-sample statistics, identical in train and eval, so like LayerNorm it needs no prediction
    # rewrite: rewrite_for_prediction must leave its output unchanged.
    X = pt.tensor("X", shape=(None, 6))
    rms_norm = RMSNorm("rms", n_in=6)
    out = rms_norm(X)
    rms_norm.scale.set_value(rng.normal(size=6).astype(floatX))

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    np.testing.assert_allclose(
        rewrite_for_prediction(out).eval({X: X_np}), out.eval({X: X_np}), rtol=1e-6
    )


def test_batch_norm_2d_learns_population_stats(rng):
    population_mean, population_std = 3.2, 6.2
    X = pt.tensor("X", shape=(None, 32))
    batch_norm = BatchNorm2D(name="BatchNorm_1", n_in=32, momentum=0.05, epsilon=1e-8)
    X_normalized = batch_norm(X)

    loss = pt.square(X_normalized - X).mean()
    d_loss = pt.grad(loss, [batch_norm.loc, batch_norm.scale])

    learning_rate = 1e-1
    updates = {
        batch_norm.loc: batch_norm.loc - learning_rate * d_loss[0],
        batch_norm.scale: batch_norm.scale - learning_rate * d_loss[1],
        batch_norm.running_mean: batch_norm.new_running_mean,
        batch_norm.running_var: batch_norm.new_running_var,
    }
    train = pytensor.function([X], X_normalized, updates=updates)

    def sample_batch():
        return rng.normal(loc=population_mean, scale=population_std, size=(100, 32)).astype(floatX)

    for _ in range(500):
        data = sample_batch()
        # Read before stepping: the affine parameters this batch is normalized with are the ones the
        # updates are about to overwrite.
        scale, loc = batch_norm.scale.get_value(), batch_norm.loc.get_value()

        np.testing.assert_allclose(
            train(data),
            (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0) + batch_norm.epsilon) * scale
            + loc,
            rtol=1e-4,
            atol=ATOL,
        )

    scale, loc = batch_norm.scale.get_value(), batch_norm.loc.get_value()
    running_mean = batch_norm.running_mean.get_value()
    running_var = batch_norm.running_var.get_value()

    np.testing.assert_allclose(loc, population_mean, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(scale, population_std, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(running_mean, population_mean, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(np.sqrt(running_var), population_std, rtol=1e-1, atol=1e-1)

    predict = pytensor.function([X], rewrite_for_prediction(X_normalized))
    data = sample_batch()

    np.testing.assert_allclose(
        predict(data),
        (data - running_mean) / np.sqrt(running_var + batch_norm.epsilon) * scale + loc,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "op_name, submodule",
    [
        ("LinearLayer", "linear"),
        ("EmbeddingLayer", "embedding"),
        ("DropoutLayer", "dropout"),
        ("BatchNormLayer", "norm"),
        ("NoRunningStatsBatchNormLayer", "norm"),
        ("PredictionBatchNormLayer", "norm"),
        ("LayerNormLayer", "norm"),
        ("RMSNormLayer", "norm"),
        ("RotaryEmbeddingLayer", "positional"),
    ],
)
def test_marker_ops_stay_reachable_from_the_package(op_name, submodule):
    # deserialize_graph resolves an op's recorded import path with getattr on this package, so these
    # bindings are load-bearing rather than convenience re-exports.
    from_package = getattr(pytensor_ml.layers, op_name)
    from_submodule = getattr(importlib.import_module(f"pytensor_ml.layers.{submodule}"), op_name)

    assert from_package is from_submodule


def test_batch_norm_without_running_stats_normalizes_with_batch_statistics(rng):
    X = pt.tensor("X", shape=(None, 4))
    normalize = pytensor.function([X], BatchNorm2D("bn", n_in=4, track_running_stats=False)(X))

    out = normalize(rng.normal(loc=5.0, scale=3.0, size=(256, 4)).astype(floatX))

    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1.0, rtol=1e-3)


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
def test_batch_norm_running_stats_write_back_to_the_right_inputs(affine):
    # The update map indexes inputs positionally, and the affine parameters shift the running
    # statistics along by two when present.
    X = pt.tensor("X", shape=(None, 4))
    batch_norm = BatchNorm2D("bn", n_in=4, affine=affine)

    updates = collect_non_trainable_updates(batch_norm(X))

    assert updates == {
        batch_norm.running_mean: batch_norm.new_running_mean,
        batch_norm.running_var: batch_norm.new_running_var,
    }


def test_batch_norm_variants_agree_on_output_arity():
    X = pt.tensor("X", shape=(None, 4))
    tracked = BatchNorm2D("tracked", n_in=4)(X)
    untracked = BatchNorm2D("untracked", n_in=4, track_running_stats=False)(X)

    # Matching arity is what lets BatchNorm2D use one code path; the untracked variant reports the
    # batch statistics but must not write them anywhere.
    assert len(tracked.owner.outputs) == len(untracked.owner.outputs)
    assert collect_non_trainable_updates(untracked) == {}
