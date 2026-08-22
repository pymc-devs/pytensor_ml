import torch
import torch.nn.functional as F

from pytensor.link.pytorch.dispatch import pytorch_funcify

from pytensor_ml.layers.conv import ConvLayer, ConvLayerGrad

_CONVOLUTIONS = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def _convolution(op):
    """The convolution both dispatches here run, so the gradient differentiates what the forward does.

    Torch is the one backend whose layouts disagree with ours at both ends: it takes the activation
    channels-first and the kernel output-channels-first, where we store both the other way round. The
    kernel permute is negligible, but the two activation moves are on the largest tensor in the graph.
    """
    n_spatial = len(op.kernel_size)
    if n_spatial not in _CONVOLUTIONS:
        raise NotImplementedError(f"Torch has no convolution over {n_spatial} spatial axes.")
    convolution = _CONVOLUTIONS[n_spatial]
    stride, dilation = op.stride, op.dilation

    def convolve(X, W):
        # Padding is a `pt.pad` node ahead of the op, so what arrives here is already padded.
        out = convolution(
            X.movedim(-1, 1),
            W.permute(n_spatial + 1, n_spatial, *range(n_spatial)),
            stride=stride,
            dilation=dilation,
        )
        return out.movedim(1, -1)

    return convolve


@pytorch_funcify.register(ConvLayer)
def pytorch_funcify_ConvLayer(op, node=None, **kwargs):
    """Dispatch the convolution marker to ``torch.nn.functional.conv{1,2,3}d``."""
    convolve = _convolution(op)

    def conv(X, W, *bias):
        out = convolve(X, W)
        return out + bias[0] if bias else out

    return conv


@pytorch_funcify.register(ConvLayerGrad)
def pytorch_funcify_ConvLayerGrad(op, node=None, **kwargs):
    """Let torch differentiate its own convolution rather than spelling the two gradients out."""
    convolve = _convolution(op)
    compute_dX, compute_dW = op.compute_dX, op.compute_dW

    def conv_grad(X, W, cotangent):
        # Only the inputs whose gradient is wanted require one, so torch never differentiates toward a
        # gradient the graph will not read.
        X_leaf = X.detach().requires_grad_(compute_dX)
        W_leaf = W.detach().requires_grad_(compute_dW)
        wrt = tuple(
            tensor for tensor, wanted in ((X_leaf, compute_dX), (W_leaf, compute_dW)) if wanted
        )
        gradients = torch.autograd.grad(convolve(X_leaf, W_leaf), wrt, grad_outputs=cotangent)
        # An op with one output is dispatched to a function returning that output, not a list of one.
        return gradients if len(gradients) > 1 else gradients[0]

    return conv_grad
