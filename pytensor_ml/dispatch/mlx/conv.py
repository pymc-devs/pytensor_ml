import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify

from pytensor_ml.layers.conv import ConvLayer, ConvLayerGrad


def _convolution(op):
    """The convolution both dispatches here run, so the gradient differentiates what the forward does."""
    no_padding = (0,) * len(op.kernel_size)

    def convolve(X, W):
        # mlx takes the input channels-last as we store it, but wants the kernel's output channels
        # leading rather than trailing. Padding is a `pt.pad` node ahead of the op.
        return mx.conv_general(
            X,
            mx.moveaxis(W, -1, 0),
            stride=op.stride,
            padding=no_padding,
            kernel_dilation=op.dilation,
        )

    return convolve


@mlx_funcify.register(ConvLayer)
def mlx_funcify_ConvLayer(op, node=None, **kwargs):
    """Dispatch the convolution marker to ``mx.conv_general`` (fused Metal kernel)."""
    convolve = _convolution(op)

    def conv(X, W, *bias):
        out = convolve(X, W)
        return out + bias[0] if bias else out

    return conv


@mlx_funcify.register(ConvLayerGrad)
def mlx_funcify_ConvLayerGrad(op, node=None, **kwargs):
    """Let mlx differentiate its own convolution rather than spelling the two gradients out.

    The forward value the pullback is built from is never evaluated, and mlx is lazy, so it costs
    nothing beyond the intermediates the gradients need anyway.
    """
    convolve = _convolution(op)
    compute_dX, compute_dW = op.compute_dX, op.compute_dW

    def conv_grad(X, W, cotangent):
        # The vjp is taken only over the inputs whose gradient is wanted; the rest are closed over as
        # constants, so mlx never differentiates toward a gradient the graph will not read.
        if compute_dX and compute_dW:
            _, pullback = mx.vjp(convolve, [X, W], [cotangent])
            return tuple(pullback)
        # An op with one output is dispatched to a function returning that output, not a list of one.
        if compute_dX:
            _, pullback = mx.vjp(lambda x: convolve(x, W), [X], [cotangent])
        else:
            _, pullback = mx.vjp(lambda w: convolve(X, w), [W], [cotangent])
        (gradient,) = pullback
        return gradient

    return conv_grad
