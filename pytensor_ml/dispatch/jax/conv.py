import jax

from pytensor.link.jax.dispatch import jax_funcify

from pytensor_ml.layers.conv import ConvLayer, ConvLayerGrad


def _spatial_letters(n_spatial: int) -> str:
    """Name each spatial axis with a distinct letter, which is how jax takes a convolution's layout."""
    letters = "XYZUVW"
    if n_spatial > len(letters):
        raise ValueError(
            f"A convolution over {n_spatial} spatial axes has more axes than there are letters to name "
            f"them with; {len(letters)} is the most jax supports here."
        )
    return letters[:n_spatial]


def _convolution(op):
    """The convolution both dispatches here run, so the gradient differentiates what the forward does."""
    spatial = _spatial_letters(len(op.kernel_size))
    # Our layouts are already the ones jax names: the input is channels-last and the kernel is taps,
    # then input channels, then output. Nothing is transposed on the way in or out.
    dimension_numbers = (f"N{spatial}C", f"{spatial}IO", f"N{spatial}C")

    def convolve(X, W):
        # Padding is a `pt.pad` node ahead of the op, so what arrives here is already padded.
        return jax.lax.conv_general_dilated(
            X,
            W,
            window_strides=op.stride,
            padding="VALID",
            rhs_dilation=op.dilation,
            dimension_numbers=dimension_numbers,
        )

    return convolve


@jax_funcify.register(ConvLayer)
def jax_funcify_ConvLayer(op, node=None, **kwargs):
    """Dispatch the convolution marker to ``jax.lax.conv_general_dilated`` (XLA/cuDNN)."""
    convolve = _convolution(op)

    def conv(X, W, *bias):
        out = convolve(X, W)
        return out + bias[0] if bias else out

    return conv


@jax_funcify.register(ConvLayerGrad)
def jax_funcify_ConvLayerGrad(op, node=None, **kwargs):
    """Let jax differentiate its own convolution rather than spelling the two gradients out."""
    convolve = _convolution(op)
    compute_dX, compute_dW = op.compute_dX, op.compute_dW

    def conv_grad(X, W, cotangent):
        # The vjp is taken only over the inputs whose gradient is wanted; the rest are closed over as
        # constants, so jax never differentiates toward a gradient the graph will not read.
        if compute_dX and compute_dW:
            _, pullback = jax.vjp(convolve, X, W)
            return tuple(pullback(cotangent))
        # An op with one output is dispatched to a function returning that output, not a list of one.
        if compute_dX:
            _, pullback = jax.vjp(lambda x: convolve(x, W), X)
        else:
            _, pullback = jax.vjp(lambda w: convolve(X, w), W)
        (gradient,) = pullback(cotangent)
        return gradient

    return conv_grad
