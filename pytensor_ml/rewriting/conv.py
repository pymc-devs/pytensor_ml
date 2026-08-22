from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.rewriting.basic import register_specialize

from pytensor_ml.layers.conv import ConvLayerGrad

DX, DW = 0, 1


def _drop_unused_gradient(fgraph, node, unused: int):
    """
    Rebuild a convolution's pullback without the output nothing reads.

    :meth:`ConvLayer.pullback` asks for both gradients, because it cannot know which the caller wants --
    only the graph knows that, and only once it is built. An unused output does not prune itself the way
    an unused node does, so dropping it takes a rewrite.
    """
    op = node.op
    if not (op.compute_dX and op.compute_dW):
        return None
    if fgraph.clients[node.outputs[unused]]:
        return None

    kept = DW if unused == DX else DX
    lowered = ConvLayerGrad(
        op.kernel_size, op.stride, op.dilation, compute_dX=kept == DX, compute_dW=kept == DW
    )
    [replacement] = lowered(*node.inputs, return_list=True)
    return {node.outputs[kept]: replacement}


@register_specialize
@node_rewriter([ConvLayerGrad])
def drop_unused_input_grad(fgraph, node):
    """Stop a convolution's pullback computing the input gradient nothing reads, which is the first
    convolution of a network, whose input is data."""
    return _drop_unused_gradient(fgraph, node, unused=DX)


@register_specialize
@node_rewriter([ConvLayerGrad])
def drop_unused_kernel_grad(fgraph, node):
    """Stop a convolution's pullback computing the kernel gradient nothing reads, which is a transposed
    convolution, where the pullback is the forward pass and the kernel's gradient is nobody's."""
    return _drop_unused_gradient(fgraph, node, unused=DW)
