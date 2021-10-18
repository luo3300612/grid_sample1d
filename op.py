import math
from torch import nn
from torch.autograd import Function
import torch
from torch.utils.cpp_extension import load

import grid_sample1d_cuda as grid_sample1d
import torch.nn.functional

# grid_sample1d = load(
#     'grid_sample1d_cuda', ['grid_sample1d_cuda.cpp', 'grid_sample1d_cuda_kernel.cu'], verbose=True)
# help(lltm_cuda)

# import lltm_cuda

torch.manual_seed(42)


class GridSample1dFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode, align_corners):
        outputs = grid_sample1d.forward(input, grid, padding_mode, align_corners)
        # print(print(outputs))
        ctx.save_for_backward(*(input, grid))
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        outputs = grid_sample1d.backward(grad_output.contiguous(), *ctx.saved_variables, ctx.padding_mode,
                                         ctx.align_corners)
        # outputs = lltm_cuda.backward(
        #     grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        # d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        # return d_input, d_weights, d_bias, d_old_h, d_old_cell
        d_input, d_grid = outputs
        # print(d_input)
        # print(d_grid)
        return d_input, d_grid, None, None


class GridSample1d(nn.Module):
    def __init__(self):
        super(GridSample1d, self).__init__()

    def forward(self, input, grid, padding_mode, align_corners):
        return GridSample1dFunction.apply(input, grid, padding_mode, align_corners)
