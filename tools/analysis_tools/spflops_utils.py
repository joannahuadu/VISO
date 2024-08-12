import torch
import spconv.pytorch as spconv
from thop.vision.basic_hooks import *
from spconv.pytorch import ops
from spconv.core import ConvAlgo
from typing import List

def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

def calculate_spconv2dk1_flops(input_size: list, output_size: list):
    out_c = output_size[1]
    return out_c * l_prod(input_size)


def calculate_spconv2d_flops(input_size: list, 
                             output_size: list, 
                             indices: torch.Tensor, 
                             spatial_shape: List[int],
                             algo: ConvAlgo,
                             kernel_size: List[int],
                             stride: List[int],
                             padding: List[int],
                             dilation: List[int],
                             output_padding: List[int],
                             batch_size: int = 1, 
                             subm: bool = True,
                             transposed: bool = False):
    _, indice_pairs, _ = ops.get_indice_pairs(
        indices, batch_size, spatial_shape, algo,
        kernel_size, stride, padding,
        dilation, output_padding, subm,
        transposed)
    
    n_ = torch.sum(indice_pairs[0].view(-1)!=-1).cpu()
    in_c = input_size[1]
    out_c = output_size[1]
    return in_c * out_c * n_

def count_submconv2d(m: spconv.SubMConv2d, x, y: torch.Tensor):
    if m.conv1x1:    
        m.total_ops += calculate_spconv2dk1_flops(
            input_size = list(x[0].features.shape),
            output_size = list(y.features.shape),
        )
    else:
        m.total_ops += calculate_spconv2d_flops(
            input_size = list(x[0].features.shape),
            output_size = list(y.features.shape),
            indices = x[0].indices,
            spatial_shape= x[0].spatial_shape,
            algo = m.algo,
            kernel_size = m.kernel_size,
            stride = m.stride,
            padding = m.padding,
            dilation = m.dilation,
            output_padding = m.output_padding
         )

