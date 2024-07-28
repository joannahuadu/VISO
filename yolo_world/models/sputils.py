# Copyright (c) Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn import ConvModule
import spconv.pytorch as spconv

class SPInfer:
    def __init__(self, sp_type = "spconv"):
        self.sp_type = sp_type
        self.make_conv = getattr(self, '_make_'+self.sp_type)
    
    def _replace_spinfer(self, name, module, parent) -> Tensor:
        if isinstance(module, ConvModule):
            weights, biases = get_params(module)
            if hasattr(module, 'activate'):
                act = module.activate
            else:
                act = None
            setattr(parent, name, self.make_conv(weights, biases, act)) # TODO: _make_spconv -> _make_conv
            return
        elif isinstance(module, nn.Conv2d):
            weights, biases = get_params(module)
            setattr(parent, name, self.make_conv(weights, biases)) # TODO: _make_spconv -> _make_conv
            return
        else:
            for name, child in module.named_children():
                self._replace_spinfer(name, child, module)
                
    def _make_vspconv(self, weights, biases, act=None):
        nets = []
        in_channel  = weights.shape[1]
        out_channel = weights.shape[0]
        k_size      = weights.shape[2]
        filter = spconv.SubMConv2d(in_channel, out_channel, k_size, 1, padding=k_size//2, indice_key="asd", algo=spconv.ConvAlgo.Native).to(device=weights.device)
        filter.weight.data[:] = weights.permute(0,2,3,1).contiguous()[:] # transpose(1,2).transpose(0,1).transpose(2,3).transpose(1,2).transpose(2,3)
        filter.bias.data   = biases
        nets.append(filter)
        if not act == None:
            nets.append(act) ## TODO: Change into SiLU
        return spconv.SparseSequential(*nets)

    def _make_spconv(self, weights, biases, act=None):
        nets = []
        in_channel  = weights.shape[1]
        out_channel = weights.shape[0]
        k_size      = weights.shape[2]
        filter = spconv.SparseConv2d(in_channel, out_channel, k_size, 2, padding=k_size//2, indice_key="asd", algo=spconv.ConvAlgo.Native).to(device=weights.device)
        filter.weight.data[:] = weights.permute(0,2,3,1).contiguous()[:] # transpose(1,2).transpose(0,1).transpose(2,3).transpose(1,2).transpose(2,3)
        filter.bias.data   = biases
        nets.append(filter)
        if not act == None:
            nets.append(act) ## TODO: Change into SiLU
        return spconv.SparseSequential(*nets)

    def _make_conv(self, weights, biases, act=None):
        nets = []
        in_channel  = weights.shape[0]
        out_channel = weights.shape[1]
        k_size      = weights.shape[2]
        filter = nn.Conv2d(in_channel, out_channel, k_size, 1, padding=k_size//2)
        filter.weight.data = weights
        filter.bias.data   = biases
        nets.append(filter)
        if not act is None:
            nets.append(act)
        return torch.nn.Sequential(*nets)

    def _make_dconv(self, weights, biases, act=None):
        nets = []
        in_channel  = weights.shape[0]
        out_channel = weights.shape[1]
        k_size      = weights.shape[2]
        filter = nn.Conv2d(in_channel, out_channel, k_size, 2, padding=k_size//2)
        filter.weight.data = weights
        filter.bias.data   = biases
        nets.append(filter)
        if not act is None:
            nets.append(act)
        return torch.nn.Sequential(*nets)
    
    def _run_spconvs(self, x, filters):
        y = filters(x)
        return y.dense(channels_first=False)

    def _run_convs(self, x, filters):
        return filters(x)

           
def get_params(module) -> Tensor:
    # if not self.bn_converted:
    if isinstance(module, ConvModule):
        _bn_convert(module)
    
    ws = module.weight.data
    bs = module.bias.data
    return ws, bs

def _bn_convert(module):
    # assert not self.training
    # if self.bn_converted:
    #     return
    running_mean = module.norm.running_mean.data
    running_var = module.norm.running_var.data
    gamma = module.norm.weight.data
    beta = module.norm.bias.data
    bn_scale = gamma * torch.rsqrt(running_var + 1e-10)
    bn_bias  = beta - bn_scale * running_mean
    setattr(module, 'weight', module.conv.weight.data * bn_scale.view(-1, 1, 1, 1))
    setattr(module, 'bias',  torch.nn.Parameter(bn_bias))
    # self.bn_converted = True

def _concat(f1, f2):
    if isinstance(f1, spconv.SparseConvTensor) and isinstance(f2, spconv.SparseConvTensor):
        concat_indices, inverse_indices = torch.unique(torch.cat([f1.indices, f2.indices], 0), sorted=True, return_inverse=True, dim=0)
        num_features = f1.features.size(1) + f2.features.size(1)
        new_features =  torch.zeros(concat_indices.size(0), num_features, device=f1.features.device)
        f1_idx = inverse_indices[:f1.indices.size(0)]
        f2_idx = inverse_indices[f1.indices.size(0):]
        new_features[f1_idx, :f1.features.size(1)] = f1.features
        new_features[f2_idx, f1.features.size(1):] = f2.features
        new_spconvtensor = spconv.SparseConvTensor(new_features, concat_indices, f1.spatial_shape, f1.batch_size)
        return new_spconvtensor, new_spconvtensor.indices 

    else:
        return torch.cat([f1, f2], 1), None

def _make_sparse_tensor(feature_value, indices, is_sparse=True, project=False):
    if not is_sparse:
        return feature_value
    else:
        _, fc, fh, fw = feature_value.shape
        if not project:
            sparse_y = indices[:, 1]
            sparse_x = indices[:, 2]
        else:
            y = indices[:, 1]
            x = indices[:, 2]
            sparse_y, sparse_x = [], []
            for i in range(2):
                for j in range(2):
                    sparse_y.append(y * 2 + i)
                    sparse_x.append(x * 2 + j)

            sparse_y = torch.cat(sparse_y, dim=0)
            sparse_x = torch.cat(sparse_x, dim=0)
        
        sparse_inds = (sparse_y * fw + sparse_x).long()
        sparse_features = feature_value.view(fc, -1).transpose(0, 1)[sparse_inds].view(-1, fc)
        sparse_indices  = torch.stack((torch.zeros_like(sparse_y), sparse_y, sparse_x), dim=-1)  
        sparse_tensor = spconv.SparseConvTensor(sparse_features, sparse_indices.int(), (fh, fw), 1)
        return sparse_tensor