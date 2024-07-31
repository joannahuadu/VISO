import spconv.pytorch as spconv
import torch
import torch.nn as nn
from mmyolo.models.layers import DarknetBottleneck
device='cuda:0'
from yolo_world.models.sputils import SPInfer

# x_d = torch.zeros((1, 128, 256, 256))
channel = 64
x_d = torch.zeros((1, channel, 512, 512))
# torch.Size([1, 256, 200, 304])
x_d[0,0,0:512,0:512] += 1.
x_d = x_d.to(device)
x = spconv.SparseConvTensor.from_dense(x_d.permute(0,2,3,1))

conv_sparse = spconv.SubMConv2d(channel, channel, kernel_size=3, stride=1, padding=1, indice_key="asd", algo=spconv.ConvAlgo.Native).to(device)
bn_sparse = nn.BatchNorm1d(channel, momentum=0.1).to(device)
conv_bn_relu_sparse = spconv.SparseSequential(conv_sparse, bn_sparse, nn.ReLU(inplace=True)).to(device)

conv_norm = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False, dilation=1).to(device)
bn_norm = nn.BatchNorm2d(channel, momentum=0.1).to(device)
conv_bn_relu_norm = nn.Sequential(conv_norm, bn_norm, nn.ReLU(inplace=True)).to(device)

for i in range(1000):
     print("round:", i)
     start_event = torch.cuda.Event(enable_timing=True)
     end_event = torch.cuda.Event(enable_timing=True)
     start_event.record()
     encoder_output1 = conv_bn_relu_norm(x_d)
     end_event.record()
     end_event.synchronize()
     elapsed_time_ms = start_event.elapsed_time(end_event)
     print(f"conv_bn_relu_norm time: {elapsed_time_ms} milliseconds")

     start_event = torch.cuda.Event(enable_timing=True)
     end_event = torch.cuda.Event(enable_timing=True)
     start_event.record()
     encoder_output = conv_bn_relu_sparse(x)
     end_event.record()
     end_event.synchronize()
     elapsed_time_ms = start_event.elapsed_time(end_event)
     print(f"conv_bn_relu_sparse time: {elapsed_time_ms} milliseconds")