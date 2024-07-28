import spconv.pytorch as spconv
import torch
import torch.nn as nn
from mmyolo.models.layers import DarknetBottleneck
device='cuda:0'
from yolo_world.models.sputils import SPInfer

x_d = torch.zeros((2, 4, 1024, 1024))
x_d[0,0,0:16,0:16] += 1.
x_d = x_d.to(device)
x = spconv.SparseConvTensor.from_dense(x_d.permute(0,2,3,1))

conv_sparse = spconv.SubMConv2d(4, 4, kernel_size=3, stride=1, padding=1, indice_key="asd", algo=spconv.ConvAlgo.Native).to(device)
bn_sparse = nn.BatchNorm1d(4, momentum=0.1).to(device)
conv_bn_relu_sparse = spconv.SparseSequential(conv_sparse, bn_sparse, nn.ReLU(inplace=True)).to(device)

conv_norm = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1).to(device)
bn_norm = nn.BatchNorm2d(4, momentum=0.1).to(device)
conv_bn_relu_norm = nn.Sequential(conv_norm, bn_norm, nn.ReLU(inplace=True)).to(device)

for i in range(10):
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
     
     
x_d = torch.zeros((1, 256, 128, 128))
x_d[0,0,0:8,0:8] += 1.
x_d = x_d.to(device)
test_module = DarknetBottleneck(
                256,
                256,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=True,
                use_depthwise=False,
                conv_cfg = None,
                norm_cfg = dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg = dict(type='SiLU', inplace=True))
test_module.to(device)
for i in range(10):
     print("round:", i)
     start_event = torch.cuda.Event(enable_timing=True)
     end_event = torch.cuda.Event(enable_timing=True)
     start_event.record()
     encoder_output1 = test_module(x_d)
     end_event.record()
     end_event.synchronize()
     elapsed_time_ms = start_event.elapsed_time(end_event)
     print(f"conv_bn_relu_norm time: {elapsed_time_ms} milliseconds")
     
     
x = spconv.SparseConvTensor.from_dense(x_d.permute(0,2,3,1))   
sp_infer = SPInfer(sp_type='vspconv')
sparse_module_name = ['conv1', 'conv2']
sparse_module_list = [getattr(test_module, name) for name in sparse_module_name]
for name, m in zip(sparse_module_name, sparse_module_list):
     sp_infer._replace_spinfer(name, m, test_module)

for i in range(10):
     print("round:", i)
     start_event = torch.cuda.Event(enable_timing=True)
     end_event = torch.cuda.Event(enable_timing=True)
     start_event.record()
     encoder_output = test_module(x)
     end_event.record()
     end_event.synchronize()
     elapsed_time_ms = start_event.elapsed_time(end_event)
     print(f"conv_bn_relu_sparse time: {elapsed_time_ms} milliseconds")