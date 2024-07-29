import spconv.pytorch as spconv
import torch
import torch.nn as nn
device='cuda:9'


chanel = 256
randn_ = False
H = 128
W = 128
if randn_:
     x_d = torch.randn(1, chanel, H, W)
     # 设置mask比例
     mask_ratio = 0.99

     # 计算需要mask的元素数量
     num_elements = x_d.numel()
     num_masked = int(num_elements * mask_ratio)

     mask = torch.ones(num_elements)
     mask[:num_masked] = 0
     mask = mask[torch.randperm(num_elements)]

     mask = mask.view(x_d.shape)
     x_d = x_d * mask
else:
     x_d = torch.zeros(1, chanel, H, W)
     # 随机生成一些物体，设置有物体的区域，正方形物体，给出x1,x2,y1,y2，表示两个对角点
     object_region = [
          [0, 24, 0, 24],
          [24, 48, 24, 48],
          [82, 106, 82, 106],
          [64,72,456,512],
          [128, 136, 128, 136],
          [256, 264, 256, 264],
          [300, 308, 300, 308],
          [400, 408, 400, 408]
     ]
     # 把object_region的每个值 512 * H
     for region in object_region:
          region[0] = region[0] * H//512
          region[1] = region[1] * H//512
          region[2] = region[2] * W//512
          region[3] = region[3] * W//512
     # 遍历所有的区域并填充随机值
     for region in object_region:
          x1, x2, y1, y2 = region
          random_tensor = torch.clip(torch.randn(1, chanel, x2-x1, y2-y1), 0, 1)
          x_d[:, :, x1:x2, y1:y2] += random_tensor

x_d = x_d.to(device)
x = spconv.SparseConvTensor.from_dense(x_d.permute(0,2,3,1))

conv_sparse = spconv.SubMConv2d(chanel, chanel, kernel_size=3, stride=1, padding=3//2, indice_key="asd", algo=spconv.ConvAlgo.Native).to(device)
bn_sparse = nn.BatchNorm1d(chanel, momentum=0.1).to(device)
conv_bn_relu_sparse = spconv.SparseSequential(conv_sparse, bn_sparse, nn.ReLU(inplace=True)).to(device)

conv_norm = nn.Conv2d(chanel, chanel, kernel_size=3, stride=1, padding=3//2, bias=False, dilation=1).to(device)
bn_norm = nn.BatchNorm2d(chanel, momentum=0.1).to(device)
conv_bn_relu_norm = nn.Sequential(conv_norm, bn_norm, nn.ReLU(inplace=True)).to(device)

for i in range(20000):
     # print("round:", i)
     if i % 1000 == 0:
          print("round:", i)
     # start_event = torch.cuda.Event(enable_timing=True)
     # end_event = torch.cuda.Event(enable_timing=True)
     # start_event.record()
     # encoder_output1 = conv_bn_relu_norm(x_d)
     # end_event.record()
     # torch.cuda.synchronize()
     # elapsed_time_ms = start_event.elapsed_time(end_event)
     # print(f"conv_bn_relu_norm time: {elapsed_time_ms} milliseconds")

     # start_event = torch.cuda.Event(enable_timing=True)
     # end_event = torch.cuda.Event(enable_timing=True)
     # start_event.record()
     encoder_output = conv_bn_relu_sparse(x)
     # end_event.record()
     
     # torch.cuda.synchronize()
     # elapsed_time_ms = start_event.elapsed_time(end_event)
     # print(f"conv_bn_relu_sparse time: {elapsed_time_ms} milliseconds")
