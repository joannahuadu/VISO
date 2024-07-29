import torch
import torch.nn as nn
import torchsparse
from torch import nn
from torchsparse import nn as spnn

# Assume `device` is set to 'cpu' or 'cuda'
device='cuda:0'

# Prepare input data
channel = 256
x_d = torch.zeros((1, channel, 32, 32)).to(device)
x_d[0, 0, 0:40, 0:40] += 1.

# Convert dense tensor to sparse tensor
coords = torch.nonzero(x_d).int()
feats = x_d[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]
coords = coords[:, [0, 2, 3, 1]]  # Change to [batch, height, width, channel]

sparse_tensor = SparseTensor(coords=coords, feats=feats).to(device)
coords -= np.min(coords, axis=0, keepdims=True)
coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
coords = torch.tensor(coords, dtype=torch.int)
feats = torch.tensor(feats[indices], dtype=torch.float)
tensor = SparseTensor(coords=coords, feats=feats)
# Define sparse convolution layers
conv_bn_relu_sparse = nn.Sequential(
    spnn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
    spnn.BatchNorm(channel),
    spnn.ReLU(True),
).to(device)


# Define normal 2D convolution layers
conv_norm = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False).to(device)
bn_norm = nn.BatchNorm2d(channel, momentum=0.1).to(device)
relu_norm = nn.ReLU(inplace=True).to(device)

conv_bn_relu_norm = nn.Sequential(conv_norm, bn_norm, relu_norm).to(device)

# Timing events for performance measurement
for i in range(100):
    print("round:", i)
    
    # Timing for 2D convolution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    encoder_output1 = conv_bn_relu_norm(x_d)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"conv_bn_relu_norm time: {elapsed_time_ms} milliseconds")
    
    # Timing for sparse convolution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    encoder_output = conv_bn_relu_sparse(sparse_tensor)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"conv_bn_relu_sparse time: {elapsed_time_ms} milliseconds")
