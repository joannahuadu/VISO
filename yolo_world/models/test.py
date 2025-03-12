import torch
import spconv.pytorch as spconv

# Define the shape of the tensor
shape = (1, 28, 28, 64)  # Example shape: (batch_size, height, width)

# Create a random tensor
x = torch.randn(shape)

# Randomly set some positions to 0 (e.g., 30% of the elements)
mask = torch.rand(shape) > 0.3  # Create a mask with approximately 70% of the values being True
x = x * mask  # Apply the mask to set some elements to 0

# Convert to SparseConvTensor
x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 28, 28, 64))

print(x_sp)
