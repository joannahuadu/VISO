import torch
import os

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")
print(f"Local rank: {local_rank}, CUDA device: {device}, Actual GPU: {torch.cuda.current_device()}")
