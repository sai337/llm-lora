from __future__ import annotations
import sys, platform
import torch
import transformers
import datasets

print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
