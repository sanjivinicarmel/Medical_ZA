import torch

# Create a tensor on GPU
x = torch.rand(1000, 1000).cuda()
y = torch.rand(1000, 1000).cuda()

# Multiply on GPU
z = torch.matmul(x, y)

print("✅ GPU computation successful!")
print(f"Result shape: {z.shape}")
print(f"Device: {z.device}")