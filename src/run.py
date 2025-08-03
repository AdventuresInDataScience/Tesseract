
#%%
import sys
import os
import torch
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}\\functions")

from model import build_transformer_model

#%%
model = build_transformer_model(t_fixed = 200)

#%%
# Create sample data to test the model
# Sample parameters
batch_size = 4
n = 10  # number of time series
t = 200  # fixed time dimension (matches t_fixed)

# Create sample matrix input (batch_size, n, t)
matrix_input = torch.randn(batch_size, n, t)

# Create sample scalar input (batch_size, 1)
scalar_input = torch.randn(batch_size, 1)

print(f"Matrix input shape: {matrix_input.shape}")
print(f"Scalar input shape: {scalar_input.shape}")

#%%
# Test forward pass
with torch.no_grad():
    output = model(matrix_input, scalar_input)
    
print(f"Output shape: {output.shape}")
print(f"Output (first batch, first 5 values): {output[0, :5]}")
print(f"Output sum per batch (should be ~1.0 due to softmax): {output.sum(dim=1)}")
print(f"Model has {model.get_num_parameters():,} parameters")

# %%
