
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

#%%
# Test with data loader (requires fixing the data module)
from data import create_time_series_batches
import pandas as pd

# Create sample DataFrame to test data loading
sample_df = pd.DataFrame(torch.randn(1000, 50).numpy())  # 1000 timesteps, 50 assets

try:
    # Test the data loader with corrected dimensions
    matrix_batch, sample_info = create_time_series_batches(
        df=sample_df,
        t_fixed=100,  # Half of model's t_fixed since data loader does t_fixed*2
        n_cols=10,    # This becomes 'n' dimension
        batch_size=4
    )
    
    print(f"\nData loader test:")
    print(f"Matrix batch shape: {matrix_batch.shape}")
    print(f"Expected model input shape: (batch_size, n, t_fixed) = (4, 10, 200)")
    
    # Test with model
    scalar_batch = torch.randn(4, 1)
    with torch.no_grad():
        output = model(matrix_batch, scalar_batch)
        print(f"Model output shape: {output.shape}")
        print("✅ Data loader compatible with model!")
        
except Exception as e:
    print(f"❌ Data loader needs adjustment: {e}")

# %%
