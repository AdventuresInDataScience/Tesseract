
#%%
import sys
import os
import torch
import numpy as np
import pandas as pd
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}\\functions")

from model import *
from data import *

#%%
model = build_transformer_model(past_window_size = 200,
                                n_transformer_blocks=4)

#%%
# Create sample data to test the model
# Sample parameters
batch_size = 4
n = 10  # number of time series
t = 200  # fixed time dimension (matches past_window_size)

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
#build some sample data and test the functions
example_data = np.random.randn(1000, 50)  # 1000 timesteps, 50 assets

# Test training with random constraints
print("Starting training with random constraint sampling...")

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Test training function
train_model(
    model=model,
    optimizer=optimizer,
    data=example_data, 
    past_window_size=100, 
    future_window_size=50,
    min_n_cols=5, 
    max_n_cols=15, 
    min_batch_size=4, 
    max_batch_size=8, 
    iterations=5  # Reduced for testing
)

print("Training completed successfully!")
# %%
# test create_portfolio_time_series function
# Example 1: NumPy stocks matrix (from pandas.values) + PyTorch weights (from model output)
stocks_matrix = np.random.randn(252, 10)  # 252 trading days, 10 stocks
weights_vector = torch.softmax(torch.randn(10), dim=0)  # Model output (normalized to sum=1)

print(f"Test 1 - Real scenario:")
print(f"  stocks_matrix type: {type(stocks_matrix)}, shape: {stocks_matrix.shape}")
print(f"  weights_vector type: {type(weights_vector)}, shape: {weights_vector.shape}")
print(f"  weights sum: {weights_vector.sum():.4f} (should be ~1.0)")

portfolio_returns = create_portfolio_time_series(stocks_matrix, weights_vector)
print(f"  portfolio_returns type: {type(portfolio_returns)}, shape: {portfolio_returns.shape}")
print(f"  portfolio mean return: {portfolio_returns.mean():.4f}")
print(f"  portfolio std: {portfolio_returns.std():.4f}")
print("  ✅ Success!\n")

# Example 2: Test with equal weights (should work like simple average)
equal_weights = torch.ones(10) / 10  # Equal weights: [0.1, 0.1, ..., 0.1]
portfolio_equal = create_portfolio_time_series(stocks_matrix, equal_weights)
manual_average = torch.from_numpy(stocks_matrix).float().mean(dim=1)  # Manual average

print(f"Test 2 - Equal weights validation:")
print(f"  Equal weights portfolio mean: {portfolio_equal.mean():.4f}")
print(f"  Manual average mean: {manual_average.mean():.4f}")
print(f"  Difference: {torch.abs(portfolio_equal - manual_average).max():.6f} (should be ~0)")
print("  ✅ Equal weights test passed!\n")

# Example 3: Test with extreme weights (all weight on first stock)
extreme_weights = torch.zeros(10)
extreme_weights[0] = 1.0  # 100% weight on first stock
portfolio_extreme = create_portfolio_time_series(stocks_matrix, extreme_weights)
first_stock_returns = torch.from_numpy(stocks_matrix[:, 0]).float()

print(f"Test 3 - Extreme weights validation:")
print(f"  100% first stock portfolio mean: {portfolio_extreme.mean():.4f}")
print(f"  First stock returns mean: {first_stock_returns.mean():.4f}")
print(f"  Difference: {torch.abs(portfolio_extreme - first_stock_returns).max():.6f} (should be ~0)")
print("  ✅ Extreme weights test passed!\n")

print("All tests completed successfully! 🎉")

# %%
#%%
#