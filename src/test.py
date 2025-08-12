#%%
import sys
import os
import torch
import numpy as np
import pandas as pd
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}\\functions")

from model import *
from data import *

# #%%
# # Setting up the environment for S&P 500 stock prices
# df = pd.read_parquet("C:/Users/malha/Documents/Data/S&P500 components/stock_prices.parquet")

# # %%
# # One off to generate correct format for wide table
# df = df[['symbol', 'report_date', 'close']]
# # Pivot the DataFrame to have report_date as index and symbols as columns
# pivoted_df = df.pivot(index='report_date', columns='symbol', values='close').reset_index()
# pivoted_df = pivoted_df.sort_values(by='report_date')
# pivoted_df.head()
# pivoted_df.to_parquet("C:/Users/malha/Documents/Data/S&P500 components/stocks_wide.parquet", compression= "gzip", index=False)
# %% -model setup
df = pd.read_parquet("C:/Users/malha/Documents/Data/S&P500 components/stocks_wide.parquet")
timeseries = df.drop(columns=['report_date']).values

# Define parameters
past_window_size = 30
future_window_size = 5

#1. Build Model
n_assets = timeseries.shape[1]
model = build_transformer_model(
    past_window_size=past_window_size,
    d_model=256,
    n_heads=8,
    n_transformer_blocks=6,
    d_ff=1024,
    max_n=n_assets
)

#2. Build Optimizer
optimizer = create_adam_optimizer(model)

#3. Portfolio constraints - These are now ranges for random sampling during training!
# The training loop will randomly sample within these ranges for each iteration
max_weight_range = (0.05, 0.5)  # Max individual asset weight: 5%-50%
min_assets_range = (5, 30)       # Minimum number of assets in portfolio: 5-30
max_assets_range = (20, 100)     # Maximum number of assets in portfolio: 20-100  
sparsity_threshold_range = (0.001, 0.05)  # Sparsity threshold: 0.1%-5%

#4. Test data pipeline for training with random constraint sampling
past_window_size = 30
future_window_size = 5
# %% - Test Training
# Train the model with random constraint sampling
print("Starting training with random constraint sampling...")
trained_model = train_model(
    model=model,
    optimizer=optimizer,
    data=timeseries,
    past_window_size=past_window_size,
    future_window_size=future_window_size,
    min_n_cols=10,
    max_n_cols=50,
    min_batch_size=16,
    max_batch_size=64,
    iterations=100,
    metric='sharpe_ratio',
    # Constraint ranges for random sampling
    max_weight_range=max_weight_range,
    min_assets_range=min_assets_range,
    max_assets_range=max_assets_range,
    sparsity_threshold_range=sparsity_threshold_range
)

print("Training completed! Model has been trained on diverse constraint combinations.")

# %% Test Inference with specific constraints
#5. Test inference with specific constraints
print("\nTesting inference with specific user-defined constraints...")

# Example: Conservative portfolio constraints
conservative_constraints = {
    'max_weight': 0.1,      # Max 10% in any single asset
    'min_assets': 15,       # At least 15 assets
    'max_assets': 50,       # At most 50 assets  
    'sparsity_threshold': 0.005  # 0.5% threshold for inclusion
}

# Prepare data for prediction - the function now handles data cleaning automatically
print("Using cleaned DataFrame for prediction...")
print(f"Original timeseries shape: {timeseries.shape}")
print(f"Original timeseries dtype: {timeseries.dtype}")

# Convert to DataFrame for easier handling (the function will handle all the cleaning)
df_for_prediction = pd.DataFrame(timeseries)

# Use the predict_portfolio_weights function with automatic data handling
portfolio_weights = predict_portfolio_weights(
    model=trained_model,
    data_input=df_for_prediction,  # Pandas DataFrame - automatically handled
    future_window_size=future_window_size,
    max_assets_subset=50,  # Use only first 50 assets
    **conservative_constraints
)

print(f"Conservative portfolio weights shape: {portfolio_weights.shape}")
print(f"Max weight: {portfolio_weights.max():.4f}")
print(f"Number of non-zero weights: {(portfolio_weights > conservative_constraints['sparsity_threshold']).sum()}")
print(f"Sum of weights: {portfolio_weights.sum():.4f}")

# Example: Aggressive portfolio constraints  
aggressive_constraints = {
    'max_weight': 0.3,      # Max 30% in any single asset
    'min_assets': 5,        # At least 5 assets
    'max_assets': 15,       # At most 15 assets
    'sparsity_threshold': 0.02  # 2% threshold for inclusion
}

# Use the same DataFrame for aggressive prediction
portfolio_weights_aggressive = predict_portfolio_weights(
    model=trained_model,
    data_input=df_for_prediction,  # Same DataFrame - automatically handled
    future_window_size=future_window_size,
    max_assets_subset=50,  # Use only first 50 assets
    **aggressive_constraints
)

print(f"\nAggressive portfolio weights shape: {portfolio_weights_aggressive.shape}")
print(f"Max weight: {portfolio_weights_aggressive.max():.4f}")
print(f"Number of non-zero weights: {(portfolio_weights_aggressive > aggressive_constraints['sparsity_threshold']).sum()}")
print(f"Sum of weights: {portfolio_weights_aggressive.sum():.4f}")

print("\nModel successfully handles different constraint combinations!")

# %%
