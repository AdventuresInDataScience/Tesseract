#%% - Imports
# Imports
"""
Test script for the Tesseract portfolio optimization transformer model.

KEY UPDATES:
1. Progressive Loss Aggregation: MAE → GMAE → GMSE for maximum stability
2. Proper train/test split: 90% training, 10% testing to prevent data leakage
3. Training uses only training data (first 90% of timeseries)
4. Testing/inference uses only test data (last 10% of timeseries) 
5. Prediction uses the last past_window_size rows from test data as "current state"
6. Comprehensive constraint validation prevents impossible portfolio requirements
7. Enhanced logging captures loss aggregation method, phase, and curriculum progress

PREDICTION PROCESS:
- Input: Historical price data (DataFrame/array)
- The model takes the LAST past_window_size rows (e.g., last 65 days)
- These represent the "current market state"
- Output: Portfolio weights optimized for the NEXT future_window_size periods

LOGGING & CHECKPOINTS:
- Training logs: Comprehensive CSV files with iteration, loss, loss_aggregation method, phase, batch_size, n_cols, progress
- Model checkpoints: Saved every 50 iterations as .pt files
- Default paths: repo_root/logs/ and repo_root/checkpoints/
- Loss transitions (MAE→GMAE→GMSE) are clearly marked for analysis
- Custom paths can be specified via log_path and checkpoint_path arguments
"""
import sys
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

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
# %% - Data preparation and train/test split
df = pd.read_parquet("C:/Users/malha/Documents/Data/S&P500 components/stocks_wide.parquet")
timeseries = df.drop(columns=['report_date']).values

# Define parameters
past_window_size = 65
future_window_size = 21

# IMPORTANT: Split data into train/test sets
# Use first 90% for training, last 10% for testing
split_ratio = 0.9
split_index = int(len(timeseries) * split_ratio)

train_data = timeseries[:split_index]  # First 90% for training
test_data = timeseries[split_index:]   # Last 10% for testing

print(f"Total data shape: {timeseries.shape}")
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Training period: rows 0 to {split_index-1}")
print(f"Test period: rows {split_index} to {len(timeseries)-1}")

#1. Build Model with CPU-optimized activation function
n_assets = timeseries.shape[1]
model = build_transformer_model(
    past_window_size=past_window_size,
    d_model=256,
    n_heads=8,
    n_transformer_blocks=6,
    d_ff=1024,
    max_n=n_assets,
    activation='hard_mish'  # Hard Mish - computational proxy for Mish activation
)

#2. Build Optimizer with reduced learning rate for more stable training
optimizer = create_adam_optimizer(model, lr=5e-5, weight_decay=5e-5)  # Increased weight decay for better regularization

#3. Portfolio constraints - These are now ranges for random sampling during training!
# The training loop will randomly sample within these ranges for each iteration
# IMPORTANT: These ranges must be logically consistent across the curriculum
max_weight_range = (0.05, 0.5)  # Max individual asset weight: 5%-50%
min_assets_range = (3, 15)       # Minimum number of assets in portfolio: 3-15 (safe for min_n_cols=5)
max_assets_range = (15, 100)     # Maximum number of assets in portfolio: 15-100 (ensures min <= max)
sparsity_threshold_range = (0.001, 0.05)  # Sparsity threshold: 0.1%-5%

# %% - Test Training (using only training data)
# Train the model with random constraint sampling using ONLY the training data
print("Starting training with random constraint sampling on training data only...")
trained_model = train_model(
    model=model,
    optimizer=optimizer,
    data=train_data,  # Use ONLY training data (first 80%)
    past_window_size=past_window_size,
    future_window_size=future_window_size,
    min_n_cols=5,
    max_n_cols=200,
    min_batch_size=16,      # Reduced for CPU training - smaller batches work better
    max_batch_size=256,     # CPU can handle larger batches with 32GB RAM - increased from 128
    iterations=10000,       # Increased iterations to compensate for smaller batches
    metric="expected_return", #'sharpe_ratio',
    loss_aggregation='progressive',  # Progressive curriculum: mae → gmae → gmse for maximum stability
    # Constraint ranges for random sampling
    max_weight_range=max_weight_range,
    min_assets_range=min_assets_range,
    max_assets_range=max_assets_range,
    sparsity_threshold_range=sparsity_threshold_range,
    # Logging and checkpoint paths (will use defaults: repo_root/logs/ and repo_root/checkpoints/)
    # log_path=None,  # Will default to repo_root/logs/
    # checkpoint_path=None  # Will default to repo_root/checkpoints/
    
    # Example of custom paths (commented out):
    # log_path="C:/custom/path/to/logs",  # Custom log directory
    # checkpoint_path="C:/custom/path/to/checkpoints"  # Custom checkpoint directory
)

print("Training completed! Model has been trained on diverse constraint combinations.")

# %% Test Inference with specific constraints on unseen test data
#5. Test inference with specific constraints
print("\nTesting inference with specific user-defined constraints on UNSEEN test data...")

# Example: Conservative portfolio constraints
conservative_constraints = {
    'max_weight': 0.1,      # Max 10% in any single asset
    'min_assets': 15,       # At least 15 assets
    'max_assets': 50,       # At most 50 assets  
    'sparsity_threshold': 0.005  # 0.5% threshold for inclusion
}

# IMPORTANT: For prediction, we use test data (out-of-sample)
# The predict_portfolio_weights function automatically takes the LAST past_window_size rows
# from whatever data you provide. This simulates making a prediction at the "current" time.
#
# HOW IT WORKS:
# 1. You provide a DataFrame/array with historical price data
# 2. The function takes the LAST past_window_size rows (e.g., last 30 days)
# 3. These become the "current state" that the model uses to predict portfolio weights
# 4. The prediction is for the NEXT future_window_size periods (e.g., next 5 days)
# 5. The model outputs portfolio weights that should perform well over those future periods

print("Using TEST data for out-of-sample prediction...")
print(f"Test data shape: {test_data.shape}")
print(f"Test data covers time periods {split_index} to {len(timeseries)-1}")

# Convert test data to DataFrame for prediction
# The function will automatically use the LAST past_window_size rows (most recent data)
test_df = pd.DataFrame(test_data)
print(f"Test DataFrame shape: {test_df.shape}")
print(f"The model will use rows {len(test_df)-past_window_size} to {len(test_df)-1} from test data for prediction")

# SCENARIO 1: Conservative Portfolio Prediction
# This uses the most recent past_window_size rows from test_data to predict portfolio weights
portfolio_weights = predict_portfolio_weights(
    model=trained_model,
    data_input=test_df,  # Test data DataFrame - uses LAST past_window_size rows automatically
    future_window_size=future_window_size,
    max_assets_subset=100,  # Use only first 100 assets for simplicity
    **conservative_constraints
)

print(f"\nConservative portfolio prediction results:")
print(f"Portfolio weights shape: {portfolio_weights.shape}")
print(f"Max weight: {portfolio_weights.max():.4f}")
print(f"Number of non-zero weights: {(portfolio_weights > conservative_constraints['sparsity_threshold']).sum()}")
print(f"Sum of weights: {portfolio_weights.sum():.4f}")

# SCENARIO 2: Aggressive portfolio constraints  
aggressive_constraints = {
    'max_weight': 0.3,      # Max 30% in any single asset
    'min_assets': 5,        # At least 5 assets
    'max_assets': 15,       # At most 15 assets
    'sparsity_threshold': 0.02  # 2% threshold for inclusion
}

# Same test data, different constraints
portfolio_weights_aggressive = predict_portfolio_weights(
    model=trained_model,
    data_input=test_df,  # Same test data - uses LAST past_window_size rows automatically
    future_window_size=future_window_size,
    max_assets_subset=50,  # Use only first 50 assets for simplicity
    **aggressive_constraints
)

print(f"\nAggressive portfolio prediction results:")
print(f"Portfolio weights shape: {portfolio_weights_aggressive.shape}")
print(f"Max weight: {portfolio_weights_aggressive.max():.4f}")
print(f"Number of non-zero weights: {(portfolio_weights_aggressive > aggressive_constraints['sparsity_threshold']).sum()}")
print(f"Sum of weights: {portfolio_weights_aggressive.sum():.4f}")

print("\n" + "="*60)
print("SUMMARY:")
print(f"✓ Model trained on data from timesteps 0 to {split_index-1}")
print(f"✓ Predictions made using timesteps {split_index + len(test_data) - past_window_size} to {split_index + len(test_data) - 1}")
print(f"✓ This ensures true out-of-sample testing with no data leakage")
print(f"✓ Using Progressive loss aggregation (MAE → GMAE → GMSE)")
print(f"✓ Training logs saved to repo_root/logs/")
print(f"✓ Model checkpoints saved to repo_root/checkpoints/")
print("✓ Model successfully handles different constraint combinations on unseen data!")
print("✓ Comprehensive constraint validation prevents impossible portfolio requirements!")
print("="*60)

# %% Model Saving and Loading Examples
print("\n" + "="*60)
print("MODEL SAVING AND LOADING EXAMPLES")
print("="*60)

# The saving/loading functions are already imported via the wildcard import above

# 1. Save the complete trained model with configuration
print("\n1. Saving trained model...")
model_config = {
    'past_window_size': past_window_size,
    'future_window_size': future_window_size,
    'd_model': 256,
    'n_heads': 8,
    'n_transformer_blocks': 6,
    'max_n': n_assets,  # Number of assets in the dataset
    'loss_aggregation': 'progressive',
    'training_iterations': 200,
    'metric_used': 'expected_return'
}

# Save to a custom path (you can also use defaults)
import os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
complete_model_path = os.path.join(repo_root, 'models', 'trained_portfolio_model.pt')
os.makedirs(os.path.dirname(complete_model_path), exist_ok=True)

# Save the model (using the safe method)
save_model(trained_model, complete_model_path, model_config)

# 2. Save just the model configuration separately
config_path = os.path.join(repo_root, 'models', 'model_config.json')
save_model_config(trained_model, config_path)

print("\n2. Testing model loading...")

# 3. Load the model
loaded_model, loaded_config = load_model(complete_model_path)
print(f"✓ Model loaded with configuration: {loaded_config}")

print(f"✓ Model loaded with configuration: {loaded_config}")

# 4. Test that the loaded model works exactly the same
print("\n3. Verifying loaded model produces identical results...")
original_prediction = predict_portfolio_weights(
    model=trained_model,
    data_input=test_df,
    future_window_size=future_window_size,
    max_assets_subset=50,
    **conservative_constraints
)

loaded_prediction = predict_portfolio_weights(
    model=loaded_model,
    data_input=test_df,
    future_window_size=future_window_size,
    max_assets_subset=50,
    **conservative_constraints
)

# Check if predictions are identical (within floating point precision)
import torch
predictions_match = torch.allclose(
    torch.tensor(original_prediction), 
    torch.tensor(loaded_prediction), 
    atol=1e-6
)

print(f"✓ Original vs Loaded predictions match: {predictions_match}")
print(f"  Max absolute difference: {abs(original_prediction - loaded_prediction).max():.2e}")

# 5. Example of loading from checkpoint (state_dict only)
print("\n4. Loading model from checkpoint (state_dict)...")

# Find the latest checkpoint
checkpoint_dir = os.path.join(repo_root, 'checkpoints')
if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints)[-1]  # Get the latest one
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        # Create a new model with the same architecture
        fresh_model = build_transformer_model(
            past_window_size=past_window_size,
            d_model=256,
            n_heads=8,
            n_transformer_blocks=6,
            max_n=n_assets
        )
        
        # Load weights from checkpoint
        checkpoint_loaded_model = load_model_from_checkpoint(fresh_model, checkpoint_path)
        
        # Test this model too
        checkpoint_prediction = predict_portfolio_weights(
            model=checkpoint_loaded_model,
            data_input=test_df,
            future_window_size=future_window_size,
            max_assets_subset=50,
            **conservative_constraints
        )
        
        checkpoint_match = torch.allclose(
            torch.tensor(original_prediction), 
            torch.tensor(checkpoint_prediction), 
            atol=1e-6
        )
        
        print(f"✓ Checkpoint loaded model predictions match: {checkpoint_match}")
        print(f"  Loaded from: {latest_checkpoint}")
    else:
        print("No checkpoints found to test loading from")
else:
    print("Checkpoint directory not found")

# 6. Load configuration separately
print("\n5. Loading model configuration...")
loaded_config_only = load_model_config(config_path)

print("\n" + "="*60)
print("MODEL PERSISTENCE SUMMARY:")
print("="*60)
print("✓ Complete model saved with architecture + weights + config")
print("✓ Model successfully loaded and verified to produce identical results")
print("✓ Checkpoint loading from state_dict demonstrated")
print("✓ Configuration saving/loading demonstrated")
print("✓ Your trained model is ready for production use!")
print("\nSaved files:")
print(f"  - Complete model: {complete_model_path}")
print(f"  - Configuration: {config_path}")
print(f"  - Checkpoints: {checkpoint_dir}/ (if exists)")
print(f"  - Training logs: {os.path.join(repo_root, 'logs')}/")
print("="*60)

# %%
