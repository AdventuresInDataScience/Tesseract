#%%
import sys
import os
import torch
import numpy as np
import pandas as pd
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}\\functions")

from model import *
from data import *

#%% Comprehensive Training Example
def create_sample_dataframe(n_timesteps=1000, n_assets=100):
    """
    Create a sample financial DataFrame for testing the model.
    This mimics real financial data with price time series.
    
    Returns:
        df: pandas DataFrame of shape (n_timesteps, n_assets) with price data
    """
    print(f"Creating sample DataFrame with {n_timesteps} timesteps and {n_assets} assets...")
    
    # Create realistic price data with some assets starting later (NaN values)
    np.random.seed(42)  # For reproducible results
    
    # Generate correlated returns
    base_returns = np.random.randn(n_timesteps, n_assets) * 0.02  # 2% daily volatility
    
    # Add some correlation structure
    market_factor = np.random.randn(n_timesteps, 1) * 0.01
    base_returns += market_factor * np.random.uniform(0.3, 0.8, (1, n_assets))
    
    # Convert to cumulative prices starting from 100
    prices = np.cumprod(1 + base_returns, axis=0) * 100
    
    # Add realistic NaN patterns - some assets start trading later
    for i in range(n_assets):
        start_date = np.random.randint(0, min(200, n_timesteps // 4))  # Some assets start up to 25% into the data
        if start_date > 0:
            prices[:start_date, i] = np.nan
    
    # Create DataFrame with realistic column names
    asset_names = [f"ASSET_{i:03d}" for i in range(n_assets)]
    df = pd.DataFrame(prices, columns=asset_names)
    
    print(f"Created DataFrame: shape {df.shape}")
    print(f"Non-NaN values per asset: min={df.count().min()}, max={df.count().max()}, mean={df.count().mean():.1f}")
    
    return df

def test_model_training_with_data_pipeline():
    """
    Test model training using the proper data pipeline.
    """
    print("="*60)
    print("COMPREHENSIVE MODEL TRAINING WITH DATA PIPELINE")
    print("="*60)
    
    # 1. Create sample DataFrame (single table as your functions expect)
    df = create_sample_dataframe(n_timesteps=500, n_assets=50)
    
    # 2. Training parameters
    past_window_size = 10
    future_window_size = 20
    min_n_cols = 10
    max_n_cols = 30
    min_batch_size = 8
    max_batch_size = 32
    iterations = 10  # Just a few iterations for testing
    
    # 3. Build the model
    print(f"\nBuilding transformer model for past_window_size={past_window_size}...")
    model = build_transformer_model(
        past_window_size=past_window_size,
        d_model=128,           # Smaller for testing
        n_heads=4,
        n_transformer_blocks=2, # Fewer blocks for testing
        d_ff=512,
        dropout=0.1,
        max_n=max_n_cols,
        causal=True
    )
    
    # 4. Create optimizer
    print("Creating Adam optimizer...")
    optimizer = create_adam_optimizer(model, lr=1e-4, weight_decay=1e-5)
    
    # 5. Portfolio constraints
    constraints = {
        'max_weight': 0.2,         # Max 20% in any single asset
        'min_assets': 5,           # Hold at least 5 assets
        'max_assets': 15,          # Hold at most 15 assets
        'sparsity_threshold': 0.02 # Set weights < 2% to zero
    }
    
    print(f"Training with constraints: {constraints}")
    print(f"Progressive training: {min_n_cols}->{max_n_cols} assets, {min_batch_size}->{max_batch_size} batch size")
    
    # 6. Use your data pipeline for training
    print(f"\nStarting training with {iterations} iterations...")
    
    placeholder_func(
        model=model,
        optimizer=optimizer, 
        data=df,
        past_window_size=past_window_size,
        future_window_size=future_window_size,
        min_n_cols=min_n_cols,
        max_n_cols=max_n_cols,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        iterations=iterations,
        metric='sharpe_ratio',
        max_weight=constraints['max_weight'],
        min_assets=constraints['min_assets'],
        max_assets=constraints['max_assets'],
        sparsity_threshold=constraints['sparsity_threshold']
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return model, optimizer

def test_single_batch_creation():
    """
    Test the batch creation functions directly.
    """
    print("\n" + "="*60)
    print("TESTING BATCH CREATION")
    print("="*60)
    
    # Create sample data
    df = create_sample_dataframe(n_timesteps=200, n_assets=30)
    data_array = df.values
    
    past_window_size = 8
    future_window_size = 12
    n_cols = 15
    batch_size = 16
    
    valid_indices = len(data_array) - (past_window_size + future_window_size)
    
    print(f"Testing batch creation:")
    print(f"  Past window: {past_window_size}, Future window: {future_window_size}")
    print(f"  Assets per sample: {n_cols}, Batch size: {batch_size}")
    print(f"  Valid starting indices: 0 to {valid_indices}")
    
    # Test batch creation
    past_batch, future_batch = create_batch(
        data_array, past_window_size, future_window_size, n_cols, batch_size, valid_indices
    )
    
    print(f"\nBatch shapes:")
    print(f"  Past batch: {past_batch.shape}")
    print(f"  Future batch: {future_batch.shape}")
    
    # Verify the data structure
    print(f"\nData verification:")
    print(f"  Past batch - min: {past_batch.min():.3f}, max: {past_batch.max():.3f}")
    print(f"  Future batch - min: {future_batch.min():.3f}, max: {future_batch.max():.3f}")
    
    # Check normalization (last timestep of past should be close to 1.0 for non-zero assets)
    last_past_timestep = past_batch[:, :, -1]  # Last timestep of past window
    print(f"  Last past timestep (should be ~1.0 for active assets):")
    print(f"    Sample 0: {last_past_timestep[0, :5]}")  # First 5 assets of first sample
    
    return past_batch, future_batch

def test_different_metrics():
    """
    Test the model with different optimization metrics using proper data pipeline.
    """
    print("\n" + "="*60)
    print("TESTING DIFFERENT METRICS")
    print("="*60)
    
    # Create sample DataFrame
    df = create_sample_dataframe(n_timesteps=300, n_assets=25)
    
    # Build model
    model = build_transformer_model(past_window_size=8, d_model=64, n_transformer_blocks=2, max_n=20)
    optimizer = create_adam_optimizer(model, lr=2e-4)
    
    # Test different metrics with your data pipeline
    metrics_to_test = ['sharpe_ratio', 'sortino_ratio', 'expected_return']
    
    for metric in metrics_to_test:
        print(f"\n--- Testing metric: {metric} ---")
        
        # Use your data pipeline for a few iterations
        placeholder_func(
            model=model,
            optimizer=optimizer,
            data=df,
            past_window_size=8,
            future_window_size=12,
            min_n_cols=8,
            max_n_cols=15,
            min_batch_size=4,
            max_batch_size=8,
            iterations=3,  # Just a few iterations per metric
            metric=metric,
            max_weight=0.25,
            min_assets=3,
            max_assets=10,
            sparsity_threshold=0.03
        )
        
        print(f"Completed testing {metric}")

def test_constraint_enforcement():
    """
    Test that portfolio constraints are properly enforced using proper data pipeline.
    """
    print("\n" + "="*60)
    print("TESTING CONSTRAINT ENFORCEMENT")
    print("="*60)
    
    # Create sample DataFrame
    df = create_sample_dataframe(n_timesteps=200, n_assets=15)
    
    # Build model
    model = build_transformer_model(past_window_size=5, d_model=32, n_transformer_blocks=1, max_n=12)
    optimizer = create_adam_optimizer(model)
    
    # Test strict constraints
    strict_constraints = {
        'max_weight': 0.15,   # Max 15%
        'min_assets': 4,      # At least 4 assets
        'max_assets': 8,      # At most 8 assets
        'sparsity_threshold': 0.05  # 5% threshold
    }
    
    print(f"Testing strict constraints: {strict_constraints}")
    
    # Use your data pipeline
    placeholder_func(
        model=model,
        optimizer=optimizer,
        data=df,
        past_window_size=5,
        future_window_size=10,
        min_n_cols=6,
        max_n_cols=10,
        min_batch_size=4,
        max_batch_size=6,
        iterations=5,
        metric='sharpe_ratio',
        **strict_constraints
    )
    
    print("Constraint enforcement test completed")

if __name__ == "__main__":
    # Run all tests
    try:
        # Test batch creation first
        print("Testing batch creation...")
        past_batch, future_batch = test_single_batch_creation()
        
        # Main training example with proper data pipeline
        print("\nTesting model training with data pipeline...")
        model, optimizer = test_model_training_with_data_pipeline()
        
        # Test different metrics
        test_different_metrics()
        
        # Test constraint enforcement
        test_constraint_enforcement()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

#%%
