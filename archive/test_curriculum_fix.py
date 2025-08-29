"""
Quick test to verify the curriculum training division by zero fix
"""
import sys
import os
import torch
import numpy as np
import pandas as pd

sys.path.append('functions')
from functions import *

print('=== Testing Curriculum Training Fix ===')

# Load minimal data for testing
df = pd.read_parquet('C:/Users/malha/Documents/Data/S&P500 components/stocks_wide.parquet')
timeseries = df.drop(columns=['report_date']).values
train_data = timeseries[:100]  # Just 100 rows for quick test

print(f"Loaded data shape: {train_data.shape}")

# Build minimal model
model = build_transformer_model(
    past_window_size=65,
    d_model=256,
    n_heads=8,
    n_transformer_blocks=6,
    d_ff=1024,
    max_n=timeseries.shape[1],
    activation='hard_mish'
)

optimizer = create_adam_optimizer(model, lr=1e-3, weight_decay=2e-4)
print("Model and optimizer created successfully")

# Test with minimal iteration count
debug_batch_schedule = {32: 1}
debug_column_schedule = {1: 1}
constraint_config = {
    'constraint_n_steps': 5,
    'max_weight_range': (0.05, 0.4),
    'min_assets_range': (3, 10),
    'max_assets_range': (10, 100),
    'sparsity_threshold_range': (0.001, 0.05),
    'future_window_range': (5, 50)
}

print("Starting curriculum training test...")
try:
    test_model = train_model_curriculum(
        model=model,
        optimizer=optimizer,
        data=train_data,
        past_window_size=65,
        batch_schedule=debug_batch_schedule,
        column_schedule=debug_column_schedule,
        **constraint_config,
        iterations=1,
        loss='expected_return',
        loss_aggregation='progressive',
        learning_rate=1e-3,
        weight_decay=2e-4,
        warmup_steps=1,
        checkpoint_frequency=1,
        log_frequency=1
    )
    print('SUCCESS: Single iteration test completed!')
    print('The division by zero issue has been fixed!')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
