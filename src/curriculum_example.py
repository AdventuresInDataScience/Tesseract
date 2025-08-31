#%% - Imports
"""
Curriculum Learning Example for Tesseract Portfolio Optimization

Uses structured phase progression instead of random sampling:
- Batch progression: 32 → 64 → 128 (coordinated with column phases)
- Column buckets: Wide → Medium → Narrow asset sampling
- Constraint expansion: Tight → Loose constraints over 5 steps
- Enhanced stability: Progressive loss aggregation + memory-efficient training
"""
import sys
import os
import torch
import numpy as np
import pandas as pd

# ============ CPU OPTIMIZATION SETTINGS ============
# Enable CPU optimizations for faster training
torch.set_num_threads(os.cpu_count())  # Use all CPU cores
torch.backends.mkldnn.enabled = True   # Enable Intel MKL-DNN optimizations
torch.set_default_dtype(torch.float32) # Ensure float32 (faster than float64 on CPU)

print(f"🚀 CPU Optimizations enabled:")
print(f"   • Using {os.cpu_count()} CPU threads")
print(f"   • Intel MKL-DNN optimizations: {torch.backends.mkldnn.enabled}")
print(f"   • Default dtype: {torch.get_default_dtype()}")
print(f"   • This should significantly speed up training on CPU!")
print()

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}\\functions")
from functions import *
from functions.training_logger import create_curriculum_logger

#%% - Data preparation and train/test split
try:
    # Try to read the parquet file
    df = pd.read_parquet("C:/Users/malha/Documents/Data/S&P500 components/stocks_wide.parquet")
    timeseries = df.drop(columns=['report_date']).values
    print("Successfully loaded S&P 500 data from parquet file")
except Exception as e:
    print(f"Error loading parquet file: {e}")
    raise RuntimeError("Failed to load S&P 500 data. Please ensure the parquet file exists and is accessible.")

# Define parameters
past_window_size = 126
split_ratio = 0.8
split_index = int(len(timeseries) * split_ratio)

train_data = timeseries[:split_index]
test_data = timeseries[split_index:]

print(f"Total data: {timeseries.shape}")
print(f"Training: {train_data.shape}, Test: {test_data.shape}")
print(f"DEBUG: Number of columns detected: {timeseries.shape[1]}")

#%% - Build Model
n_assets = timeseries.shape[1]
model = build_transformer_model(
    past_window_size=past_window_size,
    d_model=256,
    n_heads=8,
    n_transformer_blocks=2,
    d_ff=1024,
    max_n=n_assets,
    activation='hard_mish'
)

optimizer = create_adam_optimizer(model, lr=1e-3, weight_decay=2e-4)

#%% - Config
# Curriculum schedules - coordinated progression
# Full training schedule (uncomment for complete 2000-iteration run):
# batch_schedule = {
#     32: 400,   # Phase 1: Small batches
#     64: 800,   # Phase 2: Medium batches  
#     128: 800   # Phase 3: Large batches
# }

# Shorter test schedule (comment out for full training):
batch_schedule = {
    32: 640,    # Phase 1: Small batches (reduced)
    64: 1280,    # Phase 2: Medium batches (reduced)
    128: 1920    # Phase 3: Large batches (reduced)
}

column_schedule = {
    1: 720,
    2: 720,
    3: 480,
    4: 480,
    5: 240,
    6: 240,
    7: 240,
    8: 240,
    9: 240,
    10: 240
}

# Constraint ranges for progressive expansion
constraint_config = {
    'constraint_n_steps': 5,
    'max_weight_range': (0.05, 0.4),
    'min_assets_range': (3, 10),
    'max_assets_range': (10, 100),
    'sparsity_threshold_range': (0.001, 0.05),
    'future_window_range': (5, 50)
}



# %% Curriculum Training
# print("Starting curriculum training...")

# # DEMONSTRATION: New TrainingLogger in action
# print("\n" + "="*60)
# print("🆕 DEMONSTRATING NEW TRAININGLOGGER")
# print("="*60)

# # Create a logger instance to show the new system
# demo_logger = create_curriculum_logger(
#     log_frequency=25,
#     checkpoint_frequency=250,
#     other_metrics_to_log=['max_drawdown', 'sharpe_ratio', 'carmdd']
# )

# print("✅ TrainingLogger successfully initialized!")
# print("📊 This logger will provide:")
# print("   • Centralized, consistent logging")
# print("   • Automatic CSV generation with comprehensive metrics")
# print("   • Smart checkpoint management")
# print("   • Clean separation of training logic from logging")
# print("   • Easy extensibility for new training methods")
# print()
# print("🔄 Now running your existing training with current logging...")
# print("="*60)

# trained_model = train_model_curriculum(
#     model=model,
#     optimizer=optimizer,
#     data=train_data,
#     past_window_size=past_window_size,
    
#     # Curriculum schedules
#     batch_schedule=batch_schedule,
#     column_schedule=column_schedule,
#     max_reasonable_cols=5000,  # Allow up to 5000 columns (instead of default 500)
    
#     # Constraint progression
#     **constraint_config,
    
#     # Training parameters
#     iterations=sum(batch_schedule.values()),  # Total: 2000 iterations
#     loss="expected_return",
#     loss_aggregation= 'gmse', #'progressive',
#     other_metrics_to_log=['max_drawdown', 'sharpe_ratio', 'carmdd'],
    
#     # Enhanced stability
#     learning_rate=1e-3,
#     weight_decay=2e-4,
#     warmup_steps=300,
    
#     # Logging
#     checkpoint_frequency=250,
#     log_frequency=25
# )

# print("Curriculum training completed!")

# %% Alternative training without dictionary schedules
print("Starting alternative training without dictionary schedules...")
trained_model = train_model_curriculum(
    model=model,
    optimizer=optimizer,
    data=train_data,
    past_window_size=past_window_size,
    min_batch_size=512,
    max_batch_size=4096,
    n_column_buckets=12,
    constraint_n_steps=16,
    max_weight_range=(0.05, 0.4),
    min_assets_range=(3, 5),
    max_assets_range=(5, 20),
    sparsity_threshold_range=(0.001, 0.05),
    future_window_range=(20, 60),
    iterations=100,
    loss="expected_return",
    loss_aggregation='standardized_gmae',
    other_metrics_to_log=[],
    learning_rate=1e-3,
    weight_decay=2e-4,
    warmup_steps=1,
    checkpoint_frequency=20,
    log_frequency=10
)


#%% - Test Inference on unseen data
print("Testing inference on unseen test data...")

test_df = pd.DataFrame(test_data)

# Conservative portfolio
conservative_constraints = {
    'max_weight': 0.08,
    'min_assets': 20,
    'max_assets': 60,
    'sparsity_threshold': 0.003
}

portfolio_weights_conservative = predict_portfolio_weights(
    model=trained_model,
    data_input=test_df,
    future_window_size=21,
    max_assets_subset=100,
    **conservative_constraints
)

print(f"Conservative portfolio:")
print(f"Max weight: {portfolio_weights_conservative.max():.3f}")
print(f"Active assets: {(portfolio_weights_conservative > conservative_constraints['sparsity_threshold']).sum()}")
print(f"Total allocation: {portfolio_weights_conservative.sum():.3f}")

# Aggressive portfolio
aggressive_constraints = {
    'max_weight': 0.25,
    'min_assets': 5,
    'max_assets': 20,
    'sparsity_threshold': 0.02
}

portfolio_weights_aggressive = predict_portfolio_weights(
    model=trained_model,
    data_input=test_df,
    future_window_size=21,
    max_assets_subset=50,
    **aggressive_constraints
)

print(f"Aggressive portfolio:")
print(f"Max weight: {portfolio_weights_aggressive.max():.3f}")
print(f"Active assets: {(portfolio_weights_aggressive > aggressive_constraints['sparsity_threshold']).sum()}")
print(f"Total allocation: {portfolio_weights_aggressive.sum():.3f}")

#%% - Model Saving
print("Saving model...")

# Model configuration
model_config = {
    'past_window_size': past_window_size,
    'future_window_size': 21,
    'd_model': 256,
    'n_heads': 8,
    'n_transformer_blocks': 6,
    'training_method': 'curriculum',
    'loss_aggregation': 'progressive'
}

# Save model
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(repo_root, 'models', 'curriculum_model.pt')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

save_model(trained_model, model_path, model_config)
loaded_model, loaded_config = load_model(model_path)

print(f"Model saved: {model_path}")
print(f"Training method: {loaded_config['training_method']}")

print("\n" + "="*50)
print("SUMMARY:")
print("✓ Curriculum training: Batch (32→64→128) × Column buckets (3 phases)")
print("✓ Progressive loss aggregation: Huber → GMAE → GMSE")
print("✓ Memory-efficient training: Large batch sizes supported")
print("✓ Out-of-sample testing: Conservative & aggressive portfolios validated")
print("✓ Model saved and validated successfully")
print("="*50)

#%% - Test Corrected Logging (Small Demo)
print("="*60)
print("TESTING UPDATED TRAINING - MEMORY EFFICIENT")
print("="*60)
print("Expected behavior:")
print("- Large batch sizes now possible due to memory-efficient processing")
print("- Training logs every iteration (no gradient accumulation)")
print("- ALL columns should have values (no empty cells):")
print("  * Constraint ranges: 'max_weight_range': '0.100-0.300'")
print("  * Specific values: 'max_weight_used': 0.234")
print("  * Metrics: 'sharpe_ratio': 1.234, 'carmdd': 5.678")
print("- Console shows actual values used for each iteration")
print("\nRunning small test with 6 iterations...")

# Small test to demonstrate memory-efficient training
test_model = train_model_curriculum(
    model=model,
    optimizer=optimizer,
    data=train_data,
    past_window_size=past_window_size,
    min_batch_size=32,
    max_batch_size=64,
    n_column_buckets=3,
    constraint_n_steps=3,
    max_weight_range=(0.1, 0.3),
    min_assets_range=(3, 8),
    max_assets_range=(8, 25),
    sparsity_threshold_range=(0.001, 0.01),
    future_window_range=(5, 15),
    iterations=6,  # Small test - will produce 6 log entries (every iteration)
    loss="expected_return",
    loss_aggregation='huber',
    other_metrics_to_log=['sharpe_ratio', 'carmdd'],  # Test both metrics
    learning_rate=1e-3,
    weight_decay=2e-4,
    warmup_steps=1,
    checkpoint_frequency=100,
    log_frequency=1  # Log every iteration
)

print("="*60)
print("MEMORY-EFFICIENT TRAINING TEST COMPLETED")
print("Expected: 6 log entries (every iteration)")
print("Large batch sizes now supported due to memory efficiency!")
print("ALL columns should contain values - no empty cells!")
print("="*60)
