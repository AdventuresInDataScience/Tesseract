"""
Debug test to check loss sign issue in curriculum vs progressive training
"""
import sys
import os
import torch
import numpy as np

sys.path.append('functions')
from functions.loss_metrics import expected_return, create_portfolio_time_series
from functions.loss_aggregations import get_loss_aggregation_function

# Create a simple test portfolio timeseries that should give negative expected return
torch.manual_seed(42)
np.random.seed(42)

# Portfolio that goes from 1.0 to 1.1 (10% gain) - should give -1.1 as loss
portfolio_timeseries = torch.tensor([1.0, 1.02, 1.05, 1.08, 1.1], dtype=torch.float32)

print("=== TESTING LOSS CALCULATION ===")
print(f"Portfolio timeseries: {portfolio_timeseries}")

# Test individual expected return
single_loss = expected_return(portfolio_timeseries)
print(f"Single expected return loss: {single_loss:.6f}")

# Test batch of losses
batch_losses = torch.tensor([
    expected_return(portfolio_timeseries),
    expected_return(portfolio_timeseries * 0.95),  # 5% loss
    expected_return(portfolio_timeseries * 1.05)   # 15% gain
])
print(f"Batch losses: {batch_losses}")

# Test different aggregation methods
aggregation_methods = ['huber', 'gmae', 'gmse']

for method in aggregation_methods:
    agg_func = get_loss_aggregation_function(method)
    aggregated_loss = agg_func(batch_losses)
    print(f"{method.upper()} aggregated loss: {aggregated_loss:.6f}")

print("\n=== EXPECTED BEHAVIOR ===")
print("- Single loss should be negative (around -1.1)")
print("- All aggregated losses should be negative")
print("- GMAE should NOT make losses positive")
