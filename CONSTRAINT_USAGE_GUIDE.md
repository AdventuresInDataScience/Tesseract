# Constraint-Aware Portfolio Model Usage Guide

## Overview

The improved model architecture allows users to specify portfolio constraints both during training and inference. The model learns to incorporate these constraints as part of its decision-making process. **No NULL/None values are used** - instead, sensible defaults represent "unconstrained" states.

## Key Changes

1. **Expanded Scalar Input**: The model now accepts a 5-dimensional constraint vector instead of a single scalar
2. **Constraint Embedding**: Portfolio constraints are embedded into the model's latent space
3. **User-Controllable Inference**: Users can specify different constraints for each prediction
4. **No NULL Values**: All constraints use sensible defaults for unconstrained behavior

## Constraint Vector Format & Defaults

The scalar input contains 5 normalized values with the following defaults for "unconstrained" behavior:
- `[0]`: Future window size (normalized: actual_value / 100)
- `[1]`: Maximum weight per asset (default: **1.0** = unconstrained, allows 100%)
- `[2]`: Minimum number of assets (normalized: actual_value / 100, default: **0** = unconstrained)
- `[3]`: Maximum number of assets (normalized: actual_value / 100, default: **1000** = unconstrained)
- `[4]`: Sparsity threshold (scaled: actual_value * 100, default: **0.01** = 1%)

## Default Values Explained

- **max_weight=1.0**: Allows any asset to have up to 100% allocation (unconstrained)
- **min_assets=0**: No minimum requirement for number of holdings
- **max_assets=1000**: Effectively unlimited (will be capped by actual number of available assets)
- **sparsity_threshold=0.01**: Still applies 1% minimum threshold for computational stability

## Training Example

```python
# Train with constraints
trained_model = train_model(
    model=model,
    optimizer=optimizer,
    data=data,
    past_window_size=20,
    future_window_size=15,
    max_weight=0.3,          # Max 30% per asset
    min_assets=5,            # Hold at least 5 assets
    max_assets=20,           # Hold at most 20 assets
    sparsity_threshold=0.02, # Ignore weights below 2%
    iterations=1000
)

# Or train unconstrained (using defaults)
unconstrained_model = train_model(
    model=model,
    optimizer=optimizer,
    data=data,
    past_window_size=20,
    future_window_size=15,
    # max_weight=1.0,        # Default: unconstrained
    # min_assets=0,          # Default: unconstrained
    # max_assets=1000,       # Default: unconstrained
    # sparsity_threshold=0.01, # Default: 1%
    iterations=1000
)
```

## Inference Examples

### Conservative Portfolio
```python
from functions.model import predict_portfolio_weights

# Prepare your price data (normalized)
price_data = torch.tensor(your_data, dtype=torch.float32)

# Conservative approach
conservative_weights = predict_portfolio_weights(
    model=trained_model,
    matrix_input=price_data,
    future_window_size=30,     # Longer horizon
    max_weight=0.2,            # Max 20% per asset
    min_assets=10,             # Diversify across 10+ assets
    max_assets=25,             # But not too many
    sparsity_threshold=0.03    # Higher threshold for stability
)
```

### Aggressive Portfolio
```python
# Aggressive approach
aggressive_weights = predict_portfolio_weights(
    model=trained_model,
    matrix_input=price_data,
    future_window_size=5,      # Shorter horizon
    max_weight=0.8,            # Allow concentration
    min_assets=2,              # Few assets OK
    max_assets=8,              # Focus on best opportunities
    sparsity_threshold=0.05    # Higher threshold, fewer positions
)
```

### Unconstrained Portfolio
```python
# Unconstrained approach - just specify what you want to change
unconstrained_weights = predict_portfolio_weights(
    model=trained_model,
    matrix_input=price_data,
    future_window_size=20     # Only specify non-default values
    # All other constraints use defaults (unconstrained)
)
```

## Benefits

1. **Flexibility**: Same model can generate different portfolio styles based on user preferences
2. **Consistency**: Model learns to respect constraints during training, leading to better constraint satisfaction
3. **Personalization**: Users can easily adjust risk preferences without retraining
4. **Real-world Applicability**: Constraints reflect actual portfolio management requirements

## Migration from Old Version

If you have an existing model trained with the old architecture:
1. Retrain with the new constraint-aware architecture
2. Update your inference code to use `predict_portfolio_weights()`
3. Specify constraints explicitly instead of relying on hardcoded values

## Advanced Usage

For batch predictions with different constraints per sample:
```python
# Create different constraint vectors for each sample
batch_constraints = torch.stack([
    torch.tensor([20/100, 0.3, 5/100, 15/100, 0.02*100]),  # Conservative
    torch.tensor([10/100, 0.6, 2/100, 8/100, 0.05*100]),   # Aggressive
    torch.tensor([15/100, 0.4, 6/100, 12/100, 0.03*100])   # Balanced
])

# Direct model call with custom constraints
weights = model(price_data, batch_constraints)
```
