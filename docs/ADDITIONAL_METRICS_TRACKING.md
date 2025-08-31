# Additional Metrics Tracking for Model Training

## Overview

The model training system now supports tracking multiple loss metrics with different aggregation methods, separate from the optimization target. This allows you to:

- **Optimize** using one loss function + aggregation method (e.g., `sharpe_ratio` + `huber` for stability)
- **Track** multiple metrics with different aggregations for analysis (e.g., raw vs winsorized, different aggregation methods)
- **Compare** different loss aggregation strategies side-by-side
- **Analyze** the loss landscape without affecting training

## Key Features

### 1. Separation of Optimization and Logging
- **Optimization**: Uses `loss` + `loss_aggregation` parameters (with winsorization for stability)
- **Tracking**: Uses `additional_metrics_config` to track any metric/aggregation combinations
- **Raw Data Access**: Track both raw and winsorized losses for complete analysis

### 2. Flexible Metric Configurations
Each tracked metric can specify:
- `metric`: Which loss function to calculate (e.g., 'sharpe_ratio', 'max_drawdown')
- `aggregation`: How to aggregate across batch (e.g., 'arithmetic', 'huber', 'gmae', 'gmse')
- `winsorize`: Whether to apply winsorization (True/False)

### 3. Rich Logging Output
The training logs now include:
- Primary optimization loss (winsorized + aggregated)
- Raw loss statistics (mean, std, min, max)
- Winsorized loss statistics (mean, std, min, max)
- All additional metrics as specified in configuration

## Usage Example

```python
from src.functions.model_train import train_model_progressive, create_additional_metrics_config

# Method 1: Using helper function with tuples
additional_metrics = create_additional_metrics_config([
    # (name, metric, aggregation, winsorize)
    ('sharpe_huber_raw', 'sharpe_ratio', 'huber', False),
    ('sharpe_huber_winsorized', 'sharpe_ratio', 'huber', True),
    ('sharpe_arithmetic_raw', 'sharpe_ratio', 'arithmetic', False),
    ('max_drawdown_raw', 'max_drawdown', 'arithmetic', False),
    ('sortino_gmae_winsorized', 'sortino_ratio', 'gmae', True)
])

# Method 2: Manual dictionary creation
additional_metrics = {
    'sharpe_huber_raw': {'metric': 'sharpe_ratio', 'aggregation': 'huber', 'winsorize': False},
    'sharpe_gmae_winsorized': {'metric': 'sharpe_ratio', 'aggregation': 'gmae', 'winsorize': True},
    'max_drawdown_raw': {'metric': 'max_drawdown', 'aggregation': 'arithmetic', 'winsorize': False}
}

# Train with additional metrics tracking
trained_model = train_model_progressive(
    model=model,
    optimizer=optimizer,
    data=data,
    past_window_size=20,
    future_window_size=5,
    
    # Primary optimization (what the model actually optimizes)
    loss='sharpe_ratio',
    loss_aggregation='progressive',  # huber → gmae → gmse progression
    
    # Additional tracking (logged but doesn't affect optimization)
    additional_metrics_config=additional_metrics,
    
    # Other parameters...
    iterations=1000,
    log_frequency=10
)
```

## Use Cases

### 1. Stability Analysis
Compare robust (huber) vs sensitive (gmse) aggregations:
```python
stability_config = create_additional_metrics_config([
    ('sharpe_huber', 'sharpe_ratio', 'huber', True),
    ('sharpe_gmse', 'sharpe_ratio', 'gmse', True),
    ('sharpe_arithmetic', 'sharpe_ratio', 'arithmetic', True)
])
```

### 2. Winsorization Impact Analysis
See the effect of winsorization on different metrics:
```python
winsorization_config = create_additional_metrics_config([
    ('sharpe_raw', 'sharpe_ratio', 'gmae', False),
    ('sharpe_winsorized', 'sharpe_ratio', 'gmae', True),
    ('drawdown_raw', 'max_drawdown', 'arithmetic', False),
    ('drawdown_winsorized', 'max_drawdown', 'arithmetic', True)
])
```

### 3. Multi-Metric Analysis
Track different financial metrics simultaneously:
```python
multi_metric_config = create_additional_metrics_config([
    ('sharpe_optimized', 'sharpe_ratio', 'huber', True),
    ('sortino_alternative', 'sortino_ratio', 'huber', True),
    ('max_drawdown', 'max_drawdown', 'arithmetic', False),
    ('expected_return', 'expected_return', 'arithmetic', False)
])
```

## Available Aggregation Methods

- `'arithmetic'` or `'mae'`: Simple arithmetic mean
- `'mse'`: Mean squared error
- `'huber'`: Huber loss (robust to outliers, good for stability)
- `'gmae'` or `'geometric'`: Geometric mean absolute error
- `'gmse'` or `'geometric_mse'`: Geometric mean squared error
- `'progressive'`: Automatic progression (huber → gmae → gmse)

## Available Metrics

All standard financial metrics are supported:
- `'sharpe_ratio'`, `'geometric_sharpe_ratio'`
- `'sortino_ratio'`, `'geometric_sortino_ratio'`
- `'max_drawdown'`, `'expected_return'`
- `'carmdd'`, `'omega_ratio'`, `'jensen_alpha'`
- `'treynor_ratio'`, `'ulcer_index'`, `'k_ratio'`

## Log File Output

The training log CSV will include columns for:
- Standard training info (iteration, loss, learning_rate, etc.)
- `raw_loss_mean`, `raw_loss_std`, `raw_loss_min`, `raw_loss_max`
- `winsorized_loss_mean`, `winsorized_loss_std`, `winsorized_loss_min`, `winsorized_loss_max`
- Each metric name from your `additional_metrics_config`

This gives you complete visibility into the training dynamics while maintaining optimization stability.
