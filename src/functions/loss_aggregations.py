"""
Loss aggregation functions for the Tesseract portfolio optimization system.
Contains methods for aggregating individual losses across batches.
"""

import torch
import math
# --- Winsorization for outlier handling ---
def winsorize_losses(tensor, lower_percentile=1, upper_percentile=98):
    """
    Apply winsorization to a tensor by clipping its values at specified percentiles.
    
    Args:
        tensor: Input tensor to be winsorized.
        lower_percentile: Lower percentile for clipping (default 1).
        upper_percentile: Upper percentile for clipping (default 98).
    
    Returns:
        Winsorized tensor.
    """
    lower_bound = torch.percentile(tensor, lower_percentile)
    upper_bound = torch.percentile(tensor, upper_percentile)
    return torch.clamp(tensor, min=lower_bound, max=upper_bound)

# --- Loss Aggregation Functions ---
def mean_square_error_aggregation(losses):
    """
    Calculate Mean Square Error (MSE) aggregation.
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
    
    Returns:
        Mean of squared losses
    """
    return torch.mean(losses ** 2)

def geometric_mean_absolute_error_aggregation(losses, epsilon=1e-8):
    """
    Aggregate losses using geometric mean absolute error with numerical stability.
    
    This uses log-space operations for numerical stability when computing geometric means.
    For financial losses that are negative (good), preserve the sign in the result.
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
        epsilon: Small constant for numerical stability
    
    Returns:
        Geometric mean of absolute losses using log-space operations,
        with sign preserved for negative losses
    
    Mathematical approach:
    geometric_mean = exp(mean(log(abs(losses) + epsilon)))
    But implemented using logsumexp for numerical stability.
    """
    # Check if all losses are negative (financial case - preserve sign)
    all_negative = torch.all(losses < 0)
    
    # Take absolute value and add epsilon for stability
    abs_losses = torch.abs(losses) + epsilon
    
    # Compute log of absolute losses
    log_losses = torch.log(abs_losses)
    
    # Compute mean in log space
    mean_log_loss = torch.mean(log_losses)
    
    # Convert back from log space
    geometric_mean = torch.exp(mean_log_loss)
    
    # If all losses were negative (financial case), preserve the negative sign
    if all_negative:
        geometric_mean = -geometric_mean
    
    return geometric_mean

def geometric_mean_square_error_aggregation(losses, epsilon=1e-8):
    """
    Aggregate losses using geometric mean square error with numerical stability.
    For financial losses that are negative (good), preserve the sign in the result.
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
        epsilon: Small constant for numerical stability
    
    Returns:
        Geometric mean of squared losses using log-space operations,
        with sign preserved for negative losses
    """
    # Check if all losses are negative (financial case - preserve sign)
    all_negative = torch.all(losses < 0)
    
    # Square the losses and add epsilon for stability
    squared_losses = losses ** 2 + epsilon
    
    # Compute log of squared losses
    log_squared_losses = torch.log(squared_losses)
    
    # Compute mean in log space
    mean_log_squared_loss = torch.mean(log_squared_losses)
    
    # Convert back from log space
    geometric_mean_squared = torch.exp(mean_log_squared_loss)
    
    # If all losses were negative (financial case), preserve the negative sign
    if all_negative:
        geometric_mean_squared = -geometric_mean_squared
    
    return geometric_mean_squared

def huber_loss_aggregation(losses, delta=1.0):
    """
    Aggregate losses using Huber loss with specified delta parameter.
    
    Huber loss is quadratic for small errors and linear for large errors,
    making it robust to outliers while maintaining smooth gradients.
    For financial losses that are negative (good), preserve the sign in the result.
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
        delta: Threshold for switching between quadratic and linear regions
    
    Returns:
        Mean Huber loss with sign preserved for negative losses
    """
    # Check if all losses are negative (financial case - preserve sign)
    all_negative = torch.all(losses < 0)
    
    abs_losses = torch.abs(losses)
    
    # For |loss| <= delta: 0.5 * loss^2
    # For |loss| > delta: delta * (|loss| - 0.5 * delta)
    huber_losses = torch.where(
        abs_losses <= delta,
        0.5 * losses ** 2,  # This keeps the squared term positive
        delta * (abs_losses - 0.5 * delta)  # This keeps the linear term positive
    )
    
    mean_huber = torch.mean(huber_losses)
    
    # If all losses were negative (financial case), preserve the negative sign
    if all_negative:
        mean_huber = -mean_huber
    
    return mean_huber

def standardized_geometric_mean_error_aggregation(losses):
    """
    Aggregate losses using standardized geometric mean error.
    
    This function computes the geometric mean absolute error and normalizes it
    by the standard deviation of the losses to reduce sensitivity to variance.
    The standard deviation is clamped to prevent division by very small values
    or excessive normalization from very large values.
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
    
    Returns:
        Standardized geometric mean error (geometric mean / clamped std dev)
    """
    geom_return = geometric_mean_absolute_error_aggregation(losses)
    std_dev = torch.std(losses)
    std_dev = torch.clamp(std_dev, min=0.003, max=2.0)  # clamp from 0.003 to 2
    return geom_return / std_dev

def sortino_geometric_mean_error_aggregation(losses, target=1.0):
    """
    Aggregate losses using Sortino-style geometric mean error.
    
    This function computes the geometric mean absolute error and normalizes it
    by the downside standard deviation (like Sortino ratio), which only considers
    losses below the target threshold (worse performance). For fractional losses
    around 1, downside losses are those < target (default 1.0).
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses (fractional around 1)
        target: Target threshold for downside calculation (default 1.0)
    
    Returns:
        Sortino-style geometric mean error (geometric mean / clamped downside std dev)
    """
    geom_return = geometric_mean_absolute_error_aggregation(losses)
    
    # Calculate downside deviation (only losses below target - worse performance)
    downside_losses = target - losses
    downside_losses = torch.where(downside_losses > 0, downside_losses, torch.zeros_like(downside_losses))
    
    # Calculate downside standard deviation
    downside_std = torch.std(downside_losses)
    
    # Clamp to prevent division by very small values or excessive normalization
    downside_std = torch.clamp(downside_std, min=0.003, max=2.0)
    
    return geom_return / downside_std

# --- Aggregation Function Selector ---
def get_loss_aggregation_function(aggregation_method: str):
    """
    Get loss aggregation function by name.
    
    Args:
        aggregation_method: String name of the aggregation method
        
    Returns:
        Aggregation function or None for methods handled by PyTorch directly
        
    Available methods:
        - 'mse' or 'mean_square_error': Mean Square Error
        - 'gmae', 'geometric_mean_absolute_error', or 'geometric': Geometric Mean Absolute Error
        - 'gmse', 'geometric_mean_square_error', or 'geometric_mse': Geometric Mean Square Error  
        - 'huber' or 'huber_loss': Huber Loss (robust to outliers)
        - 'standardized_gmae' or 'standardized_geometric_mean_error': Standardized Geometric Mean Absolute Error
        - 'sortino_gmae' or 'sortino_geometric_mean_error': Sortino-style Geometric Mean Absolute Error (downside deviation)
        - 'mae' or 'arithmetic': Returns None (handled as stacked_losses.mean())
    """
    aggregation_map = {
        'mse': mean_square_error_aggregation,
        'gmae': geometric_mean_absolute_error_aggregation,
        'gmse': geometric_mean_square_error_aggregation,
        'huber': huber_loss_aggregation,
        'standardized_gmae': standardized_geometric_mean_error_aggregation,
        'sortino_gmae': sortino_geometric_mean_error_aggregation,
        # Full descriptive names
        'mean_square_error': mean_square_error_aggregation,
        'geometric_mean_absolute_error': geometric_mean_absolute_error_aggregation,
        'geometric_mean_square_error': geometric_mean_square_error_aggregation,
        'huber_loss': huber_loss_aggregation,
        'standardized_geometric_mean_error': standardized_geometric_mean_error_aggregation,
        'sortino_geometric_mean_error': sortino_geometric_mean_error_aggregation,
        # Aliases
        'geometric': geometric_mean_absolute_error_aggregation,
        'geometric_mse': geometric_mean_square_error_aggregation,
        # Methods handled by PyTorch directly (return None)
        'mae': None,
        'arithmetic': None
    }
    
    if aggregation_method not in aggregation_map:
        available = ', '.join(sorted(aggregation_map.keys()))
        raise ValueError(f"Unknown aggregation method '{aggregation_method}'. Available: {available}")
    
    return aggregation_map[aggregation_method]
