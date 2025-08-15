"""
Loss aggregation functions for the Tesseract portfolio optimization system.
Contains methods for aggregating individual losses across batches.
"""

import torch
import math


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
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
        epsilon: Small constant for numerical stability
    
    Returns:
        Geometric mean of absolute losses using log-space operations
    
    Mathematical approach:
    geometric_mean = exp(mean(log(abs(losses) + epsilon)))
    But implemented using logsumexp for numerical stability.
    """
    # Take absolute value and add epsilon for stability
    abs_losses = torch.abs(losses) + epsilon
    
    # Compute log of absolute losses
    log_losses = torch.log(abs_losses)
    
    # Compute mean in log space
    mean_log_loss = torch.mean(log_losses)
    
    # Convert back from log space
    return torch.exp(mean_log_loss)


def geometric_mean_square_error_aggregation(losses, epsilon=1e-8):
    """
    Aggregate losses using geometric mean square error with numerical stability.
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
        epsilon: Small constant for numerical stability
    
    Returns:
        Geometric mean of squared losses using log-space operations
    """
    # Square the losses and add epsilon for stability
    squared_losses = losses ** 2 + epsilon
    
    # Compute log of squared losses
    log_squared_losses = torch.log(squared_losses)
    
    # Compute mean in log space
    mean_log_squared_loss = torch.mean(log_squared_losses)
    
    # Convert back from log space
    return torch.exp(mean_log_squared_loss)


def huber_loss_aggregation(losses, delta=1.0):
    """
    Aggregate losses using Huber loss with specified delta parameter.
    
    Huber loss is quadratic for small errors and linear for large errors,
    making it robust to outliers while maintaining smooth gradients.
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
        delta: Threshold for switching between quadratic and linear regions
    
    Returns:
        Mean Huber loss
    """
    abs_losses = torch.abs(losses)
    
    # For |loss| <= delta: 0.5 * loss^2
    # For |loss| > delta: delta * (|loss| - 0.5 * delta)
    huber_losses = torch.where(
        abs_losses <= delta,
        0.5 * losses ** 2,
        delta * (abs_losses - 0.5 * delta)
    )
    
    return torch.mean(huber_losses)


def get_loss_aggregation_function(aggregation_method: str):
    """
    Get loss aggregation function by name.
    
    Args:
        aggregation_method: String name of the aggregation method
        
    Returns:
        Aggregation function
        
    Available methods:
        - 'mse': Mean Square Error
        - 'gmae': Geometric Mean Absolute Error
        - 'gmse': Geometric Mean Square Error  
        - 'huber': Huber Loss (robust to outliers)
    """
    aggregation_map = {
        'mse': mean_square_error_aggregation,
        'gmae': geometric_mean_absolute_error_aggregation,
        'gmse': geometric_mean_square_error_aggregation,
        'huber': huber_loss_aggregation
    }
    
    if aggregation_method not in aggregation_map:
        available = ', '.join(sorted(aggregation_map.keys()))
        raise ValueError(f"Unknown aggregation method '{aggregation_method}'. Available: {available}")
    
    return aggregation_map[aggregation_method]
