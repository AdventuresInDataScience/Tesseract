"""
Model training functions for the Tesseract portfolio optimization system.
Contains training loops, optimization logic, and model update functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import os
import math
from datetime import datetime

# Import functions from other modules
try:
    from .loss_metrics import create_portfolio_time_series, calculate_expected_metric
    from .loss_aggregations import (
        mean_square_error_aggregation, 
        geometric_mean_absolute_error_aggregation,
        geometric_mean_square_error_aggregation,
        huber_loss_aggregation
    )
except ImportError:
    # Fallback for when running as standalone module
    from loss_metrics import create_portfolio_time_series, calculate_expected_metric
    from loss_aggregations import (
        mean_square_error_aggregation, 
        geometric_mean_absolute_error_aggregation,
        geometric_mean_square_error_aggregation,
        huber_loss_aggregation
    )


def create_single_sample(data_array, past_window_size, future_window_size, n_cols, valid_indices, max_retries=10):
    """
    Create a single sample from the numpy array.
    
    Args:
        data_array: numpy array of shape (n_timesteps, n_assets)
        past_window_size: Number of timesteps for past window
        future_window_size: Number of timesteps for future window
        n_cols: Number of columns (assets) to sample
        valid_indices: Maximum starting index to avoid out-of-bounds
        max_retries: Maximum retry attempts if insufficient columns found
        
    Returns:
        numpy array of shape (n_cols, past_window_size + future_window_size)
    """
    # Choose a random starting point
    start_idx = random.randint(0, valid_indices)
    # Slice the array
    data_slice = data_array[start_idx:start_idx + (past_window_size + future_window_size), :]
    
    # Ensure data_slice is numeric (convert to float if needed)
    if data_slice.dtype == 'object' or not np.issubdtype(data_slice.dtype, np.number):
        try:
            data_slice = data_slice.astype(np.float64)
        except (ValueError, TypeError):
            # If conversion fails, treat all values as invalid
            valid_col_indices = np.array([])
        else:
            # Find columns with no NaN values in this slice
            valid_cols = ~np.isnan(data_slice).any(axis=0)
            valid_col_indices = np.where(valid_cols)[0]
    else:
        # Find columns with no NaN values in this slice
        valid_cols = ~np.isnan(data_slice).any(axis=0)
        valid_col_indices = np.where(valid_cols)[0]
    
    # Ensure we always get a sample, even if we need to retry or pad
    retry_count = 0
    
    while len(valid_col_indices) < n_cols and retry_count < max_retries:
        # Try a different random starting point
        start_idx = random.randint(0, valid_indices)
        data_slice = data_array[start_idx:start_idx + (past_window_size + future_window_size), :]
        valid_cols = ~np.isnan(data_slice).any(axis=0)
        valid_col_indices = np.where(valid_cols)[0]
        retry_count += 1
    
    # If we still don't have enough columns after retries, pad with zeros
    if len(valid_col_indices) >= n_cols:
        # Randomly sample n_cols from valid columns
        selected_cols = np.random.choice(valid_col_indices, size=n_cols, replace=False)
        col_sample = data_slice[:, selected_cols]
        sample_array = col_sample.T  # Transpose to (n_cols, past_window_size+future_window_size)
    else:
        # Create zero-padded sample to maintain batch size
        sample_array = np.zeros((n_cols, past_window_size + future_window_size))
        if len(valid_col_indices) > 0:
            # Fill with available data
            available_cols = min(len(valid_col_indices), n_cols)
            selected_cols = np.random.choice(valid_col_indices, size=available_cols, replace=False)
            sample_array[:available_cols, :] = data_slice[:, selected_cols].T
    
    # Apply scaling: divide each row by the value at the end of past_window_size
    # This ensures the last timestep of past_window becomes 1.0 for each asset
    scaling_values = sample_array[:, past_window_size - 1]  # Values at end of past window
    
    # Avoid division by zero - only scale rows where scaling value is non-zero
    # This should never happen. May need to change this logic in future
    non_zero_mask = scaling_values != 0
    if np.any(non_zero_mask):
        sample_array[non_zero_mask, :] = sample_array[non_zero_mask, :] / scaling_values[non_zero_mask, np.newaxis]
    
    return sample_array


def create_batch(data_array, past_window_size, future_window_size, n_cols, batch_size, valid_indices):
    """
    Create a single batch and return past and future tensors.
    
    Args:
        data_array: numpy array of shape (n_timesteps, n_assets)
        past_window_size: Number of timesteps for past window
        future_window_size: Number of timesteps for future window
        n_cols: Number of columns (assets) to sample
        batch_size: Number of samples in the batch
        valid_indices: Maximum starting index to avoid out-of-bounds
    
    Returns:
        past_batch: torch.Tensor of shape (batch_size, n_cols, past_window_size)
        future_batch: torch.Tensor of shape (batch_size, n_cols, future_window_size)
    """
    batch = []
    
    for j in range(batch_size):
        sample = create_single_sample(data_array, past_window_size, future_window_size, n_cols, valid_indices)
        batch.append(sample)
    
    # Convert list of numpy arrays to tensor - automatically gets correct shape!
    batch_array = np.array(batch)  # Shape: (batch_size, n_cols, past_window_size + future_window_size)
    
    # Split into past and future batches
    past_batch = torch.tensor(batch_array[:, :, :past_window_size], dtype=torch.float32)      # First past_window_size timesteps
    future_batch = torch.tensor(batch_array[:, :, past_window_size:past_window_size + future_window_size], dtype=torch.float32)   # Next future_window_size timesteps
    
    return past_batch, future_batch


def update_model(model, optimizer, past_batch, future_batch, loss, 
                max_weight=1.0, min_assets=0, max_assets=1000, sparsity_threshold=0.01,
                regularization_lambda=0.0, loss_aggregation='arithmetic', 
                gradient_accumulation_steps=1, accumulation_step=0, *args, **kwargs):
    """
    Perform forward and backward pass to update the model with gradient accumulation support.
    
    Args:
        model: The GPT2LikeTransformer model to update
        optimizer: PyTorch optimizer (e.g., Adam, SGD)
        past_batch: Dictionary containing past data for model input:
            - 'matrix_input': Tensor of shape (batch_size, n, past_window_size)
            - 'scalar_input': Tensor of shape (batch_size, 1) containing normalized future_window_size
            - 'constraint_input': Tensor of shape (batch_size, 4) containing normalized constraints
            - 'raw_constraints': Dictionary with raw constraint values for enforcement
        future_batch: Dictionary containing future data for loss calculation:
            - 'returns': Future returns matrix for portfolio evaluation
        loss: String name of the loss function to optimize (e.g., 'sharpe_ratio')
        max_weight: Maximum weight constraint (used with raw constraint values)
        min_assets: Minimum number of assets constraint (used with raw constraint values)
        max_assets: Maximum number of assets constraint (used with raw constraint values)
        sparsity_threshold: Threshold for setting small weights to zero (used with raw constraint values)
        regularization_lambda: Weight for portfolio regularization loss
        loss_aggregation: Method to aggregate losses across batch:
            - 'mae' or 'arithmetic': Mean Absolute Error - arithmetic mean
            - 'mse': Mean Square Error - mean of squared losses
            - 'huber': Huber Loss - robust to outliers (good for Phase 1 stability)
            - 'gmae' or 'geometric': Geometric Mean Absolute Error - log-space geometric mean
            - 'gmse' or 'geometric_mse': Geometric Mean Square Error
        gradient_accumulation_steps: Number of steps to accumulate gradients
        accumulation_step: Current step in the accumulation cycle
        *args, **kwargs: Additional arguments passed to metric calculation
    
    Returns:
        Dictionary containing:
            - 'loss': The total loss value
            - 'metric_loss': The primary metric loss
            - 'reg_loss': The regularization loss (if applied)
            - 'weights': The predicted portfolio weights
    """
    # Zero gradients only at the start of accumulation cycle (handled by calling function)
    # The calling function (train_model) handles optimizer.zero_grad() timing
    
    # Extract inputs from past_batch
    matrix_input = past_batch['matrix_input']
    scalar_input = past_batch['scalar_input'] 
    constraint_input = past_batch['constraint_input']
    
    # Forward pass through model (with normalized inputs for neural network)
    weights = model(matrix_input, scalar_input, constraint_input)
    
    # Extract future returns for portfolio evaluation
    future_returns = future_batch['returns']  # Shape: (timesteps, n_assets)
    
    # Calculate portfolio time series for each sample in the batch
    batch_size = weights.shape[0]
    metric_losses = []
    
    for i in range(batch_size):
        # Create portfolio time series from weights and future returns
        portfolio_timeseries = create_portfolio_time_series(future_returns, weights[i])
        
        # Calculate the loss function (e.g., Sharpe ratio, etc.)
        metric_loss = calculate_expected_metric(portfolio_timeseries, None, loss, *args, **kwargs)
        metric_losses.append(metric_loss)
    
    # Stack all losses
    stacked_losses = torch.stack(metric_losses)
    
    # Apply the selected aggregation method (passed as parameter)
    if loss_aggregation == 'mae' or loss_aggregation == 'arithmetic':
        # Mean Absolute Error: Standard arithmetic mean (backward compatibility)
        metric_loss = stacked_losses.mean()
    elif loss_aggregation == 'mse':
        # Mean Square Error: Mean of squared losses
        metric_loss = mean_square_error_aggregation(stacked_losses)
    elif loss_aggregation == 'huber':
        # Huber Loss: Robust to outliers, good for Phase 1 stability
        metric_loss = huber_loss_aggregation(stacked_losses)
    elif loss_aggregation == 'gmae' or loss_aggregation == 'geometric':
        # Geometric Mean Absolute Error: Using log-space operations (numerically stable)
        metric_loss = geometric_mean_absolute_error_aggregation(stacked_losses)
    elif loss_aggregation == 'gmse' or loss_aggregation == 'geometric_mse':
        # Geometric Mean Square Error (default, user's proposed method)
        metric_loss = geometric_mean_square_error_aggregation(stacked_losses)
    else:
        # Fallback to arithmetic mean
        metric_loss = stacked_losses.mean()
        print(f"Warning: Unknown loss_aggregation method '{loss_aggregation}', using MAE (arithmetic mean)")
    
    # Calculate regularization loss if requested (currently disabled)
    reg_loss = torch.tensor(0.0, device=weights.device)
    # TODO: Implement portfolio_regularization_loss if needed
    # if regularization_lambda > 0.0:
    #     reg_loss = portfolio_regularization_loss(weights, ...)
    
    # Total loss (currently just metric loss)
    total_loss = metric_loss  # + regularization_lambda * reg_loss
    
    # Scale loss by accumulation steps for gradient accumulation
    scaled_loss = total_loss / gradient_accumulation_steps
    
    # Backward pass (accumulate gradients)
    scaled_loss.backward()
    
    # Apply optimizer step only on the last accumulation step
    if (accumulation_step + 1) % gradient_accumulation_steps == 0:
        # Gradient clipping for stability (very important for transformer training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
    
    # Return loss information and weights (using unscaled loss for logging)
    return {
        'loss': total_loss.item(),
        'metric_loss': metric_loss.item(),
        'reg_loss': reg_loss.item() if regularization_lambda > 0.0 else 0.0,
        'weights': weights.detach()
    }


def train_model(model, optimizer, data, past_window_size, future_window_size, min_n_cols = 10, 
                       max_n_cols = 100, min_batch_size = 32, max_batch_size = 256, iterations = 1000, 
                       loss ='sharpe_ratio', loss_aggregation='progressive',
                       # Additional metrics to log
                       other_metrics_to_log=None,
                       # Constraint ranges for random sampling
                       max_weight_range=(0.1, 1.0), min_assets_range=(0, 50), 
                       max_assets_range=(5, 200), sparsity_threshold_range=(0.005, 0.05),
                       # Logging and checkpoint paths
                       log_path=None, checkpoint_path=None,
                       # Frequency controls
                       checkpoint_frequency=50, log_frequency=10,
                       # Stability controls
                       use_scheduler=True, scheduler_patience=500,
                       # Early stopping  
                       early_stopping_patience=2000, early_stopping_threshold=1e-6,
                       # Enhanced optimizer configuration
                       learning_rate=1e-3, weight_decay=2e-4, warmup_steps=500,
                       # Gradient accumulation for larger effective batch size
                       gradient_accumulation_steps=4):
    """
    Progressive batch creation with curriculum learning, random constraint sampling, and enhanced stability features.
    Starts with small batch_size and n_cols, gradually increases both.
    Includes input normalization, Huber loss for Phase 1, enhanced optimizer configuration, and gradient accumulation.
    
    Args:
        data: pandas DataFrame or numpy array of shape (n_timesteps, n_assets)
        past_window_size: Number of timesteps for past window
        future_window_size: Number of timesteps for future window
        min_n_cols: Starting number of columns (assets)
        max_n_cols: Final number of columns (assets)
        min_batch_size: Starting batch size
        max_batch_size: Final batch size
        iterations: Total number of iterations
        loss: Loss function to optimize ('sharpe_ratio', 'geometric_sharpe_ratio', etc.)
        loss_aggregation: Method to aggregate losses across batch
            - 'huber': Huber Loss - robust to outliers (Phase 1 stability)
            - 'mse': Mean Square Error - mean of squared losses  
            - 'gmae': Geometric Mean Absolute Error - log-space geometric mean (balanced)
            - 'gmse': Geometric Mean Square Error (most sensitive to outliers)
            - 'progressive': Progressive curriculum: huber → gmae → gmse (recommended for stability)
        other_metrics_to_log: Additional metrics to calculate and log alongside the primary metric.
            Can be a string (single metric) or list of strings (multiple metrics).
            Available metrics: 'sharpe_ratio', 'geometric_sharpe_ratio', 'max_drawdown', 
            'sortino_ratio', 'geometric_sortino_ratio', 'expected_return', 'carmdd', 
            'omega_ratio', 'jensen_alpha', 'treynor_ratio', 'ulcer_index', 'k_ratio'
        max_weight_range: (min, max) range for max_weight constraint sampling
        min_assets_range: (min, max) range for min_assets constraint sampling  
        max_assets_range: (min, max) range for max_assets constraint sampling
        sparsity_threshold_range: (min, max) range for sparsity_threshold sampling
        log_path: Path to save training logs (default: repo_root/logs/)
        checkpoint_path: Path to save model checkpoints (default: repo_root/checkpoints/)
        checkpoint_frequency: How often to save model checkpoints (default: every 50 iterations)
        log_frequency: How often to save loss data and print progress (default: every 10 iterations)
        learning_rate: Initial learning rate for enhanced optimizer (default: 1e-3)
        weight_decay: L2 regularization for enhanced optimizer (default: 2e-4)
        warmup_steps: Learning rate warmup steps (default: 500)
        gradient_accumulation_steps: Steps to accumulate gradients for larger effective batch size (default: 4)
        
    Returns:
        Trained model
        
    Example:
        >>> # Train with progressive loss aggregation and enhanced stability
        >>> trained_model = train_model(
        ...     model=model, 
        ...     optimizer=optimizer, 
        ...     data=df, 
        ...     past_window_size=20, 
        ...     future_window_size=10,
        ...     loss='sharpe_ratio',
        ...     other_metrics_to_log=['max_drawdown', 'sortino_ratio'],  # Log additional metrics
        ...     loss_aggregation='progressive',  # Huber → GMAE → GMSE for maximum stability
        ...     learning_rate=1e-3,  # Enhanced optimizer settings
        ...     weight_decay=2e-4,
        ...     gradient_accumulation_steps=4,  # 4x larger effective batch size
        ...     checkpoint_frequency=100,  # Save every 100 iterations
        ...     log_frequency=20  # Log every 20 iterations
        ... )
    """
    # start time
    start_time = datetime.now()
    print("Training started at", start_time)

    # Process other_metrics_to_log parameter
    if other_metrics_to_log is None:
        other_metrics_list = []
    elif isinstance(other_metrics_to_log, str):
        other_metrics_list = [other_metrics_to_log]
    elif isinstance(other_metrics_to_log, list):
        other_metrics_list = other_metrics_to_log
    else:
        raise ValueError("other_metrics_to_log must be None, a string, or a list of strings")
    
    # Validate that all requested metrics are available
    available_metrics = ['sharpe_ratio', 'geometric_sharpe_ratio', 'max_drawdown', 
                        'sortino_ratio', 'geometric_sortino_ratio', 'expected_return', 
                        'carmdd', 'omega_ratio', 'jensen_alpha', 'treynor_ratio', 
                        'ulcer_index', 'k_ratio']
    
    for metric_name in other_metrics_list:
        if metric_name not in available_metrics:
            raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {available_metrics}")
    
    if other_metrics_list:
        print(f"Additional metrics to log: {other_metrics_list}")

    # Set up default paths if not provided
    if log_path is None:
        # Get the repository root (assuming this file is in src/functions/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from src/functions/
        log_path = os.path.join(repo_root, "logs")
    
    if checkpoint_path is None:
        # Get the repository root (assuming this file is in src/functions/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from src/functions/
        checkpoint_path = os.path.join(repo_root, "checkpoints")
    
    # Create directories if they don't exist
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    print(f"Logs will be saved to: {log_path}")
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    print(f"Checkpoint frequency: every {checkpoint_frequency} iterations")
    print(f"Log frequency: every {log_frequency} iterations")

    # Enhanced optimizer configuration with parameter grouping
    print(f"Configuring enhanced optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    
    # Parameter grouping: no weight decay on biases and layer norms
    param_groups = []
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Don't apply weight decay to biases and layer norm parameters
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Replace the existing optimizer with enhanced AdamW
    from torch.optim import AdamW
    enhanced_optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),  # Better betas for transformer training
        eps=1e-7,           # Smaller epsilon for numerical stability
        weight_decay=weight_decay  # This will be overridden by param groups
    )
    
    print(f"Enhanced optimizer: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")
    
    # Learning rate scheduler with warmup and cosine decay
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (iterations - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    lr_scheduler = LambdaLR(enhanced_optimizer, lr_lambda)
    print(f"Learning rate scheduler: {warmup_steps} warmup steps, cosine decay over {iterations} iterations")

    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0

    # Gradient accumulation setup
    print(f"Gradient accumulation: {gradient_accumulation_steps} steps (effective batch size multiplier)")
    accumulated_loss = 0.0
    accumulation_step = 0
    
    # Initialize variables for final reporting
    current_iteration_loss = 0.0
    current_lr = learning_rate
    effective_batch_size = min_batch_size * gradient_accumulation_steps
    current_loss_aggregation = 'progressive'

    # Create log with comprehensive training information
    log_columns = [
        'iteration', 'loss', 'metric_loss', 'reg_loss', 
        'loss_aggregation', 'phase', 'batch_size', 'effective_batch_size', 'n_cols', 'progress', 'learning_rate'
    ]
    # Add columns for additional metrics
    log_columns.extend(other_metrics_list)
    log = pd.DataFrame(columns=log_columns)

    # Set up additional plateau scheduler for fine-tuning (optional, in addition to cosine)
    if use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        plateau_scheduler = ReduceLROnPlateau(enhanced_optimizer, mode='min', factor=0.5, 
                                            patience=scheduler_patience, 
                                            min_lr=1e-7)
        print(f"Additional plateau scheduler enabled with patience={scheduler_patience}")
    else:
        plateau_scheduler = None

    # Convert DataFrame to numpy array if needed
    if hasattr(data, 'values'):  # Check if it's a DataFrame
        data_array = data.values
        print("Converted DataFrame to numpy array for faster processing")
    else:
        data_array = data
    
    valid_indices = len(data_array) - (past_window_size + future_window_size)
    
    print(f"Starting training for {iterations} iterations...")
    
    # Loop through fixed number of iterations with progressive difficulty
    for i in range(iterations):
        # Calculate progressive values (linear interpolation)
        progress = i / (iterations - 1) if iterations > 1 else 0  # 0 to 1
        
        # Progressive loss aggregation curriculum for stability
        prev_aggregation = None
        if i > 0:  # Track previous aggregation method for transition detection
            prev_progress = (i - 1) / (iterations - 1) if iterations > 1 else 0
            if loss_aggregation == 'progressive':
                if prev_progress <= 0.4:
                    prev_aggregation = 'huber'
                elif prev_progress <= 0.7:
                    prev_aggregation = 'gmae'
                else:
                    prev_aggregation = 'gmse'
        
        if loss_aggregation == 'progressive':
            # Phase 1 (0-40%): Huber (stable gradients, robust to outliers)
            # Phase 2 (40-70%): GMAE (balanced, emphasizes consistency)  
            # Phase 3 (70-100%): GMSE (most sensitive, best final performance)
            if progress <= 0.4:
                current_loss_aggregation = 'huber'
                phase = "Stability (Huber)"
            elif progress <= 0.7:
                current_loss_aggregation = 'gmae'
                phase = "Balanced (GMAE)"
            else:
                current_loss_aggregation = 'gmse'
                phase = "Performance (GMSE)"
            
            # Detect and announce phase transitions
            if prev_aggregation and prev_aggregation != current_loss_aggregation:
                print(f"\n🔄 PHASE TRANSITION at iteration {i + 1}: {prev_aggregation.upper()} → {current_loss_aggregation.upper()}")
                print(f"   Expect step change in loss due to different aggregation method")
                print(f"   Progress: {progress*100:.1f}% | Phase: {phase}\n")
        else:
            # Use fixed aggregation method
            current_loss_aggregation = loss_aggregation
            phase = f"Fixed ({loss_aggregation.upper()})"

        # Progressive n_cols: start small, end large
        current_n_cols = int(min_n_cols + progress * (max_n_cols - min_n_cols))
        
        # Progressive batch_size: start small, end large  
        current_batch_size = int(min_batch_size + progress * (max_batch_size - min_batch_size))
        
        # More structured constraint progression (smoother than fully random)
        # Use a mix of curriculum learning (progressive) and random sampling
        curriculum_weight = 0.7  # 70% curriculum, 30% random
        
        # Progressive constraint difficulty
        prog_max_weight = max_weight_range[1] - progress * (max_weight_range[1] - max_weight_range[0])  # Start loose, get stricter
        prog_min_assets = min_assets_range[0] + progress * (min_assets_range[1] - min_assets_range[0])  # Start few, require more
        prog_max_assets = max_assets_range[1] - progress * (max_assets_range[1] - max_assets_range[0])  # Start many, limit more
        prog_sparsity = sparsity_threshold_range[0] + progress * (sparsity_threshold_range[1] - sparsity_threshold_range[0])  # Start low, increase
        
        # Random sampling within ranges
        rand_max_weight = np.random.uniform(max_weight_range[0], max_weight_range[1])
        rand_min_assets = np.random.randint(min_assets_range[0], min_assets_range[1] + 1)
        rand_max_assets = np.random.randint(max_assets_range[0], max_assets_range[1] + 1)
        rand_sparsity = np.random.uniform(sparsity_threshold_range[0], sparsity_threshold_range[1])
        
        # Combine curriculum and random constraints
        max_weight = curriculum_weight * prog_max_weight + (1 - curriculum_weight) * rand_max_weight
        min_assets = int(curriculum_weight * prog_min_assets + (1 - curriculum_weight) * rand_min_assets)
        max_assets = int(curriculum_weight * prog_max_assets + (1 - curriculum_weight) * rand_max_assets)
        sparsity_threshold = curriculum_weight * prog_sparsity + (1 - curriculum_weight) * rand_sparsity
        
        # Use the create_batch function to get each batch
        past_batch_tensor, future_batch_tensor = create_batch(data_array, past_window_size, future_window_size, current_n_cols, current_batch_size, valid_indices)
        
        # Prepare data in the format expected by update_model with dual-purpose input normalization
        # past_batch_tensor shape: (batch_size, n_cols, past_window_size)
        # future_batch_tensor shape: (batch_size, n_cols, future_window_size)
        
        # Create input vectors for the model:
        # 1. Scalar input: future_window_size (prediction parameter) - both normalized and raw versions
        # 2. Constraint vector: [max_weight, min_assets, max_assets, sparsity_threshold] - both normalized and raw versions
        
        # Raw scalar input for constraint logic (actual future_window_size value)
        raw_scalar_input = torch.tensor(future_window_size, dtype=torch.float32)
        
        # Normalized scalar input for neural network (scale to reasonable range)
        # Normalize future_window_size to [0, 1] range assuming max reasonable value of 100
        normalized_scalar_input = torch.tensor(future_window_size / 100.0, dtype=torch.float32).unsqueeze(0).repeat(current_batch_size, 1)  # (batch_size, 1)
        
        # Constraint vector for portfolio constraints
        # CRITICAL: Ensure constraints are logically consistent with current batch
        
        # 1. Ensure max_assets doesn't exceed current number of assets
        effective_max_assets = min(max_assets, current_n_cols)
        
        # 2. Ensure min_assets doesn't exceed max_assets or current assets
        effective_min_assets = min(min_assets, effective_max_assets, current_n_cols)
        
        # 3. Ensure min_assets is reasonable (at least 1)
        effective_min_assets = max(1, effective_min_assets)
        
        # 4. Additional safety check: if constraints are impossible, use safe defaults
        if effective_min_assets > current_n_cols:
            print(f"Warning: min_assets ({effective_min_assets}) > available assets ({current_n_cols}). Using {current_n_cols//2}")
            effective_min_assets = max(1, current_n_cols // 2)
            
        if effective_max_assets < effective_min_assets:
            print(f"Warning: max_assets ({effective_max_assets}) < min_assets ({effective_min_assets}). Adjusting...")
            effective_max_assets = max(effective_min_assets, current_n_cols // 2)

        # Raw constraint values for constraint enforcement logic
        raw_constraints = {
            'max_weight': max_weight,
            'min_assets': effective_min_assets,
            'max_assets': effective_max_assets,
            'sparsity_threshold': sparsity_threshold
        }

        # Normalized constraint vector for neural network (consistent scale with other inputs)
        normalized_constraint_vector = torch.tensor([
            max_weight,  # Max weight already in [0, 1] range
            effective_min_assets / 100.0,  # Normalize min assets to [0, 1] range
            effective_max_assets / 100.0,  # Normalize max assets to [0, 1] range
            sparsity_threshold * 10.0  # Scale sparsity threshold to [0, 1] range (0.01-0.1 → 0.1-1.0)
        ], dtype=torch.float32)
        
        # Expand normalized constraint vector to batch size
        normalized_constraint_input = normalized_constraint_vector.unsqueeze(0).repeat(current_batch_size, 1)  # (batch_size, 4)
        
        # Format data as dictionaries expected by update_model (using normalized inputs for neural network)
        past_batch = {
            'matrix_input': past_batch_tensor,  # (batch_size, n_cols, past_window_size)
            'scalar_input': normalized_scalar_input,  # (batch_size, 1) - normalized future_window_size
            'constraint_input': normalized_constraint_input,  # (batch_size, 4) - normalized constraints
            'raw_constraints': raw_constraints  # Raw constraint values for enforcement logic
        }
        
        # For future batch, we need to convert the tensor to returns format
        # The future_batch_tensor contains normalized price continuations
        # We need to convert this to a format suitable for portfolio evaluation
        future_batch = {
            'returns': future_batch_tensor[0].T  # Use first sample, transpose to (timesteps, n_assets)
        }
        
        # Gradient accumulation: accumulate gradients over multiple sub-batches for larger effective batch size
        if accumulation_step == 0:
            # Zero gradients at the start of accumulation cycle
            enhanced_optimizer.zero_grad()
        
        # Update model with current loss aggregation method (accumulate gradients)
        loss_dict = update_model(model=model, optimizer=enhanced_optimizer, past_batch=past_batch, future_batch=future_batch, loss=loss,
                     max_weight=raw_constraints['max_weight'], 
                     min_assets=raw_constraints['min_assets'], 
                     max_assets=raw_constraints['max_assets'], 
                     sparsity_threshold=raw_constraints['sparsity_threshold'],
                     loss_aggregation=current_loss_aggregation,
                     gradient_accumulation_steps=gradient_accumulation_steps,
                     accumulation_step=accumulation_step)
        
        # Accumulate loss for logging
        accumulated_loss += loss_dict['loss'] / gradient_accumulation_steps
        accumulation_step += 1
        
        # Apply optimizer step and reset accumulation when cycle is complete
        if accumulation_step >= gradient_accumulation_steps:
            # Optimizer step is handled inside update_model for gradient accumulation
            # Update learning rate schedulers
            lr_scheduler.step()  # Step the cosine scheduler
            
            # Reset accumulation
            accumulation_step = 0
            current_iteration_loss = accumulated_loss
            accumulated_loss = 0.0
        else:
            # Still accumulating, use current loss for logging
            current_iteration_loss = loss_dict['loss']

        # Calculate additional metrics if requested
        additional_metrics = {}
        if other_metrics_list:
            try:
                # Get portfolio weights from the first sample in the batch
                portfolio_weights = loss_dict['weights'][0]  # Shape: (n_assets,)
                future_returns = future_batch['returns']  # Shape: (timesteps, n_assets)
                
                # Create portfolio time series for additional metric calculations
                portfolio_timeseries = create_portfolio_time_series(future_returns, portfolio_weights)
                
                # Calculate each additional metric
                for metric_name in other_metrics_list:
                    try:
                        metric_value = calculate_expected_metric(portfolio_timeseries, None, metric_name)
                        additional_metrics[metric_name] = metric_value.item() if hasattr(metric_value, 'item') else float(metric_value)
                    except Exception as e:
                        print(f"Warning: Could not calculate {metric_name}: {e}")
                        additional_metrics[metric_name] = float('nan')
                        
            except Exception as e:
                print(f"Warning: Could not calculate additional metrics: {e}")
                # Fill with NaN values if calculation fails
                for metric_name in other_metrics_list:
                    additional_metrics[metric_name] = float('nan')

        # Update learning rate schedulers (using enhanced optimizer)
        if plateau_scheduler is not None:
            plateau_scheduler.step(current_iteration_loss)

        # Early stopping check (use accumulated loss when available)
        if current_iteration_loss < best_loss - early_stopping_threshold:
            best_loss = current_iteration_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at iteration {i + 1}")
            print(f"Best loss: {best_loss:.6f}, Current loss: {current_iteration_loss:.6f}")
            break

        # Get current learning rate for logging
        current_lr = lr_scheduler.get_last_lr()[0]
        
        # Calculate effective batch size
        effective_batch_size = current_batch_size * gradient_accumulation_steps

        # ALWAYS log to dataframe (complete record for analysis)
        new_row_data = {
            'iteration': [i + 1], 
            'loss': [current_iteration_loss],
            'metric_loss': [loss_dict['metric_loss']],
            'reg_loss': [loss_dict['reg_loss']],
            'loss_aggregation': [current_loss_aggregation],
            'phase': [phase],
            'batch_size': [current_batch_size],
            'effective_batch_size': [effective_batch_size],
            'n_cols': [current_n_cols],
            'progress': [progress],
            'learning_rate': [current_lr]
        }
        # Add additional metrics to the row data
        for metric_name in other_metrics_list:
            new_row_data[metric_name] = [additional_metrics.get(metric_name, float('nan'))]
        
        new_row = pd.DataFrame(new_row_data)
        log = pd.concat([log, new_row], ignore_index=True)

        # Console output at specified frequency only
        if (i + 1) % log_frequency == 0:
            print(f"Iteration {i + 1}/{iterations} | Phase: {phase} | Loss: {current_iteration_loss:.6f} | Agg: {current_loss_aggregation.upper()} | Progress: {progress*100:.1f}% | LR: {current_lr:.2e} | Eff.Batch: {effective_batch_size}")

        # Save model checkpoint at specified frequency
        if (i + 1) % checkpoint_frequency == 0:
            checkpoint_filename = f'model_checkpoint_{i + 1}.pt'
            checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_filepath)

    # Save final complete model (architecture + weights)
    final_model_filename = 'final_trained_model.pt'
    final_model_filepath = os.path.join(checkpoint_path, final_model_filename)
    torch.save(model, final_model_filepath)
    
    # Also save final state dict for compatibility
    final_state_dict_filename = 'final_model_state_dict.pt'
    final_state_dict_filepath = os.path.join(checkpoint_path, final_state_dict_filename)
    torch.save(model.state_dict(), final_state_dict_filepath)

    # Save final log to CSV with datetime filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'training_log_{timestamp}.csv'
    log_filepath = os.path.join(log_path, log_filename)
    log.to_csv(log_filepath, index=False)

    end_time = datetime.now()
    print(f"\nTraining completed! Total time: {end_time - start_time}")
    print(f"Final loss: {current_iteration_loss:.6f} | Final aggregation: {current_loss_aggregation.upper()}")
    print(f"Final learning rate: {current_lr:.2e} | Final effective batch size: {effective_batch_size}")
    
    # Show final values of additional metrics if any
    if other_metrics_list and additional_metrics:
        final_metrics_str = " | ".join([f"{name}: {additional_metrics.get(name, float('nan')):.6f}" 
                                       for name in other_metrics_list 
                                       if not np.isnan(additional_metrics.get(name, float('nan')))])
        if final_metrics_str:
            print(f"Final additional metrics: {final_metrics_str}")
    
    print(f"Final model saved to: {final_model_filepath}")
    print(f"Training log saved to: {log_filepath}")
    return model
