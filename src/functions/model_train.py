"""
Model training functions for the Tesseract portfolio optimization system.
Contains training loops, optimization logic, and model update functions.
"""
#%% --------------- IMPORTS-----------------
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

#%% --------------- SHARED HELPER FUNCTIONS-----------------

def _create_single_sample(data_array, past_window_size, future_window_size, n_cols, valid_indices, max_retries=10):
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
            data_slice = data_slice.astype(np.float32)
        except (ValueError, TypeError):
            # If conversion fails, treat all values as invalid
            valid_col_indices = np.array([])
        else:
            # Find columns with no NaN values in this slice
            try:
                valid_cols = ~np.isnan(data_slice).any(axis=0)
                valid_col_indices = np.where(valid_cols)[0]
            except TypeError:
                # Handle case where isnan still fails after conversion
                valid_col_indices = np.array([])
    else:
        # Find columns with no NaN values in this slice
        try:
            valid_cols = ~np.isnan(data_slice).any(axis=0)
            valid_col_indices = np.where(valid_cols)[0]
        except TypeError:
            # Handle non-float data types that can't use isnan
            # Check for finite values instead
            try:
                valid_cols = np.isfinite(data_slice).all(axis=0)
                valid_col_indices = np.where(valid_cols)[0]
            except (TypeError, ValueError):
                # If all else fails, assume all columns are valid
                valid_col_indices = np.arange(data_slice.shape[1])
    
    # Ensure we always get a sample, even if we need to retry or pad
    retry_count = 0
    
    while len(valid_col_indices) < n_cols and retry_count < max_retries:
        # Try a different random starting point
        start_idx = random.randint(0, valid_indices)
        data_slice = data_array[start_idx:start_idx + (past_window_size + future_window_size), :]
        
        # Handle data type conversion and validation (same as above)
        if data_slice.dtype == 'object' or not np.issubdtype(data_slice.dtype, np.number):
            try:
                data_slice = data_slice.astype(np.float32)
            except (ValueError, TypeError):
                valid_col_indices = np.array([])
                retry_count += 1
                continue
        
        try:
            valid_cols = ~np.isnan(data_slice).any(axis=0)
            valid_col_indices = np.where(valid_cols)[0]
        except TypeError:
            try:
                valid_cols = np.isfinite(data_slice).all(axis=0)
                valid_col_indices = np.where(valid_cols)[0]
            except (TypeError, ValueError):
                valid_col_indices = np.arange(data_slice.shape[1])
        
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

def _create_batch(data_array, past_window_size, future_window_size, n_cols, batch_size, valid_indices):
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
        sample = _create_single_sample(data_array, past_window_size, future_window_size, n_cols, valid_indices)
        batch.append(sample)
    
    # Convert list of numpy arrays to tensor - automatically gets correct shape!
    batch_array = np.array(batch)  # Shape: (batch_size, n_cols, past_window_size + future_window_size)
    
    # Split into past and future batches
    past_batch = torch.tensor(batch_array[:, :, :past_window_size], dtype=torch.float32)      # First past_window_size timesteps
    future_batch = torch.tensor(batch_array[:, :, past_window_size:past_window_size + future_window_size], dtype=torch.float32)   # Next future_window_size timesteps
    
    return past_batch, future_batch

def _update_model(model, optimizer, past_batch, future_batch, loss, 
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
    # The calling function (train_model_progressive) handles optimizer.zero_grad() timing
    
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

# %% --------------- PROGRESSIVE MODEL TRAINING FUNCTIONS-----------------
# Private helper functions for progressive training
def _progressive_process_other_metrics(other_metrics_to_log):
    """Process and validate the other_metrics_to_log parameter."""
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
    
    return other_metrics_list

def _progressive_setup_default_paths(log_path, checkpoint_path, checkpoint_frequency, log_frequency):
    """Set up default paths for logs and checkpoints."""
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
    
    return log_path, checkpoint_path

def _progressive_setup_enhanced_optimizer(model, learning_rate, weight_decay):
    """Set up enhanced AdamW optimizer with parameter grouping."""
    print(f"Configuring enhanced optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    
    # Parameter grouping: no weight decay on biases and layer norms
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
    return enhanced_optimizer

def _progressive_setup_learning_rate_scheduler(enhanced_optimizer, warmup_steps, iterations):
    """Set up learning rate scheduler with warmup and cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay after warmup
            # Avoid division by zero when iterations == warmup_steps
            if iterations == warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / (iterations - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    lr_scheduler = LambdaLR(enhanced_optimizer, lr_lambda)
    print(f"Learning rate scheduler: {warmup_steps} warmup steps, cosine decay over {iterations} iterations")
    return lr_scheduler

def _progressive_setup_plateau_scheduler(enhanced_optimizer, use_scheduler, scheduler_patience):
    """Set up additional plateau scheduler for fine-tuning."""
    if use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        plateau_scheduler = ReduceLROnPlateau(enhanced_optimizer, mode='min', factor=0.5, 
                                            patience=scheduler_patience, 
                                            min_lr=1e-7)
        print(f"Additional plateau scheduler enabled with patience={scheduler_patience}")
        return plateau_scheduler
    else:
        return None

def _progressive_convert_data_to_array(data):
    """Convert DataFrame to numpy array if needed."""
    if hasattr(data, 'values'):  # Check if it's a DataFrame
        data_array = data.values
        print("Converted DataFrame to numpy array for faster processing")
        return data_array
    else:
        return data

def _progressive_determine_loss_aggregation(loss_aggregation, progress, i):
    """Determine current loss aggregation method and phase."""
    prev_aggregation = None
    if i > 0:  # Track previous aggregation method for transition detection
        prev_progress = (i - 1) / (max(1, i - 1)) if i > 1 else 0
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
    
    return current_loss_aggregation, phase

def _progressive_calculate_progressive_values(progress, min_n_cols, max_n_cols, min_batch_size, max_batch_size):
    """Calculate progressive n_cols and batch_size values."""
    # Progressive n_cols: start small, end large
    current_n_cols = int(min_n_cols + progress * (max_n_cols - min_n_cols))
    
    # Progressive batch_size: start small, end large  
    current_batch_size = int(min_batch_size + progress * (max_batch_size - min_batch_size))
    
    return current_n_cols, current_batch_size

def _progressive_calculate_constraint_values(progress, max_weight_range, min_assets_range, 
                                           max_assets_range, sparsity_threshold_range):
    """Calculate progressive and random constraint values."""
    # More structured constraint progression (smoother than fully random)
    # Use a mix of curriculum learning (progressive) and random sampling
    curriculum_weight = 0.7  # 70% curriculum, 30% random
    
    # Progressive constraint difficulty
    prog_max_weight = max_weight_range[1] - progress * (max_weight_range[1] - max_weight_range[0])  # Start loose, get stricter
    prog_min_assets = min_assets_range[0] + progress * (min_assets_range[1] - min_assets_range[0])  # Start few, require more
    prog_max_assets = max_assets_range[1] - progress * (max_assets_range[1] - max_assets_range[0])  # Start many, limit more
    prog_sparsity = sparsity_threshold_range[0] + progress * (sparsity_threshold_range[1] - sparsity_threshold_range[0])  # Start low, increase
    
    # Random sampling within ranges
    import numpy as np
    rand_max_weight = np.random.uniform(max_weight_range[0], max_weight_range[1])
    rand_min_assets = np.random.randint(min_assets_range[0], min_assets_range[1] + 1)
    rand_max_assets = np.random.randint(max_assets_range[0], max_assets_range[1] + 1)
    rand_sparsity = np.random.uniform(sparsity_threshold_range[0], sparsity_threshold_range[1])
    
    # Combine curriculum and random constraints
    max_weight = curriculum_weight * prog_max_weight + (1 - curriculum_weight) * rand_max_weight
    min_assets = int(curriculum_weight * prog_min_assets + (1 - curriculum_weight) * rand_min_assets)
    max_assets = int(curriculum_weight * prog_max_assets + (1 - curriculum_weight) * rand_max_assets)
    sparsity_threshold = curriculum_weight * prog_sparsity + (1 - curriculum_weight) * rand_sparsity
    
    return max_weight, min_assets, max_assets, sparsity_threshold

def _progressive_validate_constraints(max_weight, min_assets, max_assets, sparsity_threshold, current_n_cols):
    """Validate and adjust constraints to be logically consistent."""
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
    
    return max_weight, effective_min_assets, effective_max_assets, sparsity_threshold

def _progressive_create_model_inputs(future_window_size, current_batch_size, max_weight, 
                                   effective_min_assets, effective_max_assets, sparsity_threshold):
    """Create normalized inputs for the model."""
    import torch
    
    # Raw scalar input for constraint logic (actual future_window_size value)
    raw_scalar_input = torch.tensor(future_window_size, dtype=torch.float32)
    
    # Normalized scalar input for neural network (scale to reasonable range)
    # Normalize future_window_size to [0, 1] range assuming max reasonable value of 100
    normalized_scalar_input = torch.tensor(future_window_size / 100.0, dtype=torch.float32).unsqueeze(0).repeat(current_batch_size, 1)  # (batch_size, 1)
    
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
    
    return normalized_scalar_input, normalized_constraint_input, raw_constraints

def _progressive_calculate_additional_metrics(other_metrics_list, loss_dict, future_batch):
    """Calculate additional metrics if requested."""
    additional_metrics = {}
    if other_metrics_list:
        try:
            # Get portfolio weights from the first sample in the batch
            portfolio_weights = loss_dict['weights'][0]  # Shape: (n_assets,)
            future_returns = future_batch['returns']  # Shape: (timesteps, n_assets)
            
            # Create portfolio time series for additional metric calculations
            portfolio_timeseries = create_portfolio_time_series(future_returns, portfolio_weights)
            
            # Metrics that are negated for PyTorch optimization (need to flip sign for logging)
            negated_metrics = {
                'sharpe_ratio', 'geometric_sharpe_ratio', 'sortino_ratio', 'geometric_sortino_ratio',
                'expected_return', 'carmdd', 'omega_ratio', 'jensen_alpha', 'treynor_ratio', 'k_ratio'
            }
            
            # Metrics that are naturally positive (no sign change needed)
            positive_metrics = {'max_drawdown', 'ulcer_index'}
            
            # Calculate each additional metric
            for metric_name in other_metrics_list:
                try:
                    metric_value = calculate_expected_metric(portfolio_timeseries, None, metric_name)
                    value = metric_value.item() if hasattr(metric_value, 'item') else float(metric_value)
                    
                    # Convert to positive values for logging readability
                    if metric_name in negated_metrics:
                        value = -value  # Flip sign back to positive for logging
                    elif metric_name in positive_metrics:
                        value = value  # Keep as-is (already positive)
                    
                    additional_metrics[metric_name] = value
                except Exception as e:
                    print(f"Warning: Could not calculate {metric_name}: {e}")
                    additional_metrics[metric_name] = float('nan')
                    
        except Exception as e:
            print(f"Warning: Could not calculate additional metrics: {e}")
            # Fill with NaN values if calculation fails
            for metric_name in other_metrics_list:
                additional_metrics[metric_name] = float('nan')
    
    return additional_metrics

def _progressive_save_final_models_and_logs(model, checkpoint_path, log_path, log, current_iteration_loss, 
                                          current_loss_aggregation, current_lr, effective_batch_size, 
                                          other_metrics_list, additional_metrics, start_time):
    """Save final models and logs."""
    import pandas as pd
    import torch
    import numpy as np
    from datetime import datetime
    
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

# Main Progressive Training Function
def train_model_progressive(model, optimizer, data, past_window_size, future_window_size, min_n_cols = 10, 
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
    
    NOTE: Logging occurs only after gradient updates (every gradient_accumulation_steps iterations) to ensure
    meaningful loss values. This reduces log size and focuses on actual optimization steps.
    
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
            NOTE: Metrics are logged with positive values for readability (signs are flipped from optimization values)
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
            NOTE: Logging only occurs after complete gradient update cycles to show meaningful loss values.
        
    Returns:
        Trained model
        
    Example:
        >>> # Train with progressive loss aggregation and enhanced stability
        >>> trained_model = train_model_progressive(
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
    # Start time
    start_time = datetime.now()
    print("Training started at", start_time)

    # Process and validate other_metrics_to_log parameter
    other_metrics_list = _progressive_process_other_metrics(other_metrics_to_log)

    # Set up default paths and create directories
    log_path, checkpoint_path = _progressive_setup_default_paths(
        log_path, checkpoint_path, checkpoint_frequency, log_frequency)

    # Set up enhanced optimizer with parameter grouping
    enhanced_optimizer = _progressive_setup_enhanced_optimizer(model, learning_rate, weight_decay)
    
    # Set up learning rate scheduler with warmup and cosine decay
    lr_scheduler = _progressive_setup_learning_rate_scheduler(enhanced_optimizer, warmup_steps, iterations)

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
        'loss_aggregation', 'phase', 'batch_size', 'effective_batch_size', 'n_cols', 'progress', 'learning_rate',
        # Add constraint ranges for context
        'max_weight_range', 'min_assets_range', 'max_assets_range', 'sparsity_threshold_range',
        # Add actual constraint values used for this specific iteration
        'max_weight_used', 'min_assets_used', 'max_assets_used', 'sparsity_threshold_used'
    ]
    log_columns.extend(other_metrics_list)
    log = pd.DataFrame(columns=log_columns)

    # Set up additional plateau scheduler for fine-tuning
    plateau_scheduler = _progressive_setup_plateau_scheduler(enhanced_optimizer, use_scheduler, scheduler_patience)

    # Convert DataFrame to numpy array if needed
    data_array = _progressive_convert_data_to_array(data)
    
    valid_indices = len(data_array) - (past_window_size + future_window_size)
    
    print(f"Starting training for {iterations} iterations...")
    
    # Loop through fixed number of iterations with progressive difficulty
    for i in range(iterations):
        # Calculate progressive values (linear interpolation)
        progress = i / (iterations - 1) if iterations > 1 else 0  # 0 to 1
        
        # Determine current loss aggregation method and phase
        current_loss_aggregation, phase = _progressive_determine_loss_aggregation(
            loss_aggregation, progress, i)

        # Calculate progressive n_cols and batch_size values
        current_n_cols, current_batch_size = _progressive_calculate_progressive_values(
            progress, min_n_cols, max_n_cols, min_batch_size, max_batch_size)
        
        # Calculate progressive and random constraint values
        max_weight, min_assets, max_assets, sparsity_threshold = _progressive_calculate_constraint_values(
            progress, max_weight_range, min_assets_range, max_assets_range, sparsity_threshold_range)
        
        # Use the create_batch function to get each batch
        past_batch_tensor, future_batch_tensor = _create_batch(
            data_array, past_window_size, future_window_size, current_n_cols, current_batch_size, valid_indices)
        
        # Validate and adjust constraints to be logically consistent
        max_weight, effective_min_assets, effective_max_assets, sparsity_threshold = _progressive_validate_constraints(
            max_weight, min_assets, max_assets, sparsity_threshold, current_n_cols)

        # Create normalized inputs for the model
        normalized_scalar_input, normalized_constraint_input, raw_constraints = _progressive_create_model_inputs(
            future_window_size, current_batch_size, max_weight, effective_min_assets, effective_max_assets, sparsity_threshold)
        
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
        loss_dict = _update_model(model=model, optimizer=enhanced_optimizer, past_batch=past_batch, future_batch=future_batch, loss=loss,
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
        gradient_update_occurred = False
        if accumulation_step >= gradient_accumulation_steps:
            # Optimizer step is handled inside update_model for gradient accumulation
            # Update learning rate schedulers
            lr_scheduler.step()  # Step the cosine scheduler
            
            # Reset accumulation
            accumulation_step = 0
            current_iteration_loss = accumulated_loss
            accumulated_loss = 0.0
            gradient_update_occurred = True
        else:
            # Still accumulating, use current loss for logging
            current_iteration_loss = loss_dict['loss']

        # Calculate additional metrics whenever we're about to log (not just on gradient updates)
        if (i + 1) % gradient_accumulation_steps == 0:
            additional_metrics = _progressive_calculate_additional_metrics(other_metrics_list, loss_dict, future_batch)
        else:
            additional_metrics = {}  # Empty dict when not logging

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

        # Log every gradient_accumulation_steps iterations
        # For gradient_accumulation_steps=2: log on iterations 2, 4, 6, 8, etc.
        if (i + 1) % gradient_accumulation_steps == 0:
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
                'learning_rate': [current_lr],
                # Add constraint ranges for context
                'max_weight_range': [f"{max_weight_range[0]:.3f}-{max_weight_range[1]:.3f}"],
                'min_assets_range': [f"{min_assets_range[0]}-{min_assets_range[1]}"],
                'max_assets_range': [f"{max_assets_range[0]}-{max_assets_range[1]}"],
                'sparsity_threshold_range': [f"{sparsity_threshold_range[0]:.3f}-{sparsity_threshold_range[1]:.3f}"],
                # Add actual constraint values used for this specific iteration
                'max_weight_used': [raw_constraints['max_weight']],
                'min_assets_used': [raw_constraints['min_assets']],
                'max_assets_used': [raw_constraints['max_assets']],
                'sparsity_threshold_used': [raw_constraints['sparsity_threshold']]
            }
            # Add additional metrics to the row data
            for metric_name in other_metrics_list:
                new_row_data[metric_name] = [additional_metrics.get(metric_name, float('nan'))]
            
            new_row = pd.DataFrame(new_row_data)
            log = pd.concat([log, new_row], ignore_index=True)

            # Console output at specified frequency only (based on logged iterations, not all iterations)
            logged_iterations = len(log)
            if logged_iterations % log_frequency == 0:
                print(f"Iteration {i + 1}/{iterations} | Phase: {phase} | Loss: {current_iteration_loss:.6f} | Agg: {current_loss_aggregation.upper()} | Progress: {progress*100:.1f}% | LR: {current_lr:.2e} | Eff.Batch: {effective_batch_size}")

            # Save model checkpoint at specified frequency (based on logged iterations)
            if logged_iterations % checkpoint_frequency == 0:
                checkpoint_filename = f'model_checkpoint_{i + 1}.pt'
                checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
                torch.save(model.state_dict(), checkpoint_filepath)

    # Save final models and logs
    _progressive_save_final_models_and_logs(
        model, checkpoint_path, log_path, log, current_iteration_loss, 
        current_loss_aggregation, current_lr, effective_batch_size, 
        other_metrics_list, additional_metrics, start_time)
    
    return model

# %% --------------- CURRICULUM MODEL TRAINING HELPER FUNCTIONS-----------------
# Private helper functions for curriculum training
def _validate_power_of_2(n):
    """Check if a number is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def _validate_and_create_batch_schedule(batch_schedule, n_batch_phases, min_batch_size, max_batch_size, iterations):
    """Validate and create batch schedule with power-of-2 validation."""
    if batch_schedule is None:
        # Auto-generate batch schedule with powers of 2
        batch_sizes = []
        current_size = min_batch_size
        while current_size <= max_batch_size and len(batch_sizes) < n_batch_phases:
            if _validate_power_of_2(current_size):
                batch_sizes.append(current_size)
            current_size *= 2
        
        # If we don't have enough powers of 2 within range, use what we have
        if len(batch_sizes) == 0:
            raise ValueError(f"No valid powers of 2 found between {min_batch_size} and {max_batch_size}")
        
        # If we have fewer batch sizes than requested phases, use what we have
        if len(batch_sizes) < n_batch_phases:
            print(f"Warning: Only {len(batch_sizes)} valid powers of 2 found, using {len(batch_sizes)} phases instead of {n_batch_phases}")
        
        # Distribute iterations evenly across batch sizes
        iterations_per_phase = iterations // len(batch_sizes)
        remainder = iterations % len(batch_sizes)
        
        batch_schedule = {}
        for i, batch_size in enumerate(batch_sizes):
            iters = iterations_per_phase + (1 if i < remainder else 0)
            batch_schedule[batch_size] = iters
        
        print(f"Auto-generated batch schedule: {batch_schedule} (requested {n_batch_phases} phases, created {len(batch_sizes)})")
    else:
        # Validate provided batch schedule
        for batch_size in batch_schedule.keys():
            if not _validate_power_of_2(batch_size):
                raise ValueError(f"Batch size {batch_size} is not a power of 2")
        
        if sum(batch_schedule.values()) != iterations:
            raise ValueError(f"Batch schedule iterations ({sum(batch_schedule.values())}) != total iterations ({iterations})")
        
        print(f"Using provided batch schedule: {batch_schedule}")
    
    return batch_schedule

def _validate_and_create_column_schedule(column_schedule, n_column_buckets, iterations):
    """Validate and create column schedule."""
    if column_schedule is None:
        # Auto-generate column schedule
        iterations_per_bucket = iterations // n_column_buckets
        remainder = iterations % n_column_buckets
        
        column_schedule = {}
        for i in range(1, n_column_buckets + 1):
            iters = iterations_per_bucket + (1 if i <= remainder else 0)
            column_schedule[i] = iters
        
        print(f"Auto-generated column schedule: {column_schedule}")
    else:
        # Validate provided column schedule
        if sum(column_schedule.values()) != iterations:
            raise ValueError(f"Column schedule iterations ({sum(column_schedule.values())}) != total iterations ({iterations})")
        
        print(f"Using provided column schedule: {column_schedule}")
    
    return column_schedule

def _create_constraint_steps(n_steps, param_range):
    """Create constraint expansion from median outward."""
    min_val, max_val = param_range
    median = (min_val + max_val) / 2
    range_size = max_val - min_val
    
    steps = []
    for step in range(n_steps):
        # Linear expansion from median outward
        expansion_factor = step / (n_steps - 1) if n_steps > 1 else 0
        half_range = (range_size / 2) * expansion_factor
        
        step_min = max(min_val, median - half_range)
        step_max = min(max_val, median + half_range)
        steps.append((step_min, step_max))
    
    return steps

def _create_cartesian_schedule(batch_schedule, column_schedule, constraint_n_steps, iterations):
    """Create cartesian product schedule and group into phases."""
    # Create all combinations
    batch_items = [(size, iters) for size, iters in batch_schedule.items()]
    column_items = [(bucket, iters) for bucket, iters in column_schedule.items()]
    
    # Expand each schedule to iteration-level assignments
    batch_assignments = []
    for batch_size, iters in batch_items:
        batch_assignments.extend([batch_size] * iters)
    
    column_assignments = []
    for bucket_id, iters in column_items:
        column_assignments.extend([bucket_id] * iters)
    
    # Constraint assignments (evenly distributed)
    constraint_assignments = []
    iters_per_constraint = iterations // constraint_n_steps
    remainder = iterations % constraint_n_steps
    for step in range(constraint_n_steps):
        iters = iters_per_constraint + (1 if step < remainder else 0)
        constraint_assignments.extend([step] * iters)
    
    # Create iteration-level schedule
    iteration_schedule = []
    for i in range(iterations):
        iteration_schedule.append({
            'iteration': i + 1,
            'batch_size': batch_assignments[i],
            'column_bucket': column_assignments[i],
            'constraint_step': constraint_assignments[i]
        })
    
    # Group consecutive iterations with same parameters into phases
    phases = []
    current_phase = None
    
    for item in iteration_schedule:
        key = (item['batch_size'], item['column_bucket'], item['constraint_step'])
        
        if current_phase is None or current_phase['key'] != key:
            # Start new phase
            if current_phase is not None:
                phases.append(current_phase)
            
            current_phase = {
                'key': key,
                'batch_size': item['batch_size'],
                'column_bucket': item['column_bucket'],
                'constraint_step': item['constraint_step'],
                'start_iteration': item['iteration'],
                'iterations': 1
            }
        else:
            # Continue current phase
            current_phase['iterations'] += 1
    
    # Don't forget the last phase
    if current_phase is not None:
        phases.append(current_phase)
    
    return phases

def _create_column_buckets(total_cols, n_buckets, max_reasonable_cols=500):
    """Create column range buckets for curriculum learning.
    
    Curriculum progression: Start with MORE columns (easier attention), progress to FEWER columns (harder).
    Bucket 1 = most columns, highest bucket = fewest columns.
    
    Args:
        total_cols: Total number of columns in dataset
        n_buckets: Number of buckets to create
        max_reasonable_cols: Maximum reasonable number of columns for memory constraints
    """
    # Cap the effective total columns to a reasonable number for memory constraints
    effective_total_cols = min(total_cols, max_reasonable_cols)
    
    buckets = {}
    for bucket_id in range(1, n_buckets + 1):
        # Curriculum learning: start with MORE columns (easier), progress to FEWER columns (harder)
        # Reverse the progress so bucket 1 = most columns, bucket n = fewest columns
        progress = (bucket_id - 1) / (n_buckets - 1) if n_buckets > 1 else 0  # 0 to 1
        reversed_progress = 1.0 - progress  # 1 to 0 (bucket 1 gets 1.0, last bucket gets 0.0)
        
        # Calculate ranges: early buckets = more cols, later buckets = fewer cols
        # Start with ~100-500 columns, end with ~10-50 columns
        min_start, min_end = 10, max(50, int(effective_total_cols * 0.2))
        max_start, max_end = 50, effective_total_cols
        
        min_cols = int(min_start + reversed_progress * (min_end - min_start))
        max_cols = int(max_start + reversed_progress * (max_end - max_start))
        
        # Ensure valid ranges
        min_cols = max(10, min(min_cols, effective_total_cols))
        max_cols = max(min_cols + 10, min(max_cols, effective_total_cols))
        
        buckets[bucket_id] = (min_cols, max_cols)
    return buckets

def _setup_enhanced_optimizer(model, learning_rate, weight_decay):
    """Set up enhanced AdamW optimizer with parameter grouping."""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    from torch.optim import AdamW
    enhanced_optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-7,
        weight_decay=weight_decay
    )
    
    print(f"Enhanced optimizer: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")
    return enhanced_optimizer

def _setup_learning_rate_scheduler(optimizer, warmup_steps, iterations):
    """Set up learning rate scheduler with warmup and cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Avoid division by zero when iterations == warmup_steps
            if iterations == warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / (iterations - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    print(f"Learning rate scheduler: {warmup_steps} warmup steps, cosine decay over {iterations} iterations")
    return lr_scheduler

def _setup_plateau_scheduler(optimizer, use_scheduler, scheduler_patience):
    """Set up optional plateau scheduler."""
    if use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                            patience=scheduler_patience, min_lr=1e-7)
        print(f"Additional plateau scheduler enabled with patience={scheduler_patience}")
        return plateau_scheduler
    else:
        return None

class _BootstrapColumnSampler:
    """Bootstrap column sampler with coverage guarantee."""
    
    def __init__(self, total_columns):
        self.total_columns = total_columns
        self.available_columns = set(range(total_columns))
        self.used_columns = set()
        self.current_bucket = None
    
    def reset_bootstrap(self, bucket_id):
        """Reset bootstrap sampling for new bucket."""
        self.used_columns = set()
        self.current_bucket = bucket_id
        print(f"🔄 Bootstrap reset for column bucket {bucket_id}")
    
    def sample_columns(self, bucket_id, n_cols, column_buckets):
        """Sample columns using bootstrap strategy with coverage guarantee."""
        min_cols, max_cols = column_buckets[bucket_id]
        
        # Reset if new bucket or if we need more columns than available unused
        if (self.current_bucket != bucket_id or 
            len(self.available_columns - self.used_columns) < n_cols):
            self.reset_bootstrap(bucket_id)
        
        available_unused = self.available_columns - self.used_columns
        
        if len(available_unused) >= n_cols:
            # Use unused columns + fill remainder randomly
            unused_list = list(available_unused)
            selected_columns = random.sample(unused_list, n_cols)
        else:
            # Use all unused + sample remainder from all available
            unused_list = list(available_unused)
            remaining_needed = n_cols - len(unused_list)
            additional = random.sample(list(self.available_columns), remaining_needed)
            selected_columns = unused_list + additional
        
        # Update used columns
        self.used_columns.update(selected_columns)
        
        return selected_columns

def _get_loss_aggregation(loss_aggregation, progress):
    """Get current loss aggregation method based on progress."""
    if loss_aggregation == 'progressive':
        if progress <= 0.4:
            return 'huber', "Stability (Huber)"
        elif progress <= 0.7:
            return 'gmae', "Balanced (GMAE)"
        else:
            return 'gmse', "Performance (GMSE)"
    else:
        return loss_aggregation, f"Fixed ({loss_aggregation.upper()})"

def _calculate_additional_metrics(other_metrics_list, loss_dict, future_batch):
    """Calculate additional metrics if requested."""
    additional_metrics = {}
    if other_metrics_list:
        try:
            portfolio_weights = loss_dict['weights'][0]
            future_returns = future_batch['returns']
            portfolio_timeseries = create_portfolio_time_series(future_returns, portfolio_weights)
            
            # Metrics that are negated for PyTorch optimization (need to flip sign for logging)
            negated_metrics = {
                'sharpe_ratio', 'geometric_sharpe_ratio', 'sortino_ratio', 'geometric_sortino_ratio',
                'expected_return', 'carmdd', 'omega_ratio', 'jensen_alpha', 'treynor_ratio', 'k_ratio'
            }
            
            # Metrics that are naturally positive (no sign change needed)
            positive_metrics = {'max_drawdown', 'ulcer_index'}
            
            for metric_name in other_metrics_list:
                try:
                    metric_value = calculate_expected_metric(portfolio_timeseries, None, metric_name)
                    value = metric_value.item() if hasattr(metric_value, 'item') else float(metric_value)
                    
                    # Convert to positive values for logging readability
                    if metric_name in negated_metrics:
                        value = -value  # Flip sign back to positive for logging
                    elif metric_name in positive_metrics:
                        value = value  # Keep as-is (already positive)
                    
                    additional_metrics[metric_name] = value
                except Exception as e:
                    additional_metrics[metric_name] = float('nan')
        except Exception as e:
            for metric_name in other_metrics_list:
                additional_metrics[metric_name] = float('nan')
    
    return additional_metrics

# Main Curriculum Training Function
def train_model_curriculum(model, optimizer, data, past_window_size, 
                           # Curriculum scheduling parameters
                           batch_schedule=None, n_batch_phases=3, min_batch_size=32, max_batch_size=256,
                           column_schedule=None, n_column_buckets=4, total_columns=None,
                           constraint_n_steps=5, max_weight_range=(0.1, 1.0), min_assets_range=(0, 50), 
                           max_assets_range=(5, 200), sparsity_threshold_range=(0.005, 0.05),
                           future_window_range=(5, 50),
                           # Training parameters
                           iterations=1000, loss='sharpe_ratio', loss_aggregation='progressive',
                           # Additional metrics to log
                           other_metrics_to_log=None,
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
    Curriculum-based training with structured phase progression, bootstrap column sampling, and coordinated scheduling.
    Uses cartesian product approach to create sequential phases with different parameters.
    
    NOTE: Logging occurs only after gradient updates (every gradient_accumulation_steps iterations) to ensure
    meaningful loss values. This reduces log size and focuses on actual optimization steps.
    
    Args:
        model: The GPT2LikeTransformer model to train
        optimizer: PyTorch optimizer (will be replaced with enhanced AdamW)
        data: pandas DataFrame or numpy array of shape (n_timesteps, n_assets)
        past_window_size: Number of timesteps for past window (fixed)
        
        # Curriculum scheduling parameters
        batch_schedule: Dictionary {batch_size: iterations} (must be powers of 2) or None for auto-generation
            Example: {32: 200, 64: 400, 128: 400} 
        n_batch_phases: Number of batch phases to auto-generate if batch_schedule=None
        min_batch_size: Minimum batch size for auto-generation (must be power of 2)
        max_batch_size: Maximum batch size for auto-generation (must be power of 2)
        
        column_schedule: Dictionary {bucket_id: iterations} or None for auto-generation
            Example: {1: 200, 2: 300, 3: 500} where 1=widest bucket, higher=narrower
        n_column_buckets: Number of column buckets to auto-generate if column_schedule=None
        total_columns: Total number of columns in dataset (auto-detected if None)
        
        constraint_n_steps: Number of constraint expansion steps (from narrow to wide ranges)
        max_weight_range: (min, max) range for max_weight constraint expansion
        min_assets_range: (min, max) range for min_assets constraint expansion
        max_assets_range: (min, max) range for max_assets constraint expansion
        sparsity_threshold_range: (min, max) range for sparsity_threshold expansion
        
        future_window_range: (min, max) range for uniform sampling of future_window_size per batch
        
        # Training parameters
        iterations: Total number of iterations
        loss: Loss function to optimize ('sharpe_ratio', 'geometric_sharpe_ratio', etc.)
        loss_aggregation: Method to aggregate losses across batch (same as progressive function)
        
        # Other parameters (same as train_model_progressive)
        other_metrics_to_log: Additional metrics to log
            NOTE: Metrics are logged with positive values for readability (signs are flipped from optimization values)
        log_path: Path to save training logs
        checkpoint_path: Path to save model checkpoints
        checkpoint_frequency: How often to save model checkpoints
        log_frequency: How often to save loss data and print progress
        learning_rate: Initial learning rate for enhanced optimizer
        weight_decay: L2 regularization for enhanced optimizer
        warmup_steps: Learning rate warmup steps
        gradient_accumulation_steps: Steps to accumulate gradients for larger effective batch size
            NOTE: Logging only occurs after complete gradient update cycles to show meaningful loss values.
        
    Returns:
        Trained model
        
    Example:
        >>> # Train with curriculum learning using structured phases
        >>> trained_model = train_model_curriculum(
        ...     model=model, 
        ...     optimizer=optimizer, 
        ...     data=df, 
        ...     past_window_size=20,
        ...     batch_schedule={32: 200, 64: 400, 128: 400},  # Powers of 2, sum to 1000
        ...     column_schedule={1: 300, 2: 400, 3: 300},     # Buckets: wide→narrow
        ...     constraint_n_steps=5,                         # 5 constraint expansion steps
        ...     future_window_range=(5, 50),                  # Uniform sampling per batch
        ...     iterations=1000,
        ...     loss='sharpe_ratio',
        ...     loss_aggregation='progressive'
        ... )
    """
    # Start time
    start_time = datetime.now()
    print("🎓 Curriculum Training started at", start_time)
    
    # Convert DataFrame to numpy array if needed and ensure numeric data
    if hasattr(data, 'values'):
        data_array = data.values
        print("Converted DataFrame to numpy array for faster processing")
    else:
        data_array = data
    
    # Ensure the data is numeric - convert to float32 for memory efficiency
    if data_array.dtype == 'object' or not np.issubdtype(data_array.dtype, np.number):
        print(f"Converting data from {data_array.dtype} to float32 for numeric operations (memory efficient)")
        try:
            data_array = data_array.astype(np.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert data to numeric format. Data contains non-numeric values: {e}")
    
    print(f"Data array shape: {data_array.shape}, dtype: {data_array.dtype}")
    
    # Auto-detect total columns if not provided
    if total_columns is None:
        total_columns = data_array.shape[1]
        print(f"Auto-detected total columns: {total_columns}")
    
    # Validate and create schedules
    batch_schedule = _validate_and_create_batch_schedule(
        batch_schedule, n_batch_phases, min_batch_size, max_batch_size, iterations)
    
    column_schedule = _validate_and_create_column_schedule(
        column_schedule, n_column_buckets, iterations)
    
    # Create constraint expansion steps
    max_weight_steps = _create_constraint_steps(constraint_n_steps, max_weight_range)
    min_assets_steps = _create_constraint_steps(constraint_n_steps, min_assets_range)
    max_assets_steps = _create_constraint_steps(constraint_n_steps, max_assets_range)
    sparsity_steps = _create_constraint_steps(constraint_n_steps, sparsity_threshold_range)
    
    print(f"Constraint expansion steps: {constraint_n_steps}")
    print(f"  Max weight: {max_weight_steps}")
    print(f"  Min assets: {min_assets_steps}")
    print(f"  Max assets: {max_assets_steps}")
    print(f"  Sparsity: {sparsity_steps}")
    
    # Create cartesian product schedule and group into phases
    phases = _create_cartesian_schedule(batch_schedule, column_schedule, constraint_n_steps, iterations)
    
    print(f"\n📅 Generated {len(phases)} training phases:")
    for i, phase in enumerate(phases):
        print(f"  Phase {i+1}: Batch={phase['batch_size']}, Column={phase['column_bucket']}, "
              f"Constraint={phase['constraint_step']}, Iterations={phase['iterations']}")
    
    # Create column bucket definitions
    column_buckets = _create_column_buckets(total_columns, max(column_schedule.keys()))
    print(f"\n🗂️  Column buckets:")
    for bucket_id, (min_cols, max_cols) in column_buckets.items():
        print(f"  Bucket {bucket_id}: {min_cols}-{max_cols} columns")
    
    # Process other_metrics_to_log parameter (same as progressive function)
    if other_metrics_to_log is None:
        other_metrics_list = []
    elif isinstance(other_metrics_to_log, str):
        other_metrics_list = [other_metrics_to_log]
    elif isinstance(other_metrics_to_log, list):
        other_metrics_list = other_metrics_to_log
    else:
        raise ValueError("other_metrics_to_log must be None, a string, or a list of strings")
    
    # Validate metrics (same as progressive function)
    available_metrics = ['sharpe_ratio', 'geometric_sharpe_ratio', 'max_drawdown', 
                        'sortino_ratio', 'geometric_sortino_ratio', 'expected_return', 
                        'carmdd', 'omega_ratio', 'jensen_alpha', 'treynor_ratio', 
                        'ulcer_index', 'k_ratio']
    
    for metric_name in other_metrics_list:
        if metric_name not in available_metrics:
            raise ValueError(f"Unknown metric '{metric_name}'. Available metrics: {available_metrics}")
    
    if other_metrics_list:
        print(f"Additional metrics to log: {other_metrics_list}")
    
    # Set up default paths (same as progressive function)
    if log_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(current_dir))
        log_path = os.path.join(repo_root, "logs")
    
    if checkpoint_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(current_dir))
        checkpoint_path = os.path.join(repo_root, "checkpoints")
    
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    print(f"\n📁 Logs: {log_path}")
    print(f"📁 Checkpoints: {checkpoint_path}")
    
    # Set up log file path for intermediate logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'curriculum_training_log_{timestamp}.csv'
    log_filepath = os.path.join(log_path, log_filename)
    
    # Set up enhanced optimizer and schedulers
    print(f"\n⚙️  Configuring enhanced optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    enhanced_optimizer = _setup_enhanced_optimizer(model, learning_rate, weight_decay)
    lr_scheduler = _setup_learning_rate_scheduler(enhanced_optimizer, warmup_steps, iterations)
    plateau_scheduler = _setup_plateau_scheduler(enhanced_optimizer, use_scheduler, scheduler_patience)
    
    # Early stopping and gradient accumulation setup
    best_loss = float('inf')
    patience_counter = 0
    accumulated_loss = 0.0
    accumulation_step = 0
    
    # Create comprehensive logging
    log_columns = [
        'iteration', 'phase', 'loss', 'metric_loss', 'reg_loss', 
        'loss_aggregation', 'batch_size', 'effective_batch_size', 'column_bucket',
        'n_cols_sampled', 'future_window_size', 'learning_rate',
        # Add constraint step and ranges for context
        'constraint_step', 'max_weight_range', 'min_assets_range', 'max_assets_range', 'sparsity_threshold_range', 'future_window_range',
        # Add actual constraint values used for this specific iteration
        'max_weight_used', 'min_assets_used', 'max_assets_used', 'sparsity_threshold_used', 'future_window_used'
    ]
    log_columns.extend(other_metrics_list)
    log = pd.DataFrame(columns=log_columns)
    
    # Bootstrap state for column sampling
    bootstrap_sampler = _BootstrapColumnSampler(total_columns)
    
    # Training data preparation
    valid_indices = len(data_array) - (past_window_size + max(future_window_range))
    
    print(f"\n🚀 Starting curriculum training for {iterations} iterations across {len(phases)} phases...")
    
    current_iteration = 0
    
    # Execute each phase
    for phase_idx, phase in enumerate(phases):
        print(f"\n🎯 PHASE {phase_idx + 1}/{len(phases)}: Batch={phase['batch_size']}, "
              f"Column={phase['column_bucket']}, Constraint={phase['constraint_step']}")
        print(f"   Iterations: {phase['iterations']} | Effective batch size: {phase['batch_size'] * gradient_accumulation_steps}")
        
        # Reset everything for new phase
        enhanced_optimizer.zero_grad()
        accumulation_step = 0
        
        # Get phase parameters
        current_batch_size = phase['batch_size']
        column_bucket_id = phase['column_bucket']
        constraint_step = phase['constraint_step']
        
        # Get constraint ranges for this step
        max_weight_range_step = max_weight_steps[constraint_step]
        min_assets_range_step = min_assets_steps[constraint_step]
        max_assets_range_step = max_assets_steps[constraint_step]
        sparsity_range_step = sparsity_steps[constraint_step]
        
        # Get column sampling range for this bucket
        min_cols_bucket, max_cols_bucket = column_buckets[column_bucket_id]
        
        # Execute iterations for this phase
        for phase_iter in range(phase['iterations']):
            current_iteration += 1
            
            # Progressive loss aggregation
            progress = (current_iteration - 1) / (iterations - 1) if iterations > 1 else 0
            current_loss_aggregation, phase_name = _get_loss_aggregation(loss_aggregation, progress)
            
            # Sample number of columns for this batch (within bucket range)
            n_cols_to_sample = random.randint(min_cols_bucket, max_cols_bucket)
            
            # Sample specific columns using bootstrap strategy
            selected_columns = bootstrap_sampler.sample_columns(column_bucket_id, n_cols_to_sample, column_buckets)
            
            # Create uniform future window sizes for this batch
            future_window_sizes = np.random.uniform(
                future_window_range[0], 
                future_window_range[1], 
                size=current_batch_size
            ).astype(int)
            
            # Use median future window size for batch creation (simplified)
            batch_future_window = int(np.median(future_window_sizes))
            
            # Sample constraints randomly within current step ranges
            max_weight = random.uniform(*max_weight_range_step)
            min_assets = random.randint(int(min_assets_range_step[0]), int(min_assets_range_step[1]))
            max_assets = random.randint(int(max_assets_range_step[0]), int(max_assets_range_step[1]))
            sparsity_threshold = random.uniform(*sparsity_range_step)
            
            # Create batch normally, then select specific columns afterward
            # This matches the progressive function approach
            past_batch_tensor, future_batch_tensor = _create_batch(
                data_array, 
                past_window_size, 
                batch_future_window, 
                n_cols_to_sample, 
                current_batch_size, 
                valid_indices
            )
            
            # Apply column selection after batch creation (if needed for curriculum)
            # Note: The _create_batch function already handles column sampling internally
            
            # Constraint validation (same as progressive function)
            effective_max_assets = min(max_assets, n_cols_to_sample)
            effective_min_assets = max(1, min(min_assets, effective_max_assets, n_cols_to_sample))
            
            if effective_min_assets > n_cols_to_sample:
                effective_min_assets = max(1, n_cols_to_sample // 2)
            if effective_max_assets < effective_min_assets:
                effective_max_assets = max(effective_min_assets, n_cols_to_sample // 2)
            
            # Prepare model inputs (same structure as progressive function)
            raw_constraints = {
                'max_weight': max_weight,
                'min_assets': effective_min_assets,
                'max_assets': effective_max_assets,
                'sparsity_threshold': sparsity_threshold
            }
            
            # Create normalized inputs for neural network
            normalized_scalar_input = torch.tensor(
                batch_future_window / 100.0, dtype=torch.float32
            ).unsqueeze(0).repeat(current_batch_size, 1)
            
            normalized_constraint_vector = torch.tensor([
                max_weight,
                effective_min_assets / 100.0,
                effective_max_assets / 100.0,
                sparsity_threshold * 10.0
            ], dtype=torch.float32)
            
            normalized_constraint_input = normalized_constraint_vector.unsqueeze(0).repeat(current_batch_size, 1)
            
            past_batch = {
                'matrix_input': past_batch_tensor,
                'scalar_input': normalized_scalar_input,
                'constraint_input': normalized_constraint_input,
                'raw_constraints': raw_constraints
            }
            
            future_batch = {
                'returns': future_batch_tensor[0].T
            }
            
            # Gradient accumulation
            if accumulation_step == 0:
                enhanced_optimizer.zero_grad()
            
            # Update model
            loss_dict = _update_model(
                model=model, optimizer=enhanced_optimizer, 
                past_batch=past_batch, future_batch=future_batch, loss=loss,
                max_weight=raw_constraints['max_weight'], 
                min_assets=raw_constraints['min_assets'], 
                max_assets=raw_constraints['max_assets'], 
                sparsity_threshold=raw_constraints['sparsity_threshold'],
                loss_aggregation=current_loss_aggregation,
                gradient_accumulation_steps=gradient_accumulation_steps,
                accumulation_step=accumulation_step
            )
            
            # Handle gradient accumulation
            accumulated_loss += loss_dict['loss'] / gradient_accumulation_steps
            accumulation_step += 1
            
            gradient_update_occurred = False
            if accumulation_step >= gradient_accumulation_steps:
                lr_scheduler.step()
                accumulation_step = 0
                current_iteration_loss = accumulated_loss
                accumulated_loss = 0.0
                gradient_update_occurred = True
            else:
                current_iteration_loss = loss_dict['loss']
            
            # Calculate additional metrics whenever we're about to log (not just on gradient updates)
            if current_iteration % gradient_accumulation_steps == 0:
                additional_metrics = _calculate_additional_metrics(other_metrics_list, loss_dict, future_batch)
            else:
                additional_metrics = {}  # Empty dict when not logging
            
            # Learning rate schedulers
            if plateau_scheduler is not None:
                plateau_scheduler.step(current_iteration_loss)
            
            # Early stopping
            if current_iteration_loss < best_loss - early_stopping_threshold:
                best_loss = current_iteration_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered at iteration {current_iteration}")
                print(f"Best loss: {best_loss:.6f}, Current loss: {current_iteration_loss:.6f}")
                break
            
            # Get current learning rate
            current_lr = lr_scheduler.get_last_lr()[0]
            effective_batch_size = current_batch_size * gradient_accumulation_steps
            
            # Log every gradient_accumulation_steps iterations
            # For gradient_accumulation_steps=2: log on iterations 2, 4, 6, 8, etc.
            if current_iteration % gradient_accumulation_steps == 0:
                new_row_data = {
                    'iteration': [current_iteration], 
                    'phase': [f"{phase_idx + 1}"],
                    'loss': [current_iteration_loss],
                    'metric_loss': [loss_dict['metric_loss']],
                    'reg_loss': [loss_dict['reg_loss']],
                    'loss_aggregation': [current_loss_aggregation],
                    'batch_size': [current_batch_size],
                    'effective_batch_size': [effective_batch_size],
                    'column_bucket': [column_bucket_id],
                    'n_cols_sampled': [n_cols_to_sample],
                    'future_window_size': [batch_future_window],
                    'learning_rate': [current_lr],
                    # Add constraint step and ranges for context
                    'constraint_step': [constraint_step],
                    'max_weight_range': [f"{max_weight_range_step[0]:.3f}-{max_weight_range_step[1]:.3f}"],
                    'min_assets_range': [f"{int(min_assets_range_step[0])}-{int(min_assets_range_step[1])}"],
                    'max_assets_range': [f"{int(max_assets_range_step[0])}-{int(max_assets_range_step[1])}"],
                    'sparsity_threshold_range': [f"{sparsity_range_step[0]:.3f}-{sparsity_range_step[1]:.3f}"],
                    'future_window_range': [f"{future_window_range[0]}-{future_window_range[1]}"],
                    # Add actual constraint values used for this specific iteration
                    'max_weight_used': [raw_constraints['max_weight']],
                    'min_assets_used': [raw_constraints['min_assets']],
                    'max_assets_used': [raw_constraints['max_assets']],
                    'sparsity_threshold_used': [raw_constraints['sparsity_threshold']],
                    'future_window_used': [batch_future_window]
                }
                
                for metric_name in other_metrics_list:
                    new_row_data[metric_name] = [additional_metrics.get(metric_name, float('nan'))]
                
                new_row = pd.DataFrame(new_row_data)
                log = pd.concat([log, new_row], ignore_index=True)
            
                # Console output and intermittent logging (based on logged iterations)
                logged_iterations = len(log)
                if logged_iterations % log_frequency == 0:
                    print(f"Iter {current_iteration}/{iterations} | Phase {phase_idx+1} | "
                          f"Loss: {current_iteration_loss:.6f} | Cols: {n_cols_to_sample} | "
                          f"Bucket: {column_bucket_id} ({column_buckets[column_bucket_id]}) | "
                          f"MaxWt: {raw_constraints['max_weight']:.3f} | "
                          f"MinAssets: {raw_constraints['min_assets']} | "
                          f"MaxAssets: {raw_constraints['max_assets']} | "
                          f"Sparsity: {raw_constraints['sparsity_threshold']:.3f} | "
                          f"LR: {current_lr:.2e}")
                    
                    # Save log intermittently
                    log.to_csv(log_filepath, index=False)
                
                # Checkpoints (based on logged iterations)
                if logged_iterations % checkpoint_frequency == 0:
                    checkpoint_filename = f'curriculum_checkpoint_{current_iteration}.pt'
                    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
                    torch.save(model.state_dict(), checkpoint_filepath)
                    # Also save log at checkpoint
                    log.to_csv(log_filepath, index=False)
        
        # Break if early stopping triggered
        if patience_counter >= early_stopping_patience:
            break
    
    # Save final model and logs (same as progressive function)
    final_model_filename = 'final_curriculum_model.pt'
    final_model_filepath = os.path.join(checkpoint_path, final_model_filename)
    torch.save(model, final_model_filepath)
    
    final_state_dict_filename = 'final_curriculum_state_dict.pt'
    final_state_dict_filepath = os.path.join(checkpoint_path, final_state_dict_filename)
    torch.save(model.state_dict(), final_state_dict_filepath)
    
    # Save final log (using the log_filepath already defined at start)
    log.to_csv(log_filepath, index=False)
    
    end_time = datetime.now()
    print(f"\n🏁 Curriculum training completed! Total time: {end_time - start_time}")
    print(f"Final loss: {current_iteration_loss:.6f}")
    print(f"Total phases executed: {phase_idx + 1}/{len(phases)}")
    print(f"Final model saved to: {final_model_filepath}")
    print(f"Training log saved to: {log_filepath}")
    
    return model
