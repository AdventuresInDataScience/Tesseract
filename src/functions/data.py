#%%
'''
loop takes df, past_window_size, n_cols and batch_size as inputs.
It starts by choosing a random index. An index is used to slice the dataframe.
The index supplies the starting point for slicing rows and columns.
The row_count of the slice is set by the past_window_size value. It will be 2 * past_window_size(
because the dataframe in turn is being spliced in half).
The chosen index must thus ensure that the slice does not exceed the dataframe
bounds after slicing (ie that, a (past_window_size length * 2) length data frame is possible without NAs).
Another problem is that I then want to randomly sample n_col number of columns,
but all columns do not have complete data. Some will have
NAs because the price timeseries hasn't started yet for that stock.
A possible solution is to first trim the data back to only those columns that have
complete data before applying the column slicing.
But the problem is that this process makes only one sample. There needs to be a
way to make batch_size number of identical dimension samples. The above process risks
returning different sized dataframes, if one row index results in too few available columns,
or the same few columns might keep getting sampled in the early parts of the dataframe,
which would unbalance the samples.
'''

# Clean solution using your original approach:
import torch
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime
from model import update_model

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

def train_model(model, optimizer, data, past_window_size, future_window_size, min_n_cols = 10, 
                       max_n_cols = 100, min_batch_size = 32, max_batch_size = 256, iterations = 1000, 
                       metric ='sharpe_ratio', loss_aggregation='gmse',
                       # Constraint ranges for random sampling
                       max_weight_range=(0.1, 1.0), min_assets_range=(0, 50), 
                       max_assets_range=(5, 200), sparsity_threshold_range=(0.005, 0.05),
                       # Logging and checkpoint paths
                       log_path=None, checkpoint_path=None):
    """
    Progressive batch creation with curriculum learning and random constraint sampling.
    Starts with small batch_size and n_cols, gradually increases both.
    Randomly samples different constraints for each iteration to ensure full coverage.
    
    Args:
        data: pandas DataFrame or numpy array of shape (n_timesteps, n_assets)
        past_window_size: Number of timesteps for past window
        future_window_size: Number of timesteps for future window
        min_n_cols: Starting number of columns (assets)
        max_n_cols: Final number of columns (assets)
        min_batch_size: Starting batch size
        max_batch_size: Final batch size
        iterations: Total number of iterations
        metric: Metric to optimize ('sharpe_ratio', 'geometric_sharpe_ratio', etc.)
        loss_aggregation: Method to aggregate losses across batch
            - 'mae': Mean Absolute Error - arithmetic mean
            - 'mse': Mean Square Error - mean of squared losses  
            - 'gmae': Geometric Mean Absolute Error - log-space geometric mean
            - 'gmse': Geometric Mean Square Error (default, user's proposed method)
        max_weight_range: (min, max) range for max_weight constraint sampling
        min_assets_range: (min, max) range for min_assets constraint sampling  
        max_assets_range: (min, max) range for max_assets constraint sampling
        sparsity_threshold_range: (min, max) range for sparsity_threshold sampling
        log_path: Path to save training logs (default: repo_root/logs/)
        checkpoint_path: Path to save model checkpoints (default: repo_root/checkpoints/)
        
    Returns:
        Trained model
        
    Example:
        >>> # Train with custom paths
        >>> trained_model = train_model(
        ...     model=model, 
        ...     optimizer=optimizer, 
        ...     data=df, 
        ...     past_window_size=20, 
        ...     future_window_size=10,
        ...     metric='sharpe_ratio',
        ...     loss_aggregation='gmse',
        ...     log_path='/path/to/custom/logs',
        ...     checkpoint_path='/path/to/custom/checkpoints'
        ... )
        >>> 
        >>> # Train with default paths (repo_root/logs/ and repo_root/checkpoints/)
        >>> trained_model = train_model(
        ...     model=model, 
        ...     optimizer=optimizer, 
        ...     data=df, 
        ...     past_window_size=20, 
        ...     future_window_size=10,
        ...     metric='geometric_sharpe_ratio',
        ...     loss_aggregation='mae'
        ... )
    """
    # start time
    start_time = datetime.now()
    print("Training started at", start_time)

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
    print(f"Model checkpoints will be saved to: {checkpoint_path}")

    # Create log
    log = pd.DataFrame(columns=['iteration', 'loss'])

    # Convert DataFrame to numpy array if needed
    if hasattr(data, 'values'):  # Check if it's a DataFrame
        data_array = data.values
        print("Converted DataFrame to numpy array for faster processing")
    else:
        data_array = data
    
    valid_indices = len(data_array) - (past_window_size + future_window_size)
    
    # Loop through fixed number of iterations with progressive difficulty
    for i in range(iterations):
        # Calculate progressive values (linear interpolation)
        progress = i / (iterations - 1) if iterations > 1 else 0  # 0 to 1
        
        # Progressive n_cols: start small, end large
        current_n_cols = int(min_n_cols + progress * (max_n_cols - min_n_cols))
        
        # Progressive batch_size: start small, end large
        current_batch_size = int(min_batch_size + progress * (max_batch_size - min_batch_size))
        
        # Randomly sample constraints for this iteration to ensure full coverage
        max_weight = np.random.uniform(max_weight_range[0], max_weight_range[1])
        min_assets = np.random.randint(min_assets_range[0], min_assets_range[1] + 1)
        max_assets = np.random.randint(max_assets_range[0], max_assets_range[1] + 1)
        sparsity_threshold = np.random.uniform(sparsity_threshold_range[0], sparsity_threshold_range[1])
        
        # Use the create_batch function to get each batch
        past_batch_tensor, future_batch_tensor = create_batch(data_array, past_window_size, future_window_size, current_n_cols, current_batch_size, valid_indices)
        
        # Prepare data in the format expected by update_model
        # past_batch_tensor shape: (batch_size, n_cols, past_window_size)
        # future_batch_tensor shape: (batch_size, n_cols, future_window_size)
        
        # Create input vectors for the model:
        # 1. Scalar input: future_window_size (prediction parameter)
        # 2. Constraint vector: [max_weight, min_assets, max_assets, sparsity_threshold] (portfolio constraints)
        
        # Scalar input for prediction horizon
        scalar_input = torch.tensor(future_window_size, dtype=torch.float32).unsqueeze(0).repeat(current_batch_size, 1)  # (batch_size, 1)
        
        # Constraint vector for portfolio constraints
        # Ensure max_assets doesn't exceed current number of columns
        effective_max_assets = min(max_assets, current_n_cols)
        
        constraint_vector = torch.tensor([
            max_weight,  # Max weight (0.0-1.0, where 1.0 = unconstrained)
            min_assets / 100.0,  # Normalize min assets (0-100, where 0 = unconstrained)
            effective_max_assets / 100.0,  # Normalize max assets (0-100)
            sparsity_threshold * 100.0  # Scale sparsity threshold (0.01 -> 1.0)
        ], dtype=torch.float32)
        
        # Expand constraint vector to batch size
        constraint_input = constraint_vector.unsqueeze(0).repeat(current_batch_size, 1)  # (batch_size, 4)
        
        # Format data as dictionaries expected by update_model
        past_batch = {
            'matrix_input': past_batch_tensor,  # (batch_size, n_cols, past_window_size)
            'scalar_input': scalar_input,  # (batch_size, 1) - future_window_size
            'constraint_input': constraint_input  # (batch_size, 4) - [max_weight, min_assets, max_assets, sparsity]
        }
        
        # For future batch, we need to convert the tensor to returns format
        # The future_batch_tensor contains normalized price continuations
        # We need to convert this to a format suitable for portfolio evaluation
        future_batch = {
            'returns': future_batch_tensor[0].T  # Use first sample, transpose to (timesteps, n_assets)
        }
        
        # Temp output example
        # print(f"Iteration {i+1}/{iterations}: n_cols={current_n_cols}, batch_size={current_batch_size}, shapes=({past_batch_tensor.shape}, {future_batch_tensor.shape})")
        loss_dict = update_model(model=model, optimizer=optimizer, past_batch=past_batch, future_batch=future_batch, metric=metric,
                     max_weight=max_weight, min_assets=min_assets, max_assets=max_assets, sparsity_threshold=sparsity_threshold,
                     loss_aggregation=loss_aggregation)

        # every 50 iterations, save weights and log losses
        if (i + 1) % 50 == 0:
            print(f"Saving model at iteration {i + 1}")
            checkpoint_filename = f'model_checkpoint_{i + 1}.pt'
            checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_filepath)
            print(f"Model checkpoint saved to: {checkpoint_filepath}")
            
            # Log the loss for this checkpoint
            new_row = pd.DataFrame({'iteration': [i + 1], 'loss': [loss_dict['loss']]})
            log = pd.concat([log, new_row], ignore_index=True)

    # Save final complete model (architecture + weights)
    final_model_filename = 'final_trained_model.pt'
    final_model_filepath = os.path.join(checkpoint_path, final_model_filename)
    torch.save(model, final_model_filepath)
    print(f"Complete final model saved to: {final_model_filepath}")
    
    # Also save final state dict for compatibility
    final_state_dict_filename = 'final_model_state_dict.pt'
    final_state_dict_filepath = os.path.join(checkpoint_path, final_state_dict_filename)
    torch.save(model.state_dict(), final_state_dict_filepath)
    print(f"Final model state dict saved to: {final_state_dict_filepath}")

    # Save final log to CSV with datetime filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'training_log_{timestamp}.csv'
    log_filepath = os.path.join(log_path, log_filename)
    log.to_csv(log_filepath, index=False)
    print(f"Training log saved to: {log_filepath}")

    end_time = datetime.now()
    print("Training completed at", end_time)
    print(f"Total training time: {end_time - start_time}")
    return model

#%%