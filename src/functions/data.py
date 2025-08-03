#%%
'''
loop takes df, t_fixed, n_cols and batch_size as inputs.
It starts by choosing a random index. An index is used to slice the dataframe.
The index supplies the starting point for slicing rows and columns.
The row_count of the slice is set by the t_fixed value. It will be 2 * t_fixed(
because the dataframe in turn is being spliced in half).
The chosen index must thus ensure that the slice does not exceed the dataframe
bounds after slicing (ie that, a (t_fixed length * 2) length data frame is possible without NAs).
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
import random

def create_single_sample(data_array, t_fixed, n_cols, valid_indices, max_retries=10):
    """
    Create a single sample from the numpy array.
    
    Args:
        data_array: numpy array of shape (n_timesteps, n_assets)
        
    Returns:
        numpy array of shape (n_cols, 2*t_fixed)
    """
    # Choose a random starting point
    start_idx = random.randint(0, valid_indices)
    # Slice the array
    data_slice = data_array[start_idx:start_idx + (2 * t_fixed), :]
    
    # Find columns with no NaN values in this slice
    valid_cols = ~np.isnan(data_slice).any(axis=0)
    valid_col_indices = np.where(valid_cols)[0]
    
    # Ensure we always get a sample, even if we need to retry or pad
    retry_count = 0
    
    while len(valid_col_indices) < n_cols and retry_count < max_retries:
        # Try a different random starting point
        start_idx = random.randint(0, valid_indices)
        data_slice = data_array[start_idx:start_idx + (2 * t_fixed), :]
        valid_cols = ~np.isnan(data_slice).any(axis=0)
        valid_col_indices = np.where(valid_cols)[0]
        retry_count += 1
    
    # If we still don't have enough columns after retries, pad with zeros
    if len(valid_col_indices) >= n_cols:
        # Randomly sample n_cols from valid columns
        selected_cols = np.random.choice(valid_col_indices, size=n_cols, replace=False)
        col_sample = data_slice[:, selected_cols]
        return col_sample.T  # Transpose to (n_cols, t_fixed*2)
    else:
        # Create zero-padded sample to maintain batch size
        sample_array = np.zeros((n_cols, 2 * t_fixed))
        if len(valid_col_indices) > 0:
            # Fill with available data
            available_cols = min(len(valid_col_indices), n_cols)
            selected_cols = np.random.choice(valid_col_indices, size=available_cols, replace=False)
            sample_array[:available_cols, :] = data_slice[:, selected_cols].T
        return sample_array

def create_batch(data_array, t_fixed, n_cols, batch_size, valid_indices):
    """
    Create a single batch and return past and future tensors.
    
    Args:
        data_array: numpy array of shape (n_timesteps, n_assets)
    
    Returns:
        past_batch: torch.Tensor of shape (batch_size, n_cols, t_fixed)
        future_batch: torch.Tensor of shape (batch_size, n_cols, t_fixed)
    """
    batch = []
    
    for j in range(batch_size):
        sample = create_single_sample(data_array, t_fixed, n_cols, valid_indices)
        batch.append(sample)
    
    # Convert list of numpy arrays to tensor - automatically gets correct shape!
    batch_array = np.array(batch)  # Shape: (batch_size, n_cols, 2*t_fixed)
    
    # Split into past and future batches
    past_batch = torch.tensor(batch_array[:, :, :t_fixed], dtype=torch.float32)      # First t_fixed timesteps
    future_batch = torch.tensor(batch_array[:, :, t_fixed:], dtype=torch.float32)   # Next t_fixed timesteps
    
    return past_batch, future_batch

def __placeholder_func(data, t_fixed, min_n_cols, max_n_cols, min_batch_size, max_batch_size, iterations):
    """
    Progressive batch creation with curriculum learning.
    Starts with small batch_size and n_cols, gradually increases both.
    
    Args:
        data: pandas DataFrame or numpy array of shape (n_timesteps, n_assets)
        t_fixed: Fixed time window size
        min_n_cols: Starting number of columns (assets)
        max_n_cols: Final number of columns (assets)
        min_batch_size: Starting batch size
        max_batch_size: Final batch size
        iterations: Total number of iterations
    """
    # Convert DataFrame to numpy array if needed
    if hasattr(data, 'values'):  # Check if it's a DataFrame
        data_array = data.values
        print("Converted DataFrame to numpy array for faster processing")
    else:
        data_array = data
    
    valid_indices = len(data_array) - (2 * t_fixed)
    
    # Loop through fixed number of iterations with progressive difficulty
    for i in range(iterations):
        # Calculate progressive values (linear interpolation)
        progress = i / (iterations - 1) if iterations > 1 else 0  # 0 to 1
        
        # Progressive n_cols: start small, end large
        current_n_cols = int(min_n_cols + progress * (max_n_cols - min_n_cols))
        
        # Progressive batch_size: start small, end large
        current_batch_size = int(min_batch_size + progress * (max_batch_size - min_batch_size))
        
        # Use the create_batch function to get each batch
        past_batch, future_batch = create_batch(data_array, t_fixed, current_n_cols, current_batch_size, valid_indices)
        
        # Temp output example
        print(f"Iteration {i+1}/{iterations}: n_cols={current_n_cols}, batch_size={current_batch_size}, shapes=({past_batch.shape}, {future_batch.shape})")
        

#%%


#%%
#build some sample data and test the functions
example_data = np.random.randn(1000, 50)  # 1000 timesteps, 50 assets

# Test progressive training: start with 2 cols/batch_size=2, end with 10 cols/batch_size=16
for past_batch, future_batch, n_cols, batch_size in __placeholder_func(
    example_data, 
    t_fixed=100, 
    min_n_cols=2, 
    max_n_cols=10, 
    min_batch_size=2, 
    max_batch_size=16, 
    iterations=10
):
    # Your training logic here
    print(f"  → Training with {n_cols} assets, batch size {batch_size}")
    # break  # Remove this to see all iterations
# %%
