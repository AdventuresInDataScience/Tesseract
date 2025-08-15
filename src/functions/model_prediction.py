"""
Model prediction functions for the Tesseract portfolio optimization system.
Contains functions for making portfolio weight predictions and handling various input formats.
"""

import torch
import pandas as pd
import numpy as np


def predict_portfolio_weights(model, data_input, future_window_size=20, 
                             max_weight=1.0, min_assets=0, max_assets=1000, 
                             sparsity_threshold=0.01, max_assets_subset=None):
    """
    Make portfolio weight predictions with user-specified prediction horizon and constraints.
    Automatically handles pandas DataFrames, numpy arrays, and torch tensors as input.
    
    Args:
        model: Trained GPT2LikeTransformer model
        data_input: Price data in one of the following formats:
            - torch.Tensor: Shape (batch_size, n_assets, past_window_size) - used directly
            - pandas.DataFrame: Rows are timesteps, columns are assets - uses last past_window_size rows
            - numpy.ndarray: Shape (timesteps, n_assets) - uses last past_window_size rows
        future_window_size: Number of timesteps to predict for (prediction horizon)
        max_weight: Maximum weight for any single asset (0.0-1.0, default 1.0 = unconstrained)
        min_assets: Minimum number of assets to hold (default 0 = unconstrained)
        max_assets: Maximum number of assets to hold (default 1000 = unconstrained)
        sparsity_threshold: Threshold below which weights are set to 0 (0.0-1.0)
        max_assets_subset: If provided, only use the first N assets from the input data
    
    Returns:
        Portfolio weights tensor of shape (batch_size, n_assets)
    
    Example:
        >>> # Using pandas DataFrame (most common case)
        >>> df = pd.read_parquet("stock_prices.parquet")  # Rows=timesteps, Cols=assets
        >>> weights = predict_portfolio_weights(
        ...     model=trained_model,
        ...     data_input=df,  # Automatically uses last 30 rows (past_window_size)
        ...     future_window_size=30,
        ...     max_weight=0.3,
        ...     max_assets_subset=50  # Only use first 50 assets
        ... )
        >>> 
        >>> # Using numpy array
        >>> numpy_data = df.values  # Shape: (timesteps, n_assets)
        >>> weights = predict_portfolio_weights(
        ...     model=trained_model,
        ...     data_input=numpy_data,
        ...     future_window_size=5
        ... )
        >>> 
        >>> # Using pre-prepared tensor (original behavior)
        >>> tensor_data = torch.randn(1, 100, 20)  # (batch_size, n_assets, past_window_size)
        >>> weights = predict_portfolio_weights(
        ...     model=trained_model,
        ...     data_input=tensor_data,
        ...     future_window_size=10
        ... )
    """
    model.eval()
    
    with torch.no_grad():
        # Handle different input types and convert to required tensor format
        if isinstance(data_input, torch.Tensor):
            # Already a tensor - use directly
            matrix_input = data_input
            if len(matrix_input.shape) == 2:
                # Add batch dimension if missing
                matrix_input = matrix_input.unsqueeze(0)
            
        elif isinstance(data_input, pd.DataFrame):
            # Pandas DataFrame - convert to tensor
            # Handle data cleaning
            df_clean = data_input.copy()
            
            # Convert object columns to numeric, coercing errors to NaN
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Drop columns that are all NaN
            df_clean = df_clean.dropna(axis=1, how='all')
            
            # Forward fill and backward fill to handle remaining NaN values
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
            
            # If still have NaN values, drop those columns
            df_clean = df_clean.dropna(axis=1, how='any')
            
            # Apply asset subset if requested
            if max_assets_subset is not None:
                df_clean = df_clean.iloc[:, :max_assets_subset]
            
            # Get recent data for prediction
            past_window_size = model.past_window_size
            recent_data = df_clean.iloc[-past_window_size:, :].values  # (past_window_size, n_assets)
            
            # Convert to tensor format: (batch_size, n_assets, past_window_size)
            matrix_input = torch.tensor(recent_data.T, dtype=torch.float32).unsqueeze(0)  # (1, n_assets, past_window_size)
            
        elif isinstance(data_input, np.ndarray):
            # Numpy array - convert to tensor
            # Handle data cleaning
            data_array = data_input.copy()
            
            # Convert to DataFrame for cleaning, then back to numpy
            df_temp = pd.DataFrame(data_array)
            
            # Convert object columns to numeric, coercing errors to NaN
            for col in df_temp.columns:
                if df_temp[col].dtype == 'object':
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
            
            # Clean the array similar to DataFrame processing
            df_temp = df_temp.dropna(axis=1, how='all')
            df_temp = df_temp.fillna(method='ffill').fillna(method='bfill')
            df_temp = df_temp.dropna(axis=1, how='any')
            
            # Apply asset subset if requested
            if max_assets_subset is not None:
                df_temp = df_temp.iloc[:, :max_assets_subset]
            
            # Get recent data for prediction
            past_window_size = model.past_window_size
            recent_data = df_temp.iloc[-past_window_size:, :].values  # (past_window_size, n_assets)
            
            # Convert to tensor format: (batch_size, n_assets, past_window_size)
            matrix_input = torch.tensor(recent_data.T, dtype=torch.float32).unsqueeze(0)  # (1, n_assets, past_window_size)
            
        else:
            raise ValueError(f"Unsupported data input type: {type(data_input)}. Expected torch.Tensor, pandas.DataFrame, or numpy.ndarray")
        
        # Validate tensor shape
        if len(matrix_input.shape) != 3:
            raise ValueError(f"Invalid tensor shape: {matrix_input.shape}. Expected (batch_size, n_assets, past_window_size)")
        
        batch_size, n_assets, _ = matrix_input.shape
        
        # CRITICAL: Validate and adjust constraints for prediction
        # Ensure all constraints are logically consistent with available assets
        
        # 1. Ensure max_assets doesn't exceed available assets
        effective_max_assets = min(max_assets, n_assets)
        
        # 2. Ensure min_assets doesn't exceed max_assets or available assets
        effective_min_assets = min(min_assets, effective_max_assets, n_assets)
        
        # 3. Ensure min_assets is at least 1 if specified
        if min_assets > 0:
            effective_min_assets = max(1, effective_min_assets)
        
        # 4. Validation warnings for user
        if min_assets > n_assets:
            print(f"Warning: min_assets ({min_assets}) exceeds available assets ({n_assets}). Using {effective_min_assets}")
        
        if max_assets < min_assets:
            print(f"Warning: max_assets ({max_assets}) < min_assets ({min_assets}). Adjusted to {effective_max_assets}")
        
        # Prepare scalar input (future window size) - normalized for neural network
        normalized_scalar_input = torch.tensor(future_window_size / 100.0, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        
        # Prepare constraint input - normalized for neural network
        normalized_constraint_vector = torch.tensor([
            max_weight,  # Max weight already in [0, 1] range
            effective_min_assets / 100.0,  # Normalize min assets to [0, 1] range
            effective_max_assets / 100.0,  # Normalize max assets to [0, 1] range
            sparsity_threshold * 10.0  # Scale sparsity threshold to reasonable range
        ], dtype=torch.float32)
        
        # Expand constraint vector to batch size
        normalized_constraint_input = normalized_constraint_vector.unsqueeze(0).repeat(batch_size, 1)
        
        # Get model prediction
        weights = model(matrix_input, normalized_scalar_input, normalized_constraint_input)
        
        return weights
