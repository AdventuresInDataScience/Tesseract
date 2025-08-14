import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np

#------------ Activation Function Factory -----------
def get_activation_function(activation: Union[str, nn.Module]) -> nn.Module:
    """
    Get activation function by name or return the module if already provided.
    
    Args:
        activation: Either a string name or a PyTorch activation module
        
    Returns:
        PyTorch activation module
        
    Available activations:
        - 'relu': Standard ReLU
        - 'relu6': ReLU clamped to [0, 6] (CPU-optimized)
        - 'leaky_relu': LeakyReLU with negative_slope=0.01
        - 'elu': Exponential Linear Unit
        - 'selu': Scaled Exponential Linear Unit
        - 'gelu': Gaussian Error Linear Unit (Transformer standard)
        - 'swish' or 'silu': Swish/SiLU activation
        - 'mish': Mish activation (modern, smooth)
        - 'hard_swish': Hard Swish (mobile-optimized)
        - 'hard_mish': Hard Mish (computational proxy)
        - 'hard_gelu': Hard GELU (computational proxy)
        - 'prelu': Parametric ReLU (learnable)
        - 'glu': Gated Linear Unit
        - 'tanh': Hyperbolic tangent
        - 'sigmoid': Sigmoid activation
    
    Example:
        >>> act = get_activation_function('mish')
        >>> act = get_activation_function('hard_swish')
        >>> act = get_activation_function(torch.nn.ReLU())  # Pass module directly
    """
    if isinstance(activation, nn.Module):
        return activation
    
    activation = activation.lower().strip()
    
    activation_map = {
        # Standard activations
        'relu': nn.ReLU(),
        'relu6': nn.ReLU6(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        
        # Transformer standard
        'gelu': nn.GELU(),
        
        # Modern smooth activations
        'swish': nn.SiLU(),  # Swish is same as SiLU
        'silu': nn.SiLU(),
        'mish': nn.Mish(),
        
        # Hard/computational proxy versions (faster)
        'hard_swish': nn.Hardswish(),
        'hard_mish': nn.Mish(),  # Using Mish as proxy since PyTorch doesn't have native HardMish
        'hard_gelu': nn.GELU(approximate='tanh'),  # Tanh approximation of GELU
        
        # Parametric and gated
        'prelu': nn.PReLU(),
        'glu': nn.GLU(),
        
        # Classical
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    
    if activation not in activation_map:
        available = ', '.join(sorted(activation_map.keys()))
        raise ValueError(f"Unknown activation '{activation}'. Available: {available}")
    
    return activation_map[activation]

#------------ Model Components -----------
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with optional causal masking."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask if enabled (GPT-2 style)
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attn_output)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, causal: bool = True, activation='relu6'):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Get activation function from factory
        activation_fn = get_activation_function(activation)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_fn,  # Configurable activation function
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class IntermediateMLP(nn.Module):
    """Intermediate MLP layer between transformer blocks."""
    
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1, activation='relu6'):
        super().__init__()
        
        # Get activation function from factory
        activation_fn = get_activation_function(activation)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            activation_fn,  # Configurable activation function
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.mlp(x))


class GPT2LikeTransformer(nn.Module):
    """
    GPT-2-like transformer model with variable input size and scalar skip connections.
    
    Architecture:
    - Input: n x t matrix + 1 scalar value
    - Multiple transformer blocks with intermediate MLP layers
    - Causal self-attention (GPT-2 style) - no look-forward bias
    - Scalar skip connections between transformer blocks
    - Output: n-dimensional vector through softmax
    """
    
    def __init__(
        self,
        past_window_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_transformer_blocks: int = 6,
        d_ff: int = 2048,
        d_mlp_hidden: int = 2048,
        dropout: float = 0.1,
        max_n: int = 1000,
        causal: bool = True,
        activation='relu6'
    ):
        super().__init__()
        
        self.past_window_size = past_window_size
        self.d_model = d_model
        self.n_layers = n_transformer_blocks
        self.causal = causal
        self.activation = activation
        
        # Input projection for the matrix input (n x t -> n x d_model)
        self.input_projection = nn.Linear(past_window_size, d_model)
        
        # Scalar embedding and projection - handle future window size
        self.scalar_embedding = nn.Linear(1, d_model)  # Future window size input
        
        # Constraint embedding and projection - handle portfolio constraints  
        self.constraint_embedding = nn.Linear(4, d_model)  # Portfolio constraint vector
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_n + 2)  # +2 for scalar and constraint tokens
        
        # Transformer blocks and intermediate MLPs
        self.transformer_blocks = nn.ModuleList()
        self.intermediate_mlps = nn.ModuleList()
        
        for _ in range(n_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(d_model, n_heads, d_ff, dropout, causal, activation)
            )
            self.intermediate_mlps.append(
                IntermediateMLP(d_model, d_mlp_hidden, dropout, activation)
            )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection (only for the matrix part, not scalar)
        self.output_projection = nn.Linear(d_model, 1)
        
        # Constraint parameters (will be set via forward pass)
        self.max_weight = None
        self.min_assets = None
        self.max_assets = None
        self.sparsity_threshold = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def constrained_softmax(self, logits: torch.Tensor, max_weight: Optional[float] = None, 
                           min_assets: Optional[int] = None, max_assets: Optional[int] = None,
                           sparsity_threshold: float = 0.01) -> torch.Tensor:
        """
        Apply constrained softmax with portfolio constraints.
        
        Args:
            logits: Raw logits from the model
            max_weight: Maximum weight for any single asset (e.g., 0.5 for 50%)
            min_assets: Minimum number of assets to hold
            max_assets: Maximum number of assets to hold
            sparsity_threshold: Threshold below which weights are set to 0
        
        Returns:
            Constrained weight vector that sums to 1
        """
        batch_size, n_assets = logits.shape
        
        # Start with standard softmax
        weights = F.softmax(logits, dim=-1)
        
        # Apply sparsity threshold - set small weights to 0
        if sparsity_threshold > 0:
            mask = weights >= sparsity_threshold
            weights = weights * mask.float()
            # Renormalize
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply maximum weight constraint
        if max_weight is not None:
            weights = torch.clamp(weights, max=max_weight)
            # Renormalize
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply asset count constraints (needs to be non-inplace)
        if min_assets is not None or max_assets is not None:
            # Process each sample in the batch
            new_weights = []
            for i in range(batch_size):
                w = weights[i].clone()  # Clone to avoid in-place operations
                non_zero_count = (w > 1e-8).sum().item()
                
                # If too few assets, select top additional ones
                if min_assets is not None and non_zero_count < min_assets:
                    # Safety check: ensure min_assets doesn't exceed available assets
                    effective_min_assets = min(min_assets, n_assets)
                    needed = effective_min_assets - non_zero_count
                    
                    if needed > 0:  # Only proceed if we actually need more assets
                        # Find indices of zero weights
                        zero_indices = (w <= 1e-8).nonzero().squeeze(-1)
                        
                        # Make sure we don't try to select more assets than available
                        if len(zero_indices) >= needed:
                            # Select top 'needed' assets from zero weights based on original logits
                            original_scores = logits[i][zero_indices]
                            _, top_indices = torch.topk(original_scores, needed)
                            selected_indices = zero_indices[top_indices]
                            # Give small positive weights (create new tensor instead of in-place)
                            values = torch.full((len(selected_indices),), 0.01, device=w.device)
                            w = w.scatter(0, selected_indices, values)
                        else:
                            # If we don't have enough zero weights, use all available zero weights
                            if len(zero_indices) > 0:
                                values = torch.full((len(zero_indices),), 0.01, device=w.device)
                                w = w.scatter(0, zero_indices, values)
                
                # If too many assets, keep only top ones
                elif max_assets is not None and non_zero_count > max_assets:
                    # Safety check: ensure max_assets is reasonable
                    effective_max_assets = min(max_assets, n_assets)
                    effective_max_assets = max(1, effective_max_assets)  # At least 1 asset
                    
                    _, top_indices = torch.topk(w, effective_max_assets)
                    new_w = torch.zeros_like(w)
                    new_w = new_w.scatter(0, top_indices, w[top_indices])
                    w = new_w
                
                # Renormalize this sample
                w = w / (w.sum() + 1e-8)
                new_weights.append(w)
            
            # Stack back into batch tensor
            weights = torch.stack(new_weights, dim=0)
        
        return weights

    def forward(self, matrix_input: torch.Tensor, scalar_input: torch.Tensor, constraint_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with prediction horizon and constraints.
        
        Args:
            matrix_input: Tensor of shape (batch_size, n, past_window_size)
            scalar_input: Tensor of shape (batch_size, 1) containing future_window_size
            constraint_input: Tensor of shape (batch_size, 4) containing [max_weight, min_assets, max_assets, sparsity]
        
        Returns:
            output: Tensor of shape (batch_size, n) - constrained portfolio weights
        """
        batch_size, n, t = matrix_input.shape
        assert t == self.past_window_size, f"Expected t={self.past_window_size}, got t={t}"
        
        # Extract constraints from constraint input
        max_weight = constraint_input[:, 0]  # Max weight constraint (1.0 = unconstrained)
        min_assets = (constraint_input[:, 1] * 100.0).long()  # Denormalize and convert to int
        max_assets = (constraint_input[:, 2] * 100.0).long()  # Denormalize and convert to int  
        sparsity_threshold = constraint_input[:, 3] / 100.0  # Denormalize sparsity threshold
        
        # Project matrix input to d_model
        x = self.input_projection(matrix_input)  # (batch_size, n, d_model)
        
        # Embed scalar (future window) and constraint inputs
        scalar_emb = self.scalar_embedding(scalar_input).unsqueeze(1)  # (batch_size, 1, d_model)
        constraint_emb = self.constraint_embedding(constraint_input).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Concatenate all embeddings: matrix + scalar + constraint
        x = torch.cat([x, scalar_emb, constraint_emb], dim=1)  # (batch_size, n+2, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Store scalar and constraint embeddings for skip connections
        combined_context = scalar_emb.squeeze(1) + constraint_emb.squeeze(1)  # (batch_size, d_model)
        
        # Pass through transformer blocks with intermediate MLPs
        for i, (transformer_block, intermediate_mlp) in enumerate(
            zip(self.transformer_blocks, self.intermediate_mlps)
        ):
            # Transformer block
            x = transformer_block(x)
            
            # Add combined context skip connection to matrix positions only
            x[:, :-2, :] = x[:, :-2, :] + combined_context.unsqueeze(1)
            
            # Intermediate MLP
            x = intermediate_mlp(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Extract only the matrix part (exclude scalar and constraint positions)
        matrix_output = x[:, :-2, :]  # (batch_size, n, d_model)
        
        # Project to single values (logits, not probabilities yet)
        logits = self.output_projection(matrix_output).squeeze(-1)  # (batch_size, n)
        
        # Apply constrained softmax using extracted constraints
        # Use the first sample's constraints (assuming homogeneous batches)
        # Convert to standard constraint format: use large numbers for "unconstrained"
        effective_max_weight = max_weight[0].item() if max_weight[0].item() < 1.0 else None
        effective_min_assets = min_assets[0].item() if min_assets[0].item() > 0 else None
        effective_max_assets = max_assets[0].item() if max_assets[0].item() < n else None
        
        output = self.constrained_softmax(
            logits, 
            max_weight=effective_max_weight,
            min_assets=effective_min_assets,
            max_assets=effective_max_assets,
            sparsity_threshold=sparsity_threshold[0].item()
        )
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#------------ Model Building Function -----------
def build_transformer_model(
    past_window_size: int,
    d_model: int = 512,
    n_heads: int = 8,
    n_transformer_blocks: int = 6,
    d_ff: int = 2048,
    d_mlp_hidden: int = 2048,
    dropout: float = 0.1,
    max_n: int = 1000,
    causal: bool = True,
    activation='relu6'
) -> GPT2LikeTransformer:
    """
    Build a GPT-2-like transformer model with specified architecture and activation function.
    
    Args:
        past_window_size: Fixed dimension t for input matrices (n x t)
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        n_transformer_blocks: Number of transformer blocks
        d_ff: Feed-forward network hidden dimension
        d_mlp_hidden: Intermediate MLP hidden dimension
        dropout: Dropout probability
        max_n: Maximum expected value of n (for positional encoding)
        causal: Whether to use causal (GPT-2) attention masking
        activation: Activation function name or PyTorch module. Options:
                   CPU-Optimized:
                   - 'relu6' (default): Fast, stable, bounded [0,6]
                   - 'relu': Classic ReLU
                   - 'leaky_relu': Prevents dead neurons
                   
                   GPU/Modern:
                   - 'gelu': Transformer standard
                   - 'swish'/'silu': Smooth, modern
                   - 'mish': Very smooth, modern
                   
                   Mobile/Edge:
                   - 'hard_swish': Mobile-optimized Swish
                   - 'hard_mish': Computational proxy for Mish
                   - 'hard_gelu': Fast GELU approximation
                   
                   Advanced:
                   - 'prelu': Learnable parameters
                   - 'elu', 'selu': Exponential variants
                   - 'glu': Gated Linear Unit
    
    Returns:
        GPT2LikeTransformer model ready for training
    
    Example:
        >>> # CPU-optimized (default)
        >>> model = build_transformer_model(past_window_size=65)
        >>> 
        >>> # GPU with modern activation
        >>> model = build_transformer_model(past_window_size=65, activation='mish')
        >>> 
        >>> # Mobile-optimized
        >>> model = build_transformer_model(past_window_size=65, activation='hard_swish')
        >>> 
        >>> # Classic transformer
        >>> model = build_transformer_model(past_window_size=65, activation='gelu')
        >>> 
        >>> # Still supports PyTorch modules directly
        >>> model = build_transformer_model(past_window_size=65, activation=torch.nn.ReLU())
    """
    model = GPT2LikeTransformer(
        past_window_size=past_window_size,
        d_model=d_model,
        n_heads=n_heads,
        n_transformer_blocks=n_transformer_blocks,
        d_ff=d_ff,
        d_mlp_hidden=d_mlp_hidden,
        dropout=dropout,
        max_n=max_n,
        causal=causal,
        activation=activation
    )
    
    # Get activation name for display
    if isinstance(activation, str):
        activation_name = activation.upper()
    else:
        activation_name = activation.__class__.__name__
    
    print(f"Model created with {model.get_num_parameters():,} parameters using {activation_name} activation")
    return model


def create_adam_optimizer(model, lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8):
    """
    Create Adam optimizer with good defaults for transformer training.
    
    Args:
        model: The model to optimize
        lr: Learning rate (default 1e-4 - good for transformers)
        weight_decay: L2 regularization weight (default 1e-5)
        betas: Adam beta parameters (default (0.9, 0.999))
        eps: Adam epsilon for numerical stability (default 1e-8)
    
    Returns:
        torch.optim.Adam optimizer ready for training
    
    Example:
        >>> model = build_transformer_model(past_window_size=10)
        >>> optimizer = create_adam_optimizer(model, lr=2e-4)
        >>> # Use in training loop
        >>> result = update_model(model, optimizer, past_batch, future_batch, 'sharpe_ratio')
    """
    return torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )


#------------ Model Saving and Loading Functions -----------
def save_model(model, filepath, model_config):
    """
    Save model using state_dict and config (safe method that avoids PyTorch security issues).
    
    Args:
        model: The trained model to save
        filepath: Path where to save the model (should end with .pt)
        model_config: Dictionary with model configuration parameters (REQUIRED)
    
    Example:
        >>> config = {
        ...     'past_window_size': 30,
        ...     'd_model': 256,
        ...     'n_heads': 8,
        ...     'n_transformer_blocks': 6,
        ...     'max_n': 100
        ... }
        >>> save_model(model, 'my_model.pt', config)
    """
    from datetime import datetime
    
    if not model_config:
        raise ValueError("model_config is required for saving. Include at least: past_window_size, d_model, n_heads, n_transformer_blocks, max_n")
    
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'save_timestamp': datetime.now().isoformat()
    }
    
    torch.save(save_data, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath, device='cpu'):
    """
    Load model by reconstructing from config and loading weights.
    
    Args:
        filepath: Path to the saved model file
        device: Device to load the model on ('cpu', 'cuda', etc.)
    
    Returns:
        tuple: (model, model_config) - loaded model and its configuration
    
    Example:
        >>> model, config = load_model('my_model.pt')
        >>> weights = predict_portfolio_weights(model, data, future_window_size=5)
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    
    model_config = checkpoint['model_config']
    
    # Reconstruct the model from config
    model = build_transformer_model(
        past_window_size=model_config['past_window_size'],
        d_model=model_config.get('d_model', 256),
        n_heads=model_config.get('n_heads', 8),
        n_transformer_blocks=model_config.get('n_transformer_blocks', 6),
        max_n=model_config.get('max_n', 100)
    )
    
    # Load the weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {filepath}")
    print(f"Save timestamp: {checkpoint.get('save_timestamp', 'Unknown')}")
    print(f"Model configuration: {model_config}")
    
    return model, model_config


def load_model_from_checkpoint(model_architecture, checkpoint_path, device='cpu'):
    """
    Load model weights from a checkpoint into an existing model architecture.
    Use this when you want to recreate the model architecture manually.
    
    Args:
        model_architecture: The model architecture (created with build_transformer_model)
        checkpoint_path: Path to the checkpoint file (state_dict)
        device: Device to load the model on
    
    Returns:
        model: Model with loaded weights
    
    Example:
        >>> # Recreate the same architecture
        >>> model = build_transformer_model(past_window_size=30, d_model=256)
        >>> # Load trained weights
        >>> model = load_model_from_checkpoint(model, 'model_checkpoint_200.pt')
        >>> weights = predict_portfolio_weights(model, data, future_window_size=5)
    """
    try:
        # Try loading as weights_only=True first (safer for state_dict)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Warning: weights_only=True failed, trying weights_only=False: {e}")
        # Fallback to weights_only=False
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_architecture.load_state_dict(state_dict)
    model_architecture = model_architecture.to(device)
    model_architecture.eval()
    
    print(f"Model weights loaded from checkpoint: {checkpoint_path}")
    return model_architecture


def save_model_config(model, config_path):
    """
    Save model configuration parameters to a JSON file.
    
    Args:
        model: The model to extract configuration from
        config_path: Path to save the configuration (should end with .json)
    
    Example:
        >>> save_model_config(model, 'model_config.json')
    """
    import json
    from datetime import datetime
    
    config = {
        'past_window_size': getattr(model, 'past_window_size', None),
        'd_model': getattr(model, 'd_model', None),
        'n_layers': getattr(model, 'n_layers', None),
        'causal': getattr(model, 'causal', None),
        'model_class': model.__class__.__name__,
        'total_parameters': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else None,
        'save_timestamp': datetime.now().isoformat()
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model configuration saved to: {config_path}")


def load_model_config(config_path):
    """
    Load model configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        dict: Model configuration parameters
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Model configuration loaded from: {config_path}")
    return config


#--------------Time series from Portfolio and weights Functions--------------
def create_portfolio_time_series(stocks_matrix, weights_vector):
    """
    Create portfolio time series from stock returns and weights.
    
    Args:
        stocks_matrix: numpy array of shape (timesteps, n_stocks) - from pandas_df.values
        weights_vector: torch tensor of shape (n_stocks,) - model output weights
    
    Returns:
        portfolio_returns: torch tensor of shape (timesteps,) - portfolio returns over time
    
    Example:
        >>> # From pandas DataFrame
        >>> stocks_df = pd.DataFrame(np.random.randn(100, 5))  # 100 timesteps, 5 stocks
        >>> stocks_matrix = stocks_df.values  # Convert to numpy
        >>> weights_vector = torch.softmax(torch.randn(5), dim=0)  # Model output (sums to 1)
        >>> portfolio = create_portfolio_time_series(stocks_matrix, weights_vector)
        >>> portfolio.shape  # torch.Size([100])
    """
    # Convert numpy stocks_matrix to torch tensor if needed
    if not isinstance(stocks_matrix, torch.Tensor):
        stocks_matrix = torch.from_numpy(stocks_matrix).float()
    
    # Ensure weights_vector is torch tensor (should already be from model output)
    if not isinstance(weights_vector, torch.Tensor):
        weights_vector = torch.tensor(weights_vector, dtype=torch.float32)
    
    # Matrix multiply: (timesteps, n_stocks) @ (n_stocks,) = (timesteps,)
    portfolio_returns = torch.matmul(stocks_matrix, weights_vector)
    
    return portfolio_returns

#--------------Objective functions/metrics--------------
def sharpe_ratio(portfolio_price_timeseries, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio from a normalized price time series using arithmetic mean.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: risk-free rate (default 0.0)
    
    Returns:
        sharpe_ratio: torch scalar
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0)
    # This brings us back to the original series length
    returns = returns[1:]
    
    # Calculate arithmetic mean return
    mean_return = returns.mean()
    
    # Calculate standard deviation of returns
    std_returns = returns.std()
    
    # Calculate Sharpe ratio (negative for PyTorch minimization - higher Sharpe is better)
    return -((mean_return - risk_free_rate) / std_returns)

def geometric_sharpe_ratio(portfolio_price_timeseries, risk_free_rate=0.0):
    """
    Calculate geometric Sharpe ratio from a normalized price time series using geometric mean.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: risk-free rate (default 0.0)
    
    Returns:
        geometric_sharpe_ratio: torch scalar
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0)
    # This brings us back to the original series length
    returns = returns[1:]
    
    # Calculate geometric mean return: (1 + r1) * (1 + r2) * ... * (1 + rn))^(1/n) - 1
    geometric_mean_return = torch.pow(torch.prod(1 + returns), 1.0 / len(returns)) - 1
    
    # Calculate standard deviation of returns
    std_returns = returns.std()
    
    # Calculate geometric Sharpe ratio (negative for PyTorch minimization - higher Sharpe is better)
    return -((geometric_mean_return - risk_free_rate) / std_returns)

def max_drawdown(portfolio_price_timeseries):
    """
    Calculate maximum drawdown from a normalized price time series.
    Maximum drawdown is the largest peak-to-trough decline in the portfolio value.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
    
    Returns:
        max_drawdown: torch scalar representing the maximum drawdown as a percentage
    """
    # Calculate running maximum (peaks) directly from the normalized series
    running_max = torch.cummax(portfolio_price_timeseries, dim=0).values
    
    # Calculate percentage drawdowns: (peak - current_value) / peak
    drawdowns = (running_max - portfolio_price_timeseries) / running_max
    
    # Return the maximum drawdown (positive value - higher drawdown is worse, so no sign change needed)
    return torch.max(drawdowns)

def sortino_ratio(portfolio_price_timeseries, risk_free_rate=0.0, target_return=0.0):
    """
    Calculate Sortino ratio from a normalized price time series using arithmetic mean.
    Only considers downside volatility (negative returns below target).
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: risk-free rate (default 0.0)
        target_return: target return threshold (default 0.0)
    
    Returns:
        sortino_ratio: torch scalar
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0)
    # This brings us back to the original series length
    returns = returns[1:]
    
    # Calculate arithmetic mean return
    mean_return = returns.mean()
    
    # Calculate downside deviation (only negative returns below target)
    downside_returns = torch.minimum(returns - target_return, torch.tensor(0.0))
    downside_deviation = torch.sqrt(torch.mean(downside_returns ** 2))
    
    # Calculate Sortino ratio (negative for PyTorch minimization - higher Sortino is better)
    return -((mean_return - risk_free_rate) / downside_deviation)

def geometric_sortino_ratio(portfolio_price_timeseries, risk_free_rate=0.0, target_return=0.0):
    """
    Calculate geometric Sortino ratio from a normalized price time series using geometric mean.
    Only considers downside volatility (negative returns below target).
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: risk-free rate (default 0.0)
        target_return: target return threshold (default 0.0)
    
    Returns:
        geometric_sortino_ratio: torch scalar
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0)
    # This brings us back to the original series length
    returns = returns[1:]
    
    # Calculate geometric mean return: (1 + r1) * (1 + r2) * ... * (1 + rn))^(1/n) - 1
    geometric_mean_return = torch.pow(torch.prod(1 + returns), 1.0 / len(returns)) - 1
    
    # Calculate downside deviation (only negative returns below target)
    downside_returns = torch.minimum(returns - target_return, torch.tensor(0.0))
    downside_deviation = torch.sqrt(torch.mean(downside_returns ** 2))
    
    # Calculate geometric Sortino ratio (negative for PyTorch minimization - higher Sortino is better)
    return -((geometric_mean_return - risk_free_rate) / downside_deviation)

def expected_return(portfolio_price_timeseries):
    """
    Calculate the expected return, which is the final value divided by the initial value.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
    
    Returns:
        expected_return: torch scalar representing (final_value / initial_value)
    """
    # The portfolio starts at 1.0 (normalized) so we return the final value
    # (negative for PyTorch minimization - higher expected return is better)
    return -portfolio_price_timeseries[-1]

def carmdd(portfolio_price_timeseries):
    """
    Calculate the Calmar ratio from a normalized price time series.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
    
    Returns:
        carmdd: torch scalar representing the Calmar ratio
    """
    # Calculate maximum drawdown
    max_drawdown = torch.max(torch.cummax(portfolio_price_timeseries, dim=0).values - portfolio_price_timeseries)
    # Calculate expected return
    expected_return = portfolio_price_timeseries[-1]
    # Calculate Calmar ratio (negative for PyTorch minimization - higher Calmar is better)
    return -(expected_return / max_drawdown) if max_drawdown != 0 else torch.tensor(0.0)

def omega_ratio(portfolio_price_timeseries, risk_free_rate=0.0, target_return=0.0):
    """
    Calculate Omega ratio from a normalized price time series.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: risk-free rate (default 0.0)
        target_return: target return threshold (default 0.0)
    
    Returns:
        omega_ratio: torch scalar
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0 and is thus an NA)
    # This brings us back to the original series length
    returns = returns[1:]
    
    # Calculate upside and downside returns
    upside_returns = returns[returns > target_return]
    downside_returns = returns[returns < target_return]
    
    # Calculate Omega ratio (negative for PyTorch minimization - higher Omega is better)
    return -(upside_returns.mean() / -downside_returns.mean()) if downside_returns.numel() > 0 else torch.tensor(0.0)

def jensen_alpha(portfolio_price_timeseries, market_price_timeseries, risk_free_rate=0.0):
    """
    Calculate Jensen's alpha from normalized price time series.

    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        market_price_timeseries: torch tensor of normalized market prices (continuation from 1.0)
        risk_free_rate: risk-free rate (default 0.0)

    Returns:
        jensen_alpha: torch scalar representing Jensen's alpha
    """
    # Add 1.0 at the start for both series to represent the normalized reference point
    portfolio_full = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                               portfolio_price_timeseries])
    market_full = torch.cat([torch.tensor([1.0], device=market_price_timeseries.device), 
                            market_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    portfolio_returns = (portfolio_full[1:] / portfolio_full[:-1]) - 1
    market_returns = (market_full[1:] / market_full[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0 and is thus an NA)
    # This brings us back to the original series length
    portfolio_returns = portfolio_returns[1:]
    market_returns = market_returns[1:]
    
    # Calculate portfolio mean return
    portfolio_mean_return = portfolio_returns.mean()
    market_mean_return = market_returns.mean()
    
    # Calculate beta (portfolio sensitivity to market)
    covariance = torch.mean((portfolio_returns - portfolio_mean_return) * (market_returns - market_mean_return))
    market_variance = torch.var(market_returns)
    beta = covariance / market_variance if market_variance != 0 else torch.tensor(0.0)
    
    # Calculate Jensen's alpha (negative for PyTorch minimization - higher alpha is better)
    return -((portfolio_mean_return - risk_free_rate) - beta * (market_mean_return - risk_free_rate))

def treynor_ratio(portfolio_price_timeseries, market_price_timeseries, risk_free_rate=0.0):
    """
    Calculate Treynor ratio from normalized price time series.

    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        market_price_timeseries: torch tensor of normalized market prices (continuation from 1.0)
        risk_free_rate: risk-free rate (default 0.0)

    Returns:
        treynor_ratio: torch scalar representing the Treynor ratio
    """
    # Add 1.0 at the start for both series to represent the normalized reference point
    portfolio_full = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                               portfolio_price_timeseries])
    market_full = torch.cat([torch.tensor([1.0], device=market_price_timeseries.device), 
                            market_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    portfolio_returns = (portfolio_full[1:] / portfolio_full[:-1]) - 1
    market_returns = (market_full[1:] / market_full[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0 and is thus an NA)
    # This brings us back to the original series length
    portfolio_returns = portfolio_returns[1:]
    market_returns = market_returns[1:]
    
    # Calculate portfolio mean return
    portfolio_mean_return = portfolio_returns.mean()
    market_mean_return = market_returns.mean()
    
    # Calculate beta (portfolio sensitivity to market)
    covariance = torch.mean((portfolio_returns - portfolio_mean_return) * (market_returns - market_mean_return))
    market_variance = torch.var(market_returns)
    beta = covariance / market_variance if market_variance != 0 else torch.tensor(0.0)
    
    # Calculate Treynor ratio (negative for PyTorch minimization - higher Treynor is better)
    return -((portfolio_mean_return - risk_free_rate) / beta) if beta != 0 else torch.tensor(0.0)

def ulcer_index(portfolio_price_timeseries):
    """
    Calculate Ulcer Index from a normalized price time series.
    Ulcer Index measures downside risk by calculating the RMS of percentage drawdowns.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
    
    Returns:
        ulcer_index: torch scalar representing the Ulcer Index
    """
    # Add 1.0 at the start to represent the normalized reference point for proper drawdown calculation
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate running maximum (peaks) for drawdown calculation
    running_max = torch.cummax(full_series, dim=0).values
    
    # Calculate percentage drawdowns: (peak - current_value) / peak
    drawdowns = (running_max - full_series) / running_max
    
    # Remove the first drawdown (which is always 0 from the reference point)
    drawdowns = drawdowns[1:]
    
    # Calculate Ulcer Index as RMS of drawdowns (positive value - higher Ulcer Index is worse, so no sign change needed)
    return torch.sqrt(torch.mean(drawdowns ** 2))

def k_ratio(portfolio_price_timeseries):
    """
    Calculate K-ratio from a normalized price time series.
    K-ratio measures the consistency of returns by analyzing the slope and R-squared 
    of a regression line fitted to the logarithmic cumulative returns.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
    
    Returns:
        k_ratio: torch scalar representing the K-ratio
    """
    # Calculate logarithmic cumulative returns directly from normalized prices
    log_cumulative_returns = torch.log(portfolio_price_timeseries)
    
    # Create time index starting from 1
    n = len(log_cumulative_returns)
    time_index = torch.arange(1, n + 1, dtype=torch.float32, device=portfolio_price_timeseries.device)
    
    # Perform linear regression: log_cumulative_returns = slope * time_index + intercept
    mean_time = time_index.mean()
    mean_log_returns = log_cumulative_returns.mean()
    
    # Calculate slope using least squares
    numerator = torch.sum((time_index - mean_time) * (log_cumulative_returns - mean_log_returns))
    denominator = torch.sum((time_index - mean_time) ** 2)
    slope = numerator / denominator if denominator != 0 else torch.tensor(0.0)
    
    # Calculate R-squared
    y_pred = slope * time_index + (mean_log_returns - slope * mean_time)
    ss_res = torch.sum((log_cumulative_returns - y_pred) ** 2)
    ss_tot = torch.sum((log_cumulative_returns - mean_log_returns) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else torch.tensor(0.0)
    
    # K-ratio = slope * sqrt(R-squared) * sqrt(n) (negative for PyTorch minimization - higher K-ratio is better)
    return -(slope * torch.sqrt(r_squared) * torch.sqrt(torch.tensor(n, dtype=torch.float32)))
#------------ Custom Loss Function -----------
# def portfolio_regularization_loss(weights: torch.Tensor, max_weight: float = 0.5, 
#                                  sparsity_lambda: float = 0.01, concentration_lambda: float = 0.01) -> torch.Tensor:
#     """
#     Calculate regularization losses for portfolio constraints.
    
#     Args:
#         weights: Portfolio weights tensor of shape (batch_size, n_assets)
#         max_weight: Maximum allowed weight for any asset
#         sparsity_lambda: Weight for sparsity regularization (encourages fewer assets)
#         concentration_lambda: Weight for concentration penalty (discourages max weight violations)
    
#     Returns:
#         Total regularization loss
#     """
#     batch_size, n_assets = weights.shape
    
#     # Sparsity regularization - L1 penalty to encourage sparse portfolios
#     sparsity_loss = sparsity_lambda * torch.sum(torch.abs(weights), dim=1).mean()
    
#     # Concentration penalty - penalize weights above max_weight
#     excess_weights = torch.clamp(weights - max_weight, min=0.0)
#     concentration_loss = concentration_lambda * torch.sum(excess_weights ** 2, dim=1).mean()
    
#     # Asset count penalty (alternative approach)
#     # Count non-zero weights and penalize if too many
#     non_zero_counts = (weights > 0.01).float().sum(dim=1)  # Count weights > 1%
#     count_penalty = 0.001 * torch.clamp(non_zero_counts - 20, min=0.0).mean()  # Penalize > 20 assets
    
#     return sparsity_loss + concentration_loss + count_penalty


def calculate_expected_metric(x_pred, df, metric, *args, **kwargs):
    """
    Calculate expected metric using a dictionary mapping approach.
    
    Args:
        x_pred: Predicted portfolio time series
        df: DataFrame (not used in current implementation but kept for compatibility)
        metric: String name of the metric to calculate
        *args, **kwargs: Additional arguments passed to the metric function
    
    Returns:
        Expected metric value
    """
    # Dictionary mapping metric names to their corresponding functions
    metric_functions = {
        'sharpe_ratio': sharpe_ratio,
        'geometric_sharpe_ratio': geometric_sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'geometric_sortino_ratio': geometric_sortino_ratio,
        'expected_return': expected_return,
        'carmdd': carmdd,
        'omega_ratio': omega_ratio,
        'jensen_alpha': jensen_alpha,
        'treynor_ratio': treynor_ratio,
        'ulcer_index': ulcer_index,
        'k_ratio': k_ratio
    }
    
    if metric not in metric_functions:
        raise ValueError(f"Unknown metric: {metric}. Available metrics: {list(metric_functions.keys())}")
    
    # Call the appropriate metric function
    return metric_functions[metric](x_pred, *args, **kwargs)

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
                # If 2D (timesteps, n_assets), assume single batch and transpose
                # Take last past_window_size timesteps
                past_window_size = model.past_window_size
                recent_data = matrix_input[-past_window_size:, :]  # (past_window_size, n_assets)
                matrix_input = recent_data.T.unsqueeze(0)  # (1, n_assets, past_window_size)
            
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
            
            # Drop columns that are all NaN
            df_temp = df_temp.dropna(axis=1, how='all')
            
            # Forward fill and backward fill to handle remaining NaN values
            df_temp = df_temp.fillna(method='ffill').fillna(method='bfill')
            
            # If still have NaN values, drop those columns
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
            raise TypeError(f"Unsupported data_input type: {type(data_input)}. "
                          f"Expected torch.Tensor, pandas.DataFrame, or numpy.ndarray")
        
        # Validate tensor shape
        if len(matrix_input.shape) != 3:
            raise ValueError(f"matrix_input must have 3 dimensions (batch_size, n_assets, past_window_size), "
                           f"got shape: {matrix_input.shape}")
        
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
            print(f"Warning: min_assets ({min_assets}) > available assets ({n_assets}). Using {effective_min_assets}")
        if max_assets > n_assets:
            print(f"Warning: max_assets ({max_assets}) > available assets ({n_assets}). Using {effective_max_assets}")
        if min_assets > max_assets:
            print(f"Warning: min_assets ({min_assets}) > max_assets ({max_assets}). Adjusted to min={effective_min_assets}, max={effective_max_assets}")

        # Scalar input: future window size (prediction horizon)
        scalar_input = torch.tensor(future_window_size, dtype=torch.float32, device=matrix_input.device)
        scalar_input = scalar_input.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, 1)
        
        # Constraint input: portfolio constraints (using validated values)
        constraint_vector = torch.tensor([
            max_weight,  # Max weight (0-1, where 1.0 = unconstrained)
            effective_min_assets / 100.0,  # Normalize min assets (0 = unconstrained)
            effective_max_assets / 100.0,  # Normalize max assets
            sparsity_threshold * 100.0  # Scale sparsity threshold (0.01 -> 1.0)
        ], dtype=torch.float32, device=matrix_input.device)
        
        # Expand constraint vector to batch size
        constraint_input = constraint_vector.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, 4)
        
        # Make prediction
        weights = model(matrix_input, scalar_input, constraint_input)
        
        return weights

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
    batch_size = losses.shape[0]
    
    # Take absolute value and add epsilon for numerical stability
    abs_losses = torch.abs(losses) + epsilon
    
    # Geometric mean using log-space operations (numerically stable)
    # log(geometric_mean) = mean(log(values)) = (sum(log(values))) / n
    # geometric_mean = exp(mean(log(values)))
    log_geometric_mean = torch.logsumexp(torch.log(abs_losses), dim=0) - torch.log(torch.tensor(batch_size, dtype=torch.float32))
    
    # Convert back from log space and preserve original sign
    result = torch.exp(log_geometric_mean)
    original_sign = torch.sign(losses.mean())
    
    return original_sign * result


def geometric_mean_square_error_aggregation(losses):
    """
    Implement Geometric Mean Square Error (GMSE) as you described.
    
    Args:
        losses: Tensor of shape (batch_size,) containing individual losses
    
    Returns:
        Geometric mean of squared losses, then square rooted
        
    This is: sqrt((prod(losses^2))^(1/n))
    Which is equivalent to: (prod(|losses|))^(1/n) with sign preservation
    """
    batch_size = losses.shape[0]
    
    # Square the losses to ensure positivity
    squared_losses = losses ** 2
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    squared_losses = squared_losses + epsilon
    
    # Geometric mean of squared losses using log-space for stability
    # log(geometric_mean) = mean(log(values))
    log_geometric_mean_squared = torch.logsumexp(torch.log(squared_losses), dim=0) - torch.log(torch.tensor(batch_size, dtype=torch.float32))
    
    # Take square root: sqrt(exp(log_geometric_mean_squared)) = exp(0.5 * log_geometric_mean_squared)
    result = torch.exp(0.5 * log_geometric_mean_squared)
    
    # Preserve the sign from the original mean
    original_sign = torch.sign(losses.mean())
    
    return original_sign * result


def update_model(model, optimizer, past_batch, future_batch, metric, 
                max_weight=1.0, min_assets=0, max_assets=1000, sparsity_threshold=0.01,
                regularization_lambda=0.0, loss_aggregation='arithmetic', *args, **kwargs):
    """
    Perform forward and backward pass to update the model.
    
    Args:
        model: The GPT2LikeTransformer model to update
        optimizer: PyTorch optimizer (e.g., Adam, SGD)
        past_batch: Dictionary containing past data for model input:
            - 'matrix_input': Tensor of shape (batch_size, n, past_window_size)
            - 'scalar_input': Tensor of shape (batch_size, 1) containing future_window_size
            - 'constraint_input': Tensor of shape (batch_size, 4) containing constraints
        future_batch: Dictionary containing future data for loss calculation:
            - 'returns': Future returns matrix for portfolio evaluation
        metric: String name of the metric to optimize (e.g., 'sharpe_ratio')
        max_weight: Maximum weight constraint (1.0 = unconstrained, used for regularization loss)
        min_assets: Minimum number of assets constraint (0 = unconstrained, used for regularization loss)
        max_assets: Maximum number of assets constraint (1000 = unconstrained, used for regularization loss)
        sparsity_threshold: Threshold for setting small weights to zero (used for regularization loss)
        regularization_lambda: Weight for portfolio regularization loss
        loss_aggregation: Method to aggregate losses across batch:
            - 'mae' or 'arithmetic': Mean Absolute Error - arithmetic mean (default)
            - 'mse': Mean Square Error - mean of squared losses
            - 'gmae' or 'geometric': Geometric Mean Absolute Error - log-space geometric mean
            - 'gmse' or 'geometric_mse': Geometric Mean Square Error (user's proposed method)
        *args, **kwargs: Additional arguments passed to metric calculation
    
    Returns:
        Dictionary containing:
            - 'loss': The total loss value
            - 'metric_loss': The primary metric loss
            - 'reg_loss': The regularization loss (if applied)
            - 'weights': The predicted portfolio weights
    """
    # Zero gradients
    optimizer.zero_grad()
    
    # Extract inputs from past_batch
    matrix_input = past_batch['matrix_input']
    scalar_input = past_batch['scalar_input']
    constraint_input = past_batch['constraint_input']
    
    # Forward pass through model (future window and constraints are now separate inputs)
    weights = model(matrix_input, scalar_input, constraint_input)
    
    # Extract future returns for portfolio evaluation
    future_returns = future_batch['returns']  # Shape: (timesteps, n_assets)
    
    # Calculate portfolio time series for each sample in the batch
    batch_size = weights.shape[0]
    metric_losses = []
    
    for i in range(batch_size):
        # Create portfolio time series from weights and future returns
        portfolio_timeseries = create_portfolio_time_series(future_returns, weights[i])
        
        # Calculate the metric (e.g., Sharpe ratio, etc.)
        metric_loss = calculate_expected_metric(portfolio_timeseries, None, metric, *args, **kwargs)
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
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping for stability (very important for transformer training)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update parameters
    optimizer.step()
    
    # Return loss information and weights
    return {
        'loss': total_loss.item(),
        'metric_loss': metric_loss.item(),
        'reg_loss': reg_loss.item() if regularization_lambda > 0.0 else 0.0,
        'weights': weights.detach()
    }

    