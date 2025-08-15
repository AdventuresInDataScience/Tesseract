"""
Model building functions for the Tesseract portfolio optimization system.
Contains model architecture definitions, activation functions, and model construction utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
import json
from datetime import datetime


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
                            # Select top assets from zero weights by original logits
                            original_values = logits[i][zero_indices]
                            _, top_relative = torch.topk(original_values, needed)
                            selected_indices = zero_indices[top_relative]
                            
                            # Assign minimum weights to selected assets
                            min_weight = 0.01 / effective_min_assets  # Small but non-zero weight
                            values = torch.full((needed,), min_weight, device=w.device)
                            w = w.scatter(0, selected_indices, values)
                        else:
                            # If we don't have enough zero weights, use all available zero weights
                            if len(zero_indices) > 0:
                                min_weight = 0.01 / len(zero_indices)
                                values = torch.full((len(zero_indices),), min_weight, device=w.device)
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
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Model configuration loaded from: {config_path}")
    return config
