import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import pandas as pd
import numpy as np

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
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
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
    
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
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
        causal: bool = True
    ):
        super().__init__()
        
        self.past_window_size = past_window_size
        self.d_model = d_model
        self.n_layers = n_transformer_blocks
        self.causal = causal
        
        # Input projection for the matrix input (n x t -> n x d_model)
        self.input_projection = nn.Linear(past_window_size, d_model)
        
        # Scalar embedding and projection
        self.scalar_embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_n + 1)  # +1 for scalar
        
        # Transformer blocks and intermediate MLPs
        self.transformer_blocks = nn.ModuleList()
        self.intermediate_mlps = nn.ModuleList()
        
        for _ in range(n_transformer_blocks):
            self.transformer_blocks.append(
                TransformerBlock(d_model, n_heads, d_ff, dropout, causal)
            )
            self.intermediate_mlps.append(
                IntermediateMLP(d_model, d_mlp_hidden, dropout)
            )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection (only for the matrix part, not scalar)
        self.output_projection = nn.Linear(d_model, 1)
        
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
    
    def forward(self, matrix_input: torch.Tensor, scalar_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            matrix_input: Tensor of shape (batch_size, n, past_window_size)
            scalar_input: Tensor of shape (batch_size, 1)
        
        Returns:
            output: Tensor of shape (batch_size, n) after softmax
        """
        batch_size, n, t = matrix_input.shape
        assert t == self.past_window_size, f"Expected t={self.past_window_size}, got t={t}"
        
        # Project matrix input to d_model
        x = self.input_projection(matrix_input)  # (batch_size, n, d_model)
        
        # Embed scalar and expand to match sequence length
        scalar_emb = self.scalar_embedding(scalar_input.unsqueeze(-1))  # (batch_size, 1, d_model)
        
        # Concatenate scalar embedding with matrix embeddings
        x = torch.cat([x, scalar_emb], dim=1)  # (batch_size, n+1, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Store scalar embedding for skip connections
        scalar_skip = scalar_emb.squeeze(1)  # (batch_size, d_model)
        
        # Pass through transformer blocks with intermediate MLPs
        for i, (transformer_block, intermediate_mlp) in enumerate(
            zip(self.transformer_blocks, self.intermediate_mlps)
        ):
            # Transformer block
            x = transformer_block(x)
            
            # Add scalar skip connection to all positions except the last (scalar) position
            x[:, :-1, :] = x[:, :-1, :] + scalar_skip.unsqueeze(1)
            
            # Intermediate MLP
            x = intermediate_mlp(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Extract only the matrix part (exclude scalar position)
        matrix_output = x[:, :-1, :]  # (batch_size, n, d_model)
        
        # Project to single values and squeeze
        output = self.output_projection(matrix_output).squeeze(-1)  # (batch_size, n)
        
        # Apply softmax
        output = F.softmax(output, dim=-1)
        
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
    causal: bool = True
) -> GPT2LikeTransformer:
    """
    Build a GPT-2-like transformer model with specified architecture.
    
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
    
    Returns:
        GPT2LikeTransformer model ready for training
    
    Example:
        >>> model = build_transformer_model(past_window_size=10, n_transformer_blocks=4, causal=True)
        >>> matrix_input = torch.randn(32, 50, 10)  # batch_size=32, n=50, t=10
        >>> scalar_input = torch.randn(32, 1)       # batch_size=32, scalar=1
        >>> output = model(matrix_input, scalar_input)  # shape: (32, 50)
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
        causal=causal
    )
    
    print(f"Model created with {model.get_num_parameters():,} parameters")
    return model


#--------------Portfolio Creation Function--------------
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
def sharpe_ratio():
    return 0

def max_drawdown():
    return 0

def sortino_ratio():
    return 0

def expected_return():
    return 0

def carmdd():
    return 0

def omega_ratio():
    return 0

def jensen_alpha():
    return 0

def treynor_ratio():
    return 0


#------------ Custom Loss Function -----------
def calculate_expected_metric(x_pred, df, metric):
    return expected_metric

def calculate_best_metric(x_pred, df, metric):
    return best_metric

def custom_loss_fn(x_pred, df_past, df_future, objective):
    
    return loss

def training_loop():
    return 0

