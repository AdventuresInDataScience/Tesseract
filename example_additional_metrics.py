"""
Example demonstrating the new additional metrics tracking functionality.

This example shows how to optimize with one loss/aggregation combination 
while tracking multiple metrics with different aggregation methods.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.functions.model_train import train_model_progressive, create_additional_metrics_config
    from src.functions.model_build import create_gpt2_like_transformer
    import torch
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def create_sample_data(n_timesteps=1000, n_assets=50):
    """Create sample financial data for demonstration."""
    np.random.seed(42)
    
    # Generate random returns with some autocorrelation
    returns = np.random.normal(0.001, 0.02, (n_timesteps, n_assets))
    
    # Add some autocorrelation to make it more realistic
    for i in range(1, n_timesteps):
        returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
    
    # Convert to price levels (cumulative returns)
    prices = np.exp(np.cumsum(returns, axis=0))
    
    # Create DataFrame with asset names
    asset_names = [f"Asset_{i:03d}" for i in range(n_assets)]
    df = pd.DataFrame(prices, columns=asset_names)
    
    return df

def main():
    print("🚀 Additional Metrics Tracking Example")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample financial data...")
    data = create_sample_data(n_timesteps=500, n_assets=30)
    print(f"   Data shape: {data.shape}")
    
    # Create model
    print("🏗️  Creating model...")
    past_window_size = 20
    future_window_size = 5
    model = create_gpt2_like_transformer(
        input_dim=1,  # Single price time series per asset
        n_positions=past_window_size + 4,  # +4 for constraints
        n_embd=128,
        n_layer=4,
        n_head=4
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Configuration 1: Using helper function with tuples
    print("📈 Configuring additional metrics (Method 1: Tuples)...")
    additional_metrics_config = create_additional_metrics_config([
        # (name, metric, aggregation, winsorize)
        ('sharpe_huber_raw', 'sharpe_ratio', 'huber', False),
        ('sharpe_huber_winsorized', 'sharpe_ratio', 'huber', True),
        ('sharpe_gmae_raw', 'sharpe_ratio', 'gmae', False),
        ('sharpe_gmae_winsorized', 'sharpe_ratio', 'gmae', True),
        ('sharpe_arithmetic_raw', 'sharpe_ratio', 'arithmetic', False),
        ('max_drawdown_raw', 'max_drawdown', 'arithmetic', False),
        ('sortino_huber_winsorized', 'sortino_ratio', 'huber', True)
    ])
    
    print(f"   Configured {len(additional_metrics_config)} additional metrics:")
    for name, config in additional_metrics_config.items():
        winsorize_str = "winsorized" if config['winsorize'] else "raw"
        print(f"     • {name}: {config['metric']} + {config['aggregation']} ({winsorize_str})")
    
    # Configuration 2: Manual dictionary creation (alternative)
    # additional_metrics_config = {
    #     'sharpe_huber_raw': {'metric': 'sharpe_ratio', 'aggregation': 'huber', 'winsorize': False},
    #     'sharpe_gmae_winsorized': {'metric': 'sharpe_ratio', 'aggregation': 'gmae', 'winsorize': True},
    #     'max_drawdown_raw': {'metric': 'max_drawdown', 'aggregation': 'arithmetic', 'winsorize': False}
    # }
    
    print("\n🎯 Training Configuration:")
    print(f"   Primary optimization: sharpe_ratio + progressive aggregation (huber → gmae → gmse)")
    print(f"   Additional tracking: {len(additional_metrics_config)} metric/aggregation combinations")
    print(f"   Iterations: 100 (demo purposes)")
    
    # Train model with additional metrics tracking
    print("\n🏃 Starting training...")
    try:
        trained_model = train_model_progressive(
            model=model,
            optimizer=optimizer,
            data=data,
            past_window_size=past_window_size,
            future_window_size=future_window_size,
            
            # Training parameters
            min_n_cols=10,
            max_n_cols=25,
            min_batch_size=16,
            max_batch_size=32,
            iterations=100,  # Short demo
            
            # Primary optimization target
            loss='sharpe_ratio',
            loss_aggregation='progressive',  # Will use huber → gmae → gmse progression
            
            # Additional metrics tracking - THE NEW FEATURE!
            additional_metrics_config=additional_metrics_config,
            
            # Logging settings
            log_frequency=10,
            checkpoint_frequency=50,
            
            # Enhanced optimization
            learning_rate=1e-3,
            weight_decay=1e-4,
            gradient_accumulation_steps=2
        )
        
        print("\n✅ Training completed successfully!")
        print("\n📊 What was logged:")
        print("   • Primary loss (sharpe_ratio with progressive aggregation)")
        print("   • Raw loss statistics (mean, std, min, max)")
        print("   • Winsorized loss statistics (mean, std, min, max)")
        print("   • All additional metric/aggregation combinations:")
        
        for name in additional_metrics_config.keys():
            print(f"     - {name}")
        
        print("\n💡 Benefits of this approach:")
        print("   • Optimize using robust aggregation methods (e.g., huber for stability)")
        print("   • Track raw metrics for analysis without winsorization")
        print("   • Compare different aggregation methods side-by-side")
        print("   • Separate optimization concerns from analysis/logging concerns")
        print("   • Full visibility into loss landscape without affecting training")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
