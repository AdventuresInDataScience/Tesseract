#!/usr/bin/env python3
"""
Test imports for the reorganized Tesseract functions.
"""

try:
    from src.functions import build_transformer_model, train_model_progressive, predict_portfolio_weights
    print("✅ All main imports successful!")
    
    from src.functions import (
        sharpe_ratio, create_portfolio_time_series, calculate_expected_metric,
        mean_square_error_aggregation, huber_loss_aggregation
    )
    print("✅ All secondary imports successful!")
    
    print("🎉 Code reorganization was successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
