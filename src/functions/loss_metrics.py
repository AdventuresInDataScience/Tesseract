"""
Portfolio performance metrics for the Tesseract portfolio optimization system.
Contains various risk-adjusted return metrics and performance evaluation functions.
"""

import torch
import numpy as np
import math


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
def sharpe_ratio(portfolio_price_timeseries, risk_free_rate=0.0, trading_days_per_year=252):
    """
    Calculate annualized Sharpe ratio from a normalized price time series.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: annualized risk-free rate (default 0.0)
        trading_days_per_year: number of trading days per year for annualization (default 252)
    
    Returns:
        sharpe_ratio: torch scalar (annualized)
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0)
    returns = returns[1:]
    
    # Calculate mean and std of returns
    mean_return = returns.mean()
    std_returns = returns.std()
    
    # Annualize the Sharpe ratio: multiply by sqrt(trading_days_per_year)
    annualization_factor = torch.sqrt(torch.tensor(trading_days_per_year, dtype=torch.float32))
    
    # Convert risk_free_rate from annual to daily
    daily_risk_free_rate = risk_free_rate / trading_days_per_year
    
    # Calculate annualized Sharpe ratio (negative for PyTorch minimization)
    return -((mean_return - daily_risk_free_rate) / std_returns * annualization_factor)


def geometric_sharpe_ratio(portfolio_price_timeseries, risk_free_rate=0.0, trading_days_per_year=252):
    """
    Calculate annualized geometric Sharpe ratio from a normalized price time series.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: annualized risk-free rate (default 0.0)
        trading_days_per_year: number of trading days per year for annualization (default 252)
    
    Returns:
        geometric_sharpe_ratio: torch scalar (annualized)
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0)
    returns = returns[1:]
    
    # Calculate geometric mean return: (1 + r1) * (1 + r2) * ... * (1 + rn))^(1/n) - 1
    geometric_mean_return = torch.pow(torch.prod(1 + returns), 1.0 / len(returns)) - 1
    
    # Annualize the geometric mean return
    annualized_geometric_return = torch.pow(1 + geometric_mean_return, trading_days_per_year) - 1
    
    # Calculate standard deviation of returns (for denominator)
    std_returns = returns.std()
    
    # Annualize the standard deviation
    annualized_std = std_returns * torch.sqrt(torch.tensor(trading_days_per_year, dtype=torch.float32))
    
    # Calculate annualized geometric Sharpe ratio (negative for PyTorch minimization)
    return -((annualized_geometric_return - risk_free_rate) / annualized_std)


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


def sortino_ratio(portfolio_price_timeseries, risk_free_rate=0.0, target_return=0.0, trading_days_per_year=252):
    """
    Calculate annualized Sortino ratio from a normalized price time series.
    Only considers downside volatility (negative returns below target).
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: annualized risk-free rate (default 0.0)
        target_return: daily target return threshold (default 0.0)
        trading_days_per_year: number of trading days per year for annualization (default 252)
    
    Returns:
        sortino_ratio: torch scalar (annualized)
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0)
    returns = returns[1:]
    
    # Calculate mean return
    mean_return = returns.mean()
    
    # Calculate downside deviation (only negative returns below target)
    downside_returns = torch.minimum(returns - target_return, torch.tensor(0.0))
    downside_deviation = torch.sqrt(torch.mean(downside_returns ** 2))
    
    # Annualize both numerator and denominator
    annualization_factor = torch.sqrt(torch.tensor(trading_days_per_year, dtype=torch.float32))
    daily_risk_free_rate = risk_free_rate / trading_days_per_year
    
    # Calculate annualized Sortino ratio (negative for PyTorch minimization)
    return -((mean_return - daily_risk_free_rate) / downside_deviation * annualization_factor)


def geometric_sortino_ratio(portfolio_price_timeseries, risk_free_rate=0.0, target_return=0.0, trading_days_per_year=252):
    """
    Calculate annualized geometric Sortino ratio from a normalized price time series.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        risk_free_rate: annualized risk-free rate (default 0.0)
        target_return: daily target return threshold (default 0.0)
        trading_days_per_year: number of trading days per year for annualization (default 252)
    
    Returns:
        geometric_sortino_ratio: torch scalar (annualized)
    """
    # Add 1.0 at the start to represent the normalized reference point
    full_series = torch.cat([torch.tensor([1.0], device=portfolio_price_timeseries.device), 
                            portfolio_price_timeseries])
    
    # Calculate percentage changes: (P_t / P_{t-1}) - 1
    returns = (full_series[1:] / full_series[:-1]) - 1
    
    # Remove the first return (which corresponds to the transition from 1.0)
    returns = returns[1:]
    
    # Calculate geometric mean return and annualize it
    geometric_mean_return = torch.pow(torch.prod(1 + returns), 1.0 / len(returns)) - 1
    annualized_geometric_return = torch.pow(1 + geometric_mean_return, trading_days_per_year) - 1
    
    # Calculate downside deviation (only negative returns below target)
    downside_returns = torch.minimum(returns - target_return, torch.tensor(0.0))
    downside_deviation = torch.sqrt(torch.mean(downside_returns ** 2))
    
    # Annualize the downside deviation
    annualized_downside_deviation = downside_deviation * torch.sqrt(torch.tensor(trading_days_per_year, dtype=torch.float32))
    
    # Calculate annualized geometric Sortino ratio (negative for PyTorch minimization)
    return -((annualized_geometric_return - risk_free_rate) / annualized_downside_deviation)


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


def carmdd(portfolio_price_timeseries, trading_days_per_year=252):
    """
    Calculate the Calmar ratio (CARMDD) from a normalized price time series.
    Uses proper compounding for annualized returns.
    
    Args:
        portfolio_price_timeseries: torch tensor of normalized prices (continuation from 1.0)
        trading_days_per_year: number of trading days per year for annualization (default 252)
    
    Returns:
        carmdd: torch scalar representing the Calmar ratio
    """
    # Calculate maximum drawdown
    running_max = torch.cummax(portfolio_price_timeseries, dim=0).values
    drawdowns = (running_max - portfolio_price_timeseries) / running_max
    max_drawdown_val = torch.max(drawdowns)
    
    # Calculate annualized return using proper compounding
    # Total return ratio = final_value / initial_value (initial is 1.0, so just final value)
    total_return_ratio = portfolio_price_timeseries[-1]
    
    # Number of periods (days) in the series
    n_periods = len(portfolio_price_timeseries)
    
    # Annualize using compound growth: (final/initial)^(252/n_periods) - 1
    # Since portfolio_price_timeseries already represents the ratio from initial value 1.0,
    # we use it directly as the total return ratio
    annualized_return = torch.pow(total_return_ratio, trading_days_per_year / n_periods) - 1
    
    # Calculate Calmar ratio (negative for PyTorch minimization - higher Calmar is better)
    # Avoid division by zero
    if max_drawdown_val > 1e-8:
        return -(annualized_return / max_drawdown_val)
    else:
        return torch.tensor(-1000.0)  # Very high value when no drawdown


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
