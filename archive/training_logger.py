"""
Training Logger Module for Tesseract Portfolio Optimization

Provides a centralized, clean logging interface for model training that separates
logging concerns from training logic. Handles console output, CSV logging, 
model checkpointing, metrics aggregation, and progress tracking.

Author: Tesseract Training System
"""

import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

try:
    from .loss_metrics import create_portfolio_time_series, calculate_expected_metric
except ImportError:
    from loss_metrics import create_portfolio_time_series, calculate_expected_metric


class TrainingLogger:
    """
    Centralized training logger that handles all logging responsibilities:
    - Console output with customizable formatting
    - CSV-based structured logging with comprehensive metrics
    - Model checkpointing and versioning
    - Path management and directory creation
    - Additional metrics calculation and aggregation
    - Progress tracking and phase transitions
    """
    
    def __init__(self, 
                 log_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 log_frequency: int = 10,
                 checkpoint_frequency: int = 50,
                 other_metrics_to_log: Optional[Union[str, List[str]]] = None,
                 training_type: str = "progressive"):
        """
        Initialize the training logger.
        
        Args:
            log_path: Path to save training logs (default: repo_root/logs/)
            checkpoint_path: Path to save model checkpoints (default: repo_root/checkpoints/)
            log_frequency: How often to print console output (default: every 10 iterations)
            checkpoint_frequency: How often to save model checkpoints (default: every 50 iterations)
            other_metrics_to_log: Additional metrics to calculate and log alongside primary metric
            training_type: Type of training ("progressive" or "curriculum") for specialized logging
        """
        self.log_frequency = log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.training_type = training_type
        
        # Setup paths
        self.log_path, self.checkpoint_path = self._setup_paths(log_path, checkpoint_path)
        
        # Process metrics to log
        self.other_metrics_list = self._process_other_metrics(other_metrics_to_log)
        
        # Initialize logging state
        self.start_time = None
        self.log_data = []
        self.best_loss = float('inf')
        self.iteration_count = 0
        
        # Setup log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'{training_type}_training_log_{timestamp}.csv'
        self.log_filepath = os.path.join(self.log_path, log_filename)
        
        print(f"🔧 TrainingLogger initialized:")
        print(f"   📁 Logs: {self.log_path}")
        print(f"   📁 Checkpoints: {self.checkpoint_path}")
        print(f"   📊 Log frequency: every {log_frequency} iterations")
        print(f"   💾 Checkpoint frequency: every {checkpoint_frequency} iterations")
        if self.other_metrics_list:
            print(f"   📈 Additional metrics: {self.other_metrics_list}")
    
    def _setup_paths(self, log_path: Optional[str], checkpoint_path: Optional[str]) -> tuple:
        """Set up default paths for logs and checkpoints."""
        if log_path is None:
            # Get the repository root (assuming this file is in src/functions/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(current_dir))
            log_path = os.path.join(repo_root, "logs")
        
        if checkpoint_path is None:
            # Get the repository root (assuming this file is in src/functions/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(current_dir))
            checkpoint_path = os.path.join(repo_root, "checkpoints")
        
        # Create directories if they don't exist
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        return log_path, checkpoint_path
    
    def _process_other_metrics(self, other_metrics_to_log: Optional[Union[str, List[str]]]) -> List[str]:
        """Process and validate the other_metrics_to_log parameter."""
        if other_metrics_to_log is None:
            return []
        elif isinstance(other_metrics_to_log, str):
            return [other_metrics_to_log]
        elif isinstance(other_metrics_to_log, list):
            return other_metrics_to_log
        else:
            raise ValueError("other_metrics_to_log must be None, a string, or a list of strings")
    
    def start_training(self, total_iterations: int) -> None:
        """Mark the start of training and initialize tracking."""
        self.start_time = datetime.now()
        self.total_iterations = total_iterations
        print(f"\n🚀 {self.training_type.title()} Training started at {self.start_time}")
        print(f"   🎯 Target iterations: {total_iterations}")
    
    def log_iteration(self, 
                     iteration: int,
                     loss_dict: Dict[str, Any],
                     training_params: Dict[str, Any],
                     additional_context: Optional[Dict[str, Any]] = None,
                     force_console_output: bool = False) -> None:
        """
        Log a single training iteration with comprehensive metrics.
        
        Args:
            iteration: Current iteration number (1-based)
            loss_dict: Dictionary containing loss information from training step
            training_params: Dictionary containing current training parameters
            additional_context: Optional additional context for specialized logging
            force_console_output: Force console output regardless of frequency
        """
        self.iteration_count = iteration
        
        # Build log entry
        log_entry = {
            'iteration': iteration,
            'loss': loss_dict.get('loss', 0.0),
            'metric_loss': loss_dict.get('metric_loss', 0.0),
            'reg_loss': loss_dict.get('reg_loss', 0.0),
        }
        
        # Add training parameters
        log_entry.update(training_params)
        
        # Add additional context if provided
        if additional_context:
            log_entry.update(additional_context)
        
        # Calculate additional metrics if requested
        if self.other_metrics_list and 'weights' in loss_dict and 'future_batch' in loss_dict:
            additional_metrics = self._calculate_additional_metrics(
                loss_dict['weights'], loss_dict['future_batch'])
            log_entry.update(additional_metrics)
        
        # Store in memory
        self.log_data.append(log_entry)
        
        # Console output at specified frequency or when forced
        if force_console_output or iteration % self.log_frequency == 0:
            self._print_progress(iteration, log_entry)
        
        # Periodic CSV saving (for long training runs)
        if iteration % (self.log_frequency * 5) == 0:  # Save every 5 console outputs
            self._save_log_to_csv()
    
    def _calculate_additional_metrics(self, weights, future_batch) -> Dict[str, float]:
        """Calculate additional metrics if requested."""
        additional_metrics = {}
        
        if not self.other_metrics_list:
            return additional_metrics
            
        try:
            # Get portfolio weights from the first sample in the batch
            portfolio_weights = weights[0] if len(weights) > 0 else weights
            future_returns = future_batch['returns']
            
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
            for metric_name in self.other_metrics_list:
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
            for metric_name in self.other_metrics_list:
                additional_metrics[metric_name] = float('nan')
        
        return additional_metrics
    
    def _print_progress(self, iteration: int, log_entry: Dict[str, Any]) -> None:
        """Print formatted progress to console."""
        # Base progress info
        loss = log_entry.get('loss', 0.0)
        
        # Build progress string based on training type
        if self.training_type == "progressive":
            progress_str = self._format_progressive_output(iteration, log_entry)
        elif self.training_type == "curriculum":
            progress_str = self._format_curriculum_output(iteration, log_entry)
        else:
            # Generic format
            progress_str = f"Iteration {iteration}/{self.total_iterations} | Loss: {loss:.6f}"
        
        print(progress_str)
    
    def _format_progressive_output(self, iteration: int, log_entry: Dict[str, Any]) -> str:
        """Format console output for progressive training."""
        loss = log_entry.get('loss', 0.0)
        phase = log_entry.get('phase', 'Unknown')
        loss_agg = log_entry.get('loss_aggregation', 'unknown')
        progress = log_entry.get('progress', 0.0)
        lr = log_entry.get('learning_rate', 0.0)
        batch_size = log_entry.get('batch_size', 0)
        
        return (f"Iteration {iteration}/{self.total_iterations} | "
                f"Phase: {phase} | Loss: {loss:.6f} | "
                f"Agg: {loss_agg.upper()} | Progress: {progress*100:.1f}% | "
                f"LR: {lr:.2e} | Batch: {batch_size}")
    
    def _format_curriculum_output(self, iteration: int, log_entry: Dict[str, Any]) -> str:
        """Format console output for curriculum training."""
        loss = log_entry.get('loss', 0.0)
        phase = log_entry.get('phase', 'Unknown')
        cols = log_entry.get('n_cols_sampled', 0)
        bucket = log_entry.get('column_bucket', 0)
        max_weight = log_entry.get('max_weight_used', 0.0)
        min_assets = log_entry.get('min_assets_used', 0)
        max_assets = log_entry.get('max_assets_used', 0)
        sparsity = log_entry.get('sparsity_threshold_used', 0.0)
        lr = log_entry.get('learning_rate', 0.0)
        
        return (f"Iter {iteration}/{self.total_iterations} | Phase {phase} | "
                f"Loss: {loss:.6f} | Cols: {cols} | Bucket: {bucket} | "
                f"MaxWt: {max_weight:.3f} | MinAssets: {min_assets} | "
                f"MaxAssets: {max_assets} | Sparsity: {sparsity:.3f} | "
                f"LR: {lr:.2e}")
    
    def checkpoint_model(self, model: torch.nn.Module, iteration: int, 
                        force_checkpoint: bool = False) -> None:
        """Save model checkpoint if frequency conditions are met."""
        if force_checkpoint or iteration % self.checkpoint_frequency == 0:
            checkpoint_filename = f'{self.training_type}_checkpoint_{iteration}.pt'
            checkpoint_filepath = os.path.join(self.checkpoint_path, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_filepath)
            
            # Also save a running log at checkpoint time
            self._save_log_to_csv()
    
    def _save_log_to_csv(self) -> None:
        """Save current log data to CSV file."""
        if self.log_data:
            log_df = pd.DataFrame(self.log_data)
            log_df.to_csv(self.log_filepath, index=False)
    
    def log_phase_transition(self, from_phase: str, to_phase: str, iteration: int) -> None:
        """Log phase transitions for curriculum/progressive training."""
        print(f"\n🔄 PHASE TRANSITION at iteration {iteration}: {from_phase} → {to_phase}")
        print(f"   Expect step change in loss due to different training parameters\n")
    
    def log_early_stopping(self, iteration: int, best_loss: float, current_loss: float) -> None:
        """Log early stopping event."""
        print(f"\n⏹️  Early stopping triggered at iteration {iteration}")
        print(f"Best loss: {best_loss:.6f}, Current loss: {current_loss:.6f}")
    
    def finalize_training(self, model: torch.nn.Module, 
                         final_metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Finalize training by saving final model and logs.
        
        Args:
            model: Trained model to save
            final_metrics: Optional final metrics summary
            
        Returns:
            Path to final log file
        """
        end_time = datetime.now()
        training_time = end_time - self.start_time if self.start_time else "Unknown"
        
        # Save final complete model (architecture + weights)
        final_model_filename = f'final_{self.training_type}_model.pt'
        final_model_filepath = os.path.join(self.checkpoint_path, final_model_filename)
        torch.save(model, final_model_filepath)
        
        # Also save final state dict for compatibility
        final_state_dict_filename = f'final_{self.training_type}_state_dict.pt'
        final_state_dict_filepath = os.path.join(self.checkpoint_path, final_state_dict_filename)
        torch.save(model.state_dict(), final_state_dict_filepath)
        
        # Save final log to CSV
        self._save_log_to_csv()
        
        # Print training summary
        print(f"\n🏁 {self.training_type.title()} Training completed!")
        print(f"   ⏱️  Total time: {training_time}")
        print(f"   🔢 Total iterations: {self.iteration_count}")
        
        if final_metrics:
            print(f"   📊 Final metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.6f}")
                else:
                    print(f"      {key}: {value}")
        
        print(f"   💾 Final model saved: {final_model_filepath}")
        print(f"   📄 Training log saved: {self.log_filepath}")
        
        return self.log_filepath


class ProgressiveTrainingLogger(TrainingLogger):
    """Specialized logger for progressive training with phase tracking."""
    
    def __init__(self, **kwargs):
        super().__init__(training_type="progressive", **kwargs)
        self.previous_aggregation = None
    
    def log_iteration(self, iteration: int, loss_dict: Dict[str, Any], 
                     training_params: Dict[str, Any], 
                     additional_context: Optional[Dict[str, Any]] = None,
                     force_console_output: bool = False) -> None:
        """Log progressive training iteration with phase transition detection."""
        
        # Detect loss aggregation transitions
        current_agg = training_params.get('loss_aggregation', 'unknown')
        if (self.previous_aggregation and 
            self.previous_aggregation != current_agg):
            self.log_phase_transition(
                self.previous_aggregation.upper(), 
                current_agg.upper(), 
                iteration
            )
        self.previous_aggregation = current_agg
        
        # Call parent method
        super().log_iteration(iteration, loss_dict, training_params, 
                             additional_context, force_console_output)


class CurriculumTrainingLogger(TrainingLogger):
    """Specialized logger for curriculum training with bootstrap and phase tracking."""
    
    def __init__(self, **kwargs):
        super().__init__(training_type="curriculum", **kwargs)
        self.current_phase = None
        self.bootstrap_resets = 0
    
    def log_phase_change(self, phase_info: Dict[str, Any], iteration: int) -> None:
        """Log curriculum phase change."""
        phase_id = phase_info.get('phase_id', 'Unknown')
        batch_size = phase_info.get('batch_size', 0)
        column_bucket = phase_info.get('column_bucket', 0)
        constraint_step = phase_info.get('constraint_step', 0)
        iterations = phase_info.get('iterations', 0)
        
        if self.current_phase != phase_id:
            print(f"\n🎯 PHASE {phase_id}: Batch={batch_size}, "
                  f"Column={column_bucket}, Constraint={constraint_step}")
            print(f"   Iterations: {iterations} | Batch size: {batch_size}")
            self.current_phase = phase_id
    
    def log_bootstrap_reset(self, bucket_id: int) -> None:
        """Log bootstrap column sampling reset."""
        self.bootstrap_resets += 1
        print(f"🔄 Bootstrap reset for column bucket {bucket_id} "
              f"(Reset #{self.bootstrap_resets})")


# Convenience factory functions
def create_progressive_logger(**kwargs) -> ProgressiveTrainingLogger:
    """Create a logger optimized for progressive training."""
    return ProgressiveTrainingLogger(**kwargs)

def create_curriculum_logger(**kwargs) -> CurriculumTrainingLogger:
    """Create a logger optimized for curriculum training."""
    return CurriculumTrainingLogger(**kwargs)
