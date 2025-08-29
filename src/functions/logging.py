"""
Training logger for model training functions.
Replaces embedded logging with a clean, separate object.
"""

import os
import pandas as pd
from datetime import datetime
import torch


class TrainingLogger:
    """
    Simple training logger to replace embedded logging in model training functions.
    Handles console output, CSV logging, and model checkpointing.
    """
    
    def __init__(self, log_path=None, checkpoint_path=None, log_frequency=10, checkpoint_frequency=50):
        """
        Initialize the training logger.
        
        Args:
            log_path: Directory to save training logs (default: repo_root/logs/)
            checkpoint_path: Directory to save model checkpoints (default: repo_root/checkpoints/)
            log_frequency: How often to print console output (default: every 10 iterations)
            checkpoint_frequency: How often to save model checkpoints (default: every 50 iterations)
        """
        self.log_frequency = log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        
        # Set up default paths
        if log_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(current_dir))
            log_path = os.path.join(repo_root, "logs")
        
        if checkpoint_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(os.path.dirname(current_dir))
            checkpoint_path = os.path.join(repo_root, "checkpoints")
        
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        
        # Create directories
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Initialize log DataFrame
        self.log_data = []
        self.start_time = datetime.now()
        
        print(f"Training logger initialized")
        print(f"Logs: {self.log_path}")
        print(f"Checkpoints: {self.checkpoint_path}")
    
    def log_start(self, iterations):
        """Log training start."""
        print(f"Training started at {self.start_time}")
        print(f"Starting training for {iterations} iterations...")
    
    def log_iteration(self, iteration, loss_dict, training_params):
        """
        Log a single training iteration.
        
        Args:
            iteration: Current iteration number
            loss_dict: Dictionary with 'loss', 'metric_loss', 'reg_loss'
            training_params: Dictionary with training parameters for this iteration
        """
        # Store data for CSV logging
        log_row = {
            'iteration': iteration,
            'loss': loss_dict['loss'],
            'metric_loss': loss_dict['metric_loss'],
            'reg_loss': loss_dict['reg_loss']
        }
        log_row.update(training_params)
        self.log_data.append(log_row)
        
        # Console output at specified frequency
        if iteration % self.log_frequency == 0:
            phase = training_params.get('phase', 'N/A')
            loss_agg = training_params.get('loss_aggregation', 'N/A')
            progress = training_params.get('progress', 0) * 100
            lr = training_params.get('learning_rate', 0)
            batch_size = training_params.get('batch_size', 0)
            
            print(f"Iteration {iteration} | Phase: {phase} | Loss: {loss_dict['loss']:.6f} | "
                  f"Agg: {loss_agg.upper()} | Progress: {progress:.1f}% | LR: {lr:.2e} | Batch: {batch_size}")
    
    def log_early_stopping(self, iteration, best_loss, current_loss):
        """Log early stopping trigger."""
        print(f"\nEarly stopping triggered at iteration {iteration}")
        print(f"Best loss: {best_loss:.6f}, Current loss: {current_loss:.6f}")
    
    def checkpoint_model(self, model, iteration):
        """Save model checkpoint if needed."""
        if iteration % self.checkpoint_frequency == 0:
            checkpoint_filename = f'model_checkpoint_{iteration}.pt'
            checkpoint_filepath = os.path.join(self.checkpoint_path, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_filepath)
    
    def finalize_training(self, model, final_metrics):
        """
        Finalize training by saving final model and logs.
        
        Args:
            model: The trained model
            final_metrics: Dictionary with final training metrics
            
        Returns:
            Path to saved log file
        """
        # Save final model
        final_model_filepath = os.path.join(self.checkpoint_path, 'final_trained_model.pt')
        torch.save(model, final_model_filepath)
        
        final_state_dict_filepath = os.path.join(self.checkpoint_path, 'final_model_state_dict.pt')
        torch.save(model.state_dict(), final_state_dict_filepath)
        
        # Save log to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'training_log_{timestamp}.csv'
        log_filepath = os.path.join(self.log_path, log_filename)
        
        log_df = pd.DataFrame(self.log_data)
        log_df.to_csv(log_filepath, index=False)
        
        # Final summary
        end_time = datetime.now()
        print(f"\nTraining completed! Total time: {end_time - self.start_time}")
        print(f"Final loss: {final_metrics.get('final_loss', 'N/A'):.6f}")
        print(f"Final model saved to: {final_model_filepath}")
        print(f"Training log saved to: {log_filepath}")
        
        return log_filepath
