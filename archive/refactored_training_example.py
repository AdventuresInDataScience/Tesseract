"""
Example refactored training function using the new TrainingLogger.

This shows how the existing train_model_progressive function would be modified
to use the centralized logging system instead of embedded logging code.
"""

# This is how the train_model_progressive function would look with the new logger:

def train_model_progressive_refactored(model, optimizer, data, past_window_size, future_window_size, 
                              min_n_cols=10, max_n_cols=100, min_batch_size=32, max_batch_size=256, 
                              iterations=1000, loss='sharpe_ratio', loss_aggregation='progressive',
                              # Logging parameters (now delegated to logger)
                              other_metrics_to_log=None, log_path=None, checkpoint_path=None,
                              checkpoint_frequency=50, log_frequency=10,
                              # Training parameters (unchanged)
                              max_weight_range=(0.1, 1.0), min_assets_range=(0, 50), 
                              max_assets_range=(5, 200), sparsity_threshold_range=(0.005, 0.05),
                              use_scheduler=True, scheduler_patience=500,
                              early_stopping_patience=2000, early_stopping_threshold=1e-6,
                              learning_rate=1e-3, weight_decay=2e-4, warmup_steps=500):
    """
    Progressive training with centralized logging.
    
    CLEAN SEPARATION OF CONCERNS:
    - Training logic focuses purely on model optimization
    - All logging responsibilities delegated to TrainingLogger
    - Reduced code duplication and improved maintainability
    """
    
    # === INITIALIZATION (MUCH CLEANER) ===
    
    # Create specialized logger for progressive training
    logger = create_progressive_logger(
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        log_frequency=log_frequency,
        checkpoint_frequency=checkpoint_frequency,
        other_metrics_to_log=other_metrics_to_log
    )
    
    # Start training tracking
    logger.start_training(iterations)
    
    # Setup training components (unchanged)
    enhanced_optimizer = _progressive_setup_enhanced_optimizer(model, learning_rate, weight_decay)
    lr_scheduler = _progressive_setup_learning_rate_scheduler(enhanced_optimizer, warmup_steps, iterations)
    plateau_scheduler = _progressive_setup_plateau_scheduler(enhanced_optimizer, use_scheduler, scheduler_patience)
    
    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    
    # Data preparation
    data_array = _progressive_convert_data_to_array(data)
    valid_indices = len(data_array) - (past_window_size + future_window_size)
    
    # === TRAINING LOOP (FOCUSED ON OPTIMIZATION) ===
    
    for i in range(iterations):
        # Calculate progressive values
        progress = i / (iterations - 1) if iterations > 1 else 0
        current_loss_aggregation, phase = _progressive_determine_loss_aggregation(loss_aggregation, progress, i)
        current_n_cols, current_batch_size = _progressive_calculate_progressive_values(
            progress, min_n_cols, max_n_cols, min_batch_size, max_batch_size)
        
        # Create training batch
        max_weight, min_assets, max_assets, sparsity_threshold = _progressive_calculate_constraint_values(
            progress, max_weight_range, min_assets_range, max_assets_range, sparsity_threshold_range)
        
        past_batch_tensor, future_batch_tensor = _create_batch(
            data_array, past_window_size, future_window_size, current_n_cols, current_batch_size, valid_indices)
        
        # Validate constraints
        max_weight, effective_min_assets, effective_max_assets, sparsity_threshold = _progressive_validate_constraints(
            max_weight, min_assets, max_assets, sparsity_threshold, current_n_cols)
        
        # Prepare model inputs
        normalized_scalar_input, normalized_constraint_input, raw_constraints = _progressive_create_model_inputs(
            future_window_size, current_batch_size, max_weight, effective_min_assets, effective_max_assets, sparsity_threshold)
        
        past_batch = {
            'matrix_input': past_batch_tensor,
            'scalar_input': normalized_scalar_input,
            'constraint_input': normalized_constraint_input,
            'raw_constraints': raw_constraints
        }
        
        future_batch = {
            'returns': future_batch_tensor[0].T
        }
        
        # Model update
        enhanced_optimizer.zero_grad()
        loss_dict = _update_model(
            model=model, optimizer=enhanced_optimizer, past_batch=past_batch, 
            future_batch=future_batch, loss=loss,
            max_weight=raw_constraints['max_weight'], 
            min_assets=raw_constraints['min_assets'], 
            max_assets=raw_constraints['max_assets'], 
            sparsity_threshold=raw_constraints['sparsity_threshold'],
            loss_aggregation=current_loss_aggregation
        )
        
        # Learning rate updates
        lr_scheduler.step()
        current_iteration_loss = loss_dict['loss']
        
        if plateau_scheduler is not None:
            plateau_scheduler.step(current_iteration_loss)
        
        # Early stopping
        if current_iteration_loss < best_loss - early_stopping_threshold:
            best_loss = current_iteration_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logger.log_early_stopping(i + 1, best_loss, current_iteration_loss)
            break
        
        # === LOGGING (COMPLETELY DELEGATED) ===
        
        # Prepare training parameters for logging
        training_params = {
            'loss_aggregation': current_loss_aggregation,
            'phase': phase,
            'batch_size': current_batch_size,
            'n_cols': current_n_cols,
            'progress': progress,
            'learning_rate': lr_scheduler.get_last_lr()[0],
            # Constraint ranges for context
            'max_weight_range': f"{max_weight_range[0]:.3f}-{max_weight_range[1]:.3f}",
            'min_assets_range': f"{min_assets_range[0]}-{min_assets_range[1]}",
            'max_assets_range': f"{max_assets_range[0]}-{max_assets_range[1]}",
            'sparsity_threshold_range': f"{sparsity_threshold_range[0]:.3f}-{sparsity_threshold_range[1]:.3f}",
            # Actual constraint values used
            'max_weight_used': raw_constraints['max_weight'],
            'min_assets_used': raw_constraints['min_assets'],
            'max_assets_used': raw_constraints['max_assets'],
            'sparsity_threshold_used': raw_constraints['sparsity_threshold']
        }
        
        # Add weights and future_batch to loss_dict for additional metrics calculation
        loss_dict['weights'] = loss_dict.get('weights', torch.zeros(1, current_n_cols))
        loss_dict['future_batch'] = future_batch
        
        # Single logger call handles everything:
        # - Console output (at specified frequency)
        # - CSV logging (every iteration)
        # - Additional metrics calculation
        # - Progress tracking
        logger.log_iteration(i + 1, loss_dict, training_params)
        
        # Model checkpointing (handled by logger)
        logger.checkpoint_model(model, i + 1)
    
    # === FINALIZATION (ONE CLEAN CALL) ===
    
    final_metrics = {
        'final_loss': current_iteration_loss,
        'final_aggregation': current_loss_aggregation,
        'final_lr': lr_scheduler.get_last_lr()[0],
        'final_batch_size': current_batch_size,
        'iterations_completed': i + 1
    }
    
    log_filepath = logger.finalize_training(model, final_metrics)
    
    return model

# COMPARISON: 
# BEFORE: ~300 lines of mixed training + logging code
# AFTER:  ~150 lines of pure training logic + centralized logging
# RESULT: 50% code reduction, clear separation of concerns, reusable logging
