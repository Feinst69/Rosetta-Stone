# Configuration for LSTM Translator
# This file contains all model hyperparameters and feature toggles

MODEL_CONFIG = {
    # Basic Architecture
    'max_seq_length': 50,
    'embedding_dim': 512,           # Increased from 256
    'hidden_units': 1024,           # Increased from 512  
    'max_vocab_size': 50000,        # Increased from 20000
    'dropout_rate': 0.2,
    
    # Training Parameters
    'batch_size': 64,
    'epochs': 50,                   # Increased from 10
    'patience': 10,                 # Increased from 3
    'learning_rate': 0.001,
    'min_lr': 0.0001,
    'lr_reduce_factor': 0.5,
    'lr_reduce_patience': 5,
    
    # Advanced Features (toggleable)
    'use_attention': True,              # Re-enabled with fixed implementation
    'use_bidirectional': True,
    'use_teacher_forcing': True,        # If False, uses free running mode
    'use_scheduled_sampling': True,     # Re-enabled - this should work fine
    
    # Attention Configuration
    'attention_type': 'luong',      # 'luong' or 'bahdanau'
    'attention_units': 512,
    
    # Scheduled Sampling Configuration
    'sampling_probability_start': 0.0,
    'sampling_probability_end': 0.5,
    'sampling_schedule': 'linear',  # 'linear', 'exponential', 'inverse_sigmoid'
    'sampling_start_epoch': 5,      # Start scheduled sampling after this epoch
    
    # Bidirectional LSTM Configuration
    'bidirectional_merge_mode': 'concat',  # 'sum', 'mul', 'concat', 'ave'
    
    # Model saving/loading
    'save_best_only': True,
    'save_weights_only': False,
    'monitor_metric': 'val_loss',
    
    # Validation
    'validation_size': 1000,
    'test_size': 0.2,
    'random_state': 42
}

def get_config():
    """Get the current model configuration"""
    return MODEL_CONFIG.copy()

def update_config(**kwargs):
    """Update configuration parameters"""
    for key, value in kwargs.items():
        if key in MODEL_CONFIG:
            MODEL_CONFIG[key] = value
            print(f"Updated {key}: {value}")
        else:
            print(f"Warning: Unknown configuration key '{key}'")

def print_config():
    """Print current configuration"""
    print("=== Current Model Configuration ===")
    for category in ['Basic Architecture', 'Training Parameters', 'Advanced Features']:
        print(f"\n{category}:")
        
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 40)