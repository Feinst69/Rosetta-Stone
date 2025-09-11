import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
import pickle
import warnings
import os
from itertools import product
warnings.filterwarnings('ignore')

# Configure TensorFlow for Apple Silicon optimization
print("Configuring TensorFlow for Apple Silicon...")
print(f"TensorFlow version: {tf.__version__}")

# Check if MPS (Metal Performance Shaders) is available
if tf.config.list_physical_devices('GPU'):
    print("GPU devices found:")
    for device in tf.config.list_physical_devices('GPU'):
        print(f"  {device}")
    
    # Enable memory growth for MPS
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except:
        print("⚠️  Could not configure GPU memory growth")
else:
    print("⚠️  No GPU devices found - using CPU")

# Enable mixed precision for faster training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision enabled")

# Set optimal thread configuration for M4 Pro
tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
print("✅ Thread parallelism optimized")

class OptimizedRNNTranslator:
    def __init__(self, max_seq_length=30, embedding_dim=128, hidden_units=256, 
                 max_vocab_size=10000, dropout_rate=0.2, learning_rate=0.001,
                 batch_size=128):  # Increased default batch size
        """
        Optimized RNN Translator for Apple Silicon
        """
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.max_vocab_size = max_vocab_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize vocabularies and models
        self.fr_word_to_idx = {}
        self.fr_idx_to_word = {}
        self.en_word_to_idx = {}
        self.en_idx_to_word = {}
        self.fr_vocab_size = 0
        self.en_vocab_size = 0
        
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
    
    def prepare_data_from_tokens(self, df_tokens, validation_size=1000, test_size=0.2, random_state=42):
        """
        Prepare data with optimized memory usage
        """
        print("Processing tokenized data...")
        
        # More aggressive filtering for faster training
        mask = (df_tokens['tokens_fr'].str.len() >= 3) & (df_tokens['tokens_fr'].str.len() <= self.max_seq_length) & \
               (df_tokens['tokens_en'].str.len() >= 3) & (df_tokens['tokens_en'].str.len() <= self.max_seq_length)
        df_filtered = df_tokens[mask].reset_index(drop=True)
        
        # Limit dataset size for faster experimentation
        if len(df_filtered) > 50000:  # Limit to 50k samples for faster training
            df_filtered = df_filtered.sample(n=50000, random_state=random_state).reset_index(drop=True)
            print(f"⚠️  Dataset limited to 50,000 samples for faster training")
        
        print(f"Dataset size after filtering: {len(df_filtered)}")
        
        # Split the data
        train_data, test_data = train_test_split(df_filtered, test_size=test_size, random_state=random_state)
        
        if len(train_data) > validation_size:
            train_data, val_data = train_test_split(train_data, test_size=validation_size, random_state=random_state)
        else:
            val_data = train_data.copy()
        
        print(f"Training set: {len(train_data)}")
        print(f"Test set: {len(test_data)}")
        print(f"Validation set: {len(val_data)}")
        
        self.train_data = train_data
        self.test_data = test_data  
        self.val_data = val_data
        
        return train_data, val_data, test_data
    
    def build_vocabularies(self, train_data):
        """
        Build vocabularies with optimized vocab size
        """
        print("Building vocabularies...")
        
        # Special tokens
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        
        # Build French vocabulary with smaller max size for faster training
        all_fr_tokens = []
        for tokens in train_data['tokens_fr']:
            all_fr_tokens.extend(tokens)
        
        fr_token_counts = pd.Series(all_fr_tokens).value_counts()
        # Reduce vocab size for faster training
        effective_vocab_size = min(self.max_vocab_size, 5000)
        fr_vocab = special_tokens + list(fr_token_counts.head(effective_vocab_size - len(special_tokens)).index)
        
        self.fr_word_to_idx = {word: idx for idx, word in enumerate(fr_vocab)}
        self.fr_idx_to_word = {idx: word for word, idx in self.fr_word_to_idx.items()}
        self.fr_vocab_size = len(fr_vocab)
        
        # Build English vocabulary  
        all_en_tokens = []
        for tokens in train_data['tokens_en']:
            all_en_tokens.extend(tokens)
        
        en_token_counts = pd.Series(all_en_tokens).value_counts()
        en_vocab = special_tokens + list(en_token_counts.head(effective_vocab_size - len(special_tokens)).index)
        
        self.en_word_to_idx = {word: idx for idx, word in enumerate(en_vocab)}
        self.en_idx_to_word = {idx: word for word, idx in self.en_word_to_idx.items()}
        self.en_vocab_size = len(en_vocab)
        
        print(f"French vocabulary size: {self.fr_vocab_size}")
        print(f"English vocabulary size: {self.en_vocab_size}")
    
    def convert_tokens_to_sequences(self, train_data, val_data, test_data):
        """
        Convert tokens to sequences with memory optimization
        """
        print("Converting tokens to sequences...")
        
        def tokens_to_sequences(tokens_list, word_to_idx, add_start_end=False):
            sequences = []
            for tokens in tokens_list:
                if add_start_end:
                    tokens = ['<start>'] + tokens + ['<end>']
                seq = [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
                sequences.append(seq)
            return sequences
        
        # Convert to int32 to save memory
        dtype = np.int32
        
        # Convert training data
        self.encoder_input_train = pad_sequences(
            tokens_to_sequences(train_data['tokens_fr'], self.fr_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post', dtype=dtype
        )
        self.decoder_input_train = pad_sequences(
            tokens_to_sequences(train_data['tokens_en'], self.en_word_to_idx, add_start_end=True),
            maxlen=self.max_seq_length, padding='post', truncating='post', dtype=dtype
        )
        self.decoder_target_train = pad_sequences(
            tokens_to_sequences([tokens + ['<end>'] for tokens in train_data['tokens_en']], self.en_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post', dtype=dtype
        )
        
        # Convert validation data
        self.encoder_input_val = pad_sequences(
            tokens_to_sequences(val_data['tokens_fr'], self.fr_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post', dtype=dtype
        )
        self.decoder_input_val = pad_sequences(
            tokens_to_sequences(val_data['tokens_en'], self.en_word_to_idx, add_start_end=True),
            maxlen=self.max_seq_length, padding='post', truncating='post', dtype=dtype
        )
        self.decoder_target_val = pad_sequences(
            tokens_to_sequences([tokens + ['<end>'] for tokens in val_data['tokens_en']], self.en_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post', dtype=dtype
        )
        
        # Convert test data
        self.encoder_input_test = pad_sequences(
            tokens_to_sequences(test_data['tokens_fr'], self.fr_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post', dtype=dtype
        )
        self.decoder_target_test = pad_sequences(
            tokens_to_sequences([tokens + ['<end>'] for tokens in test_data['tokens_en']], self.en_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post', dtype=dtype
        )
    
    def build_model(self):
        """
        Build optimized RNN model
        """
        print(f"Building optimized RNN model...")
        
        # Encoder
        encoder_inputs = Input(shape=(None,), name='encoder_inputs', dtype=tf.int32)
        encoder_embedding = Embedding(
            self.fr_vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        encoder_embedding = Dropout(self.dropout_rate)(encoder_embedding)
        
        # Use GRU instead of SimpleRNN for better performance
        encoder_rnn = tf.keras.layers.GRU(
            self.hidden_units,
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=0.0,  # Disable recurrent dropout for speed
            name='encoder_rnn'
        )
        _, encoder_state = encoder_rnn(encoder_embedding)
        
        # Decoder
        decoder_inputs = Input(shape=(None,), name='decoder_inputs', dtype=tf.int32)
        decoder_embedding = Embedding(
            self.en_vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)
        decoder_embedding = Dropout(self.dropout_rate)(decoder_embedding)
        
        decoder_rnn = tf.keras.layers.GRU(
            self.hidden_units, 
            return_sequences=True, 
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=0.0,  # Disable recurrent dropout for speed
            name='decoder_rnn'
        )
        decoder_outputs, _ = decoder_rnn(decoder_embedding, initial_state=encoder_state)
        
        # Dense layer for output
        decoder_dense = Dense(self.en_vocab_size, activation='softmax', name='decoder_dense', dtype=tf.float32)
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Use optimized optimizer settings
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train_model(self, epochs=50, patience=5):
        """
        Train with optimized settings
        """
        print(f"Starting training with batch size {self.batch_size}...")
        
        # Optimized callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model with optimized batch size
        history = self.model.fit(
            [self.encoder_input_train, self.decoder_input_train], 
            self.decoder_target_train,
            batch_size=self.batch_size,  # Use larger batch size
            epochs=epochs,
            validation_data=(
                [self.encoder_input_val, self.decoder_input_val], 
                self.decoder_target_val
            ),
            callbacks=callbacks,
            verbose=1,
            workers=4,  # Use multiple workers
            use_multiprocessing=False,  # Keep False for Apple Silicon
            max_queue_size=10
        )
        
        return history
    
    def evaluate_model(self):
        """
        Evaluate the model on test data
        """
        test_loss, test_accuracy = self.model.evaluate(
            [self.encoder_input_test, self.decoder_target_test[:, :-1]], 
            self.decoder_target_test[:, 1:],
            batch_size=self.batch_size,
            verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy

def optimized_grid_search_rnn(df_tokens, param_grid, validation_size=1000, test_size=0.2, 
                             epochs=15, patience=3, n_best=3):
    """
    Optimized grid search for faster experimentation
    """
    print("Starting Optimized RNN Grid Search...")
    print(f"Parameter grid: {param_grid}")
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total configurations to test: {len(param_combinations)}")
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        print('='*60)
        
        try:
            # Create optimized RNN translator
            rnn = OptimizedRNNTranslator(**params)
            
            # Prepare data
            train_data, val_data, test_data = rnn.prepare_data_from_tokens(
                df_tokens, validation_size=validation_size, test_size=test_size
            )
            
            # Build vocabularies and convert data
            rnn.build_vocabularies(train_data)
            rnn.convert_tokens_to_sequences(train_data, val_data, test_data)
            
            # Build and train model
            rnn.build_model()
            history = rnn.train_model(epochs=epochs, patience=patience)
            
            # Evaluate model
            test_loss, test_accuracy = rnn.evaluate_model()
            
            # Store results
            result = {
                'params': params.copy(),
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'val_loss': min(history.history['val_loss']),
                'val_accuracy': max(history.history['val_accuracy']),
                'epochs_trained': len(history.history['loss']),
                'total_params': rnn.model.count_params()
            }
            results.append(result)
            
            print(f"✅ Configuration {i+1} completed:")
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            print(f"   Test Loss: {test_loss:.4f}")
            print(f"   Total Parameters: {rnn.model.count_params():,}")
            
            # Clean up to free memory
            del rnn
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"❌ Configuration {i+1} failed: {str(e)}")
            continue
    
    # Sort by test accuracy (descending)
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print("OPTIMIZED GRID SEARCH RESULTS")
    print('='*80)
    
    for i, result in enumerate(results[:n_best]):
        print(f"\nRank {i+1}:")
        print(f"  Parameters: {result['params']}")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Test Loss: {result['test_loss']:.4f}")
        print(f"  Total Parameters: {result['total_params']:,}")
        print(f"  Epochs Trained: {result['epochs_trained']}")
    
    return results[:n_best]

# Optimized parameter grid for faster experimentation
optimized_param_grid = {
    'embedding_dim': [64, 128],      # Smaller embeddings
    'hidden_units': [128, 256],      # Reasonable hidden sizes
    'dropout_rate': [0.1, 0.2],
    'learning_rate': [0.001, 0.002],
    'batch_size': [128, 256]         # Larger batch sizes for efficiency
}

print("Optimized RNN Translator for Apple Silicon ready!")
print("Key optimizations:")
print("- Metal Performance Shaders (MPS) GPU acceleration")
print("- Mixed precision training")
print("- Optimized thread configuration")
print("- Larger batch sizes")
print("- GRU instead of SimpleRNN")
print("- Reduced vocabulary size")
print("- Memory-efficient data types")
print("\nUse optimized_grid_search_rnn(df_tokens, optimized_param_grid) to start!")