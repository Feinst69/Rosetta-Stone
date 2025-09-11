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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')

class RNNTranslator:
    def __init__(self, max_seq_length=30, embedding_dim=128, hidden_units=256, 
                 max_vocab_size=10000, dropout_rate=0.2, learning_rate=0.001):
        """
        Initialize a simple RNN Translator for sequence-to-sequence translation
        
        Parameters:
        - max_seq_length: Maximum sequence length for padding
        - embedding_dim: Embedding dimension 
        - hidden_units: RNN hidden units
        - max_vocab_size: Maximum vocabulary size
        - dropout_rate: Dropout rate for regularization
        - learning_rate: Learning rate for optimizer
        """
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.max_vocab_size = max_vocab_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
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
        Prepare data from a DataFrame containing tokenized French and English sentences
        
        Parameters:
        - df_tokens: DataFrame with 'tokens_fr' and 'tokens_en' columns
        - validation_size: Size of validation set
        - test_size: Proportion for test set
        - random_state: Random seed for reproducibility
        """
        print("Processing tokenized data...")
        
        # Filter out very short or very long sequences
        mask = (df_tokens['tokens_fr'].str.len() >= 3) & (df_tokens['tokens_fr'].str.len() <= self.max_seq_length) & \
               (df_tokens['tokens_en'].str.len() >= 3) & (df_tokens['tokens_en'].str.len() <= self.max_seq_length)
        df_filtered = df_tokens[mask].reset_index(drop=True)
        
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
        Build French and English vocabularies from training data
        """
        print("Building vocabularies...")
        
        # Special tokens
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        
        # Build French vocabulary
        all_fr_tokens = []
        for tokens in train_data['tokens_fr']:
            all_fr_tokens.extend(tokens)
        
        fr_token_counts = pd.Series(all_fr_tokens).value_counts()
        fr_vocab = special_tokens + list(fr_token_counts.head(self.max_vocab_size - len(special_tokens)).index)
        
        self.fr_word_to_idx = {word: idx for idx, word in enumerate(fr_vocab)}
        self.fr_idx_to_word = {idx: word for word, idx in self.fr_word_to_idx.items()}
        self.fr_vocab_size = len(fr_vocab)
        
        # Build English vocabulary  
        all_en_tokens = []
        for tokens in train_data['tokens_en']:
            all_en_tokens.extend(tokens)
        
        en_token_counts = pd.Series(all_en_tokens).value_counts()
        en_vocab = special_tokens + list(en_token_counts.head(self.max_vocab_size - len(special_tokens)).index)
        
        self.en_word_to_idx = {word: idx for idx, word in enumerate(en_vocab)}
        self.en_idx_to_word = {idx: word for word, idx in self.en_word_to_idx.items()}
        self.en_vocab_size = len(en_vocab)
        
        print(f"French vocabulary size: {self.fr_vocab_size}")
        print(f"English vocabulary size: {self.en_vocab_size}")
    
    def convert_tokens_to_sequences(self, train_data, val_data, test_data):
        """
        Convert token sequences to padded integer sequences
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
        
        # Convert training data
        self.encoder_input_train = pad_sequences(
            tokens_to_sequences(train_data['tokens_fr'], self.fr_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post'
        )
        self.decoder_input_train = pad_sequences(
            tokens_to_sequences(train_data['tokens_en'], self.en_word_to_idx, add_start_end=True),
            maxlen=self.max_seq_length, padding='post', truncating='post'  
        )
        self.decoder_target_train = pad_sequences(
            tokens_to_sequences([tokens + ['<end>'] for tokens in train_data['tokens_en']], self.en_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post'
        )
        
        # Convert validation data
        self.encoder_input_val = pad_sequences(
            tokens_to_sequences(val_data['tokens_fr'], self.fr_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post'
        )
        self.decoder_input_val = pad_sequences(
            tokens_to_sequences(val_data['tokens_en'], self.en_word_to_idx, add_start_end=True),
            maxlen=self.max_seq_length, padding='post', truncating='post'
        )
        self.decoder_target_val = pad_sequences(
            tokens_to_sequences([tokens + ['<end>'] for tokens in val_data['tokens_en']], self.en_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post'
        )
        
        # Convert test data
        self.encoder_input_test = pad_sequences(
            tokens_to_sequences(test_data['tokens_fr'], self.fr_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post'
        )
        self.decoder_target_test = pad_sequences(
            tokens_to_sequences([tokens + ['<end>'] for tokens in test_data['tokens_en']], self.en_word_to_idx),
            maxlen=self.max_seq_length, padding='post', truncating='post'
        )
    
    def build_model(self):
        """
        Build a simple RNN sequence-to-sequence model
        """
        print(f"Building simple RNN model...")
        
        # Encoder
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(
            self.fr_vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        encoder_embedding = Dropout(self.dropout_rate)(encoder_embedding)
        
        encoder_rnn = SimpleRNN(
            self.hidden_units,
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name='encoder_rnn'
        )
        _, encoder_state = encoder_rnn(encoder_embedding)
        
        # Decoder
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(
            self.en_vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)
        decoder_embedding = Dropout(self.dropout_rate)(decoder_embedding)
        
        decoder_rnn = SimpleRNN(
            self.hidden_units, 
            return_sequences=True, 
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name='decoder_rnn'
        )
        decoder_outputs, _ = decoder_rnn(decoder_embedding, initial_state=encoder_state)
        
        # Dense layer for output
        decoder_dense = Dense(self.en_vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train_model(self, epochs=50, batch_size=512, patience=5):
        """
        Train the RNN model
        """
        print(f"Starting training for maximum {epochs} epochs with patience {patience}...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            [self.encoder_input_train, self.decoder_input_train], 
            self.decoder_target_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [self.encoder_input_val, self.decoder_input_val], 
                self.decoder_target_val
            ),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self):
        """
        Evaluate the model on test data
        """
        test_loss, test_accuracy = self.model.evaluate(
            [self.encoder_input_test, self.decoder_target_test[:, :-1]], 
            self.decoder_target_test[:, 1:],
            verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def build_inference_models(self):
        """
        Build separate encoder and decoder models for inference
        """
        # Encoder model
        encoder_inputs = self.model.input[0]
        encoder_embedding = self.model.get_layer('encoder_embedding')(encoder_inputs)
        encoder_embedding = Dropout(self.dropout_rate)(encoder_embedding)
        encoder_rnn = self.model.get_layer('encoder_rnn')
        _, encoder_state = encoder_rnn(encoder_embedding)
        
        self.encoder_model = Model(encoder_inputs, encoder_state)
        
        # Decoder model
        decoder_state_input = Input(shape=(self.hidden_units,))
        decoder_inputs = Input(shape=(1,))
        
        decoder_embedding = self.model.get_layer('decoder_embedding')
        decoder_rnn = self.model.get_layer('decoder_rnn') 
        decoder_dense = self.model.get_layer('decoder_dense')
        
        decoder_embedding_layer = decoder_embedding(decoder_inputs)
        decoder_embedding_layer = Dropout(self.dropout_rate)(decoder_embedding_layer)
        
        decoder_outputs, decoder_state = decoder_rnn(
            decoder_embedding_layer, initial_state=decoder_state_input
        )
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.decoder_model = Model(
            [decoder_inputs, decoder_state_input],
            [decoder_outputs, decoder_state]
        )
    
    def translate_tokens(self, fr_tokens):
        """
        Translate French tokens to English tokens
        """
        if self.encoder_model is None or self.decoder_model is None:
            self.build_inference_models()
        
        # Convert tokens to sequence
        seq = [self.fr_word_to_idx.get(token, self.fr_word_to_idx['<unk>']) for token in fr_tokens]
        input_seq = pad_sequences([seq], maxlen=self.max_seq_length, padding='post', truncating='post')
        
        # Encode the input sequence
        state_value = self.encoder_model.predict(input_seq, verbose=0)
        
        # Generate empty target sequence of length 1
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.en_word_to_idx.get('<start>', 1)
        
        # Start translation
        decoded_tokens = []
        
        for _ in range(self.max_seq_length):
            output_tokens, state_value = self.decoder_model.predict([target_seq, state_value], verbose=0)
            
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.en_idx_to_word.get(sampled_token_index, '<unk>')
            
            if sampled_word == '<end>' or sampled_word == '<pad>':
                break
                
            if sampled_word not in ['<start>', '<unk>']:
                decoded_tokens.append(sampled_word)
            
            # Update target sequence
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
        
        return decoded_tokens

def plot_training_history(history, config_params, config_num, save_dir='rnn_training_plots'):
    """
    Plot and save training history for a configuration
    
    Parameters:
    - history: Training history object
    - config_params: Configuration parameters dictionary
    - config_num: Configuration number
    - save_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title(f'Model Accuracy - Config {config_num}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title(f'Model Loss - Config {config_num}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add configuration details as text
    config_text = "\n".join([f"{k}: {v}" for k, v in config_params.items()])
    plt.figtext(0.02, 0.02, f"Config {config_num}:\n{config_text}", 
                fontsize=8, verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"config_{config_num:02d}_"
    filename += "_".join([f"{k}{v}" for k, v in config_params.items()])
    filename = filename.replace(".", "_") + ".png"
    
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Training plot saved: {filepath}")

def grid_search_rnn(df_tokens, param_grid, validation_size=1000, test_size=0.2, 
                   epochs=20, patience=3, n_best=3, save_plots=True):
    """
    Perform grid search for RNN hyperparameters
    
    Parameters:
    - df_tokens: DataFrame with tokenized French/English pairs
    - param_grid: Dictionary of parameter ranges to search
    - validation_size: Size of validation set
    - test_size: Test set proportion  
    - epochs: Maximum epochs per configuration
    - patience: Early stopping patience
    - n_best: Number of best configurations to return
    - save_plots: Whether to save training plots for each configuration
    
    Returns:
    - List of best configurations with their performance metrics
    """
    print("Starting RNN Grid Search...")
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
            # Create RNN translator with current parameters
            rnn = RNNTranslator(**params)
            
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
            
            # Save training plot if requested
            if save_plots:
                plot_training_history(history, params, i+1)
            
            # Store results
            result = {
                'params': params.copy(),
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'val_loss': min(history.history['val_loss']),
                'val_accuracy': max(history.history['val_accuracy']),
                'epochs_trained': len(history.history['loss']),
                'total_params': rnn.model.count_params(),
                'history': history.history if save_plots else None  # Store history for later analysis
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
    
    # Print summary of best results
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS SUMMARY")
    print('='*80)
    
    for i, result in enumerate(results[:n_best]):
        print(f"\nRank {i+1}:")
        print(f"  Parameters: {result['params']}")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Test Loss: {result['test_loss']:.4f}")
        print(f"  Validation Accuracy: {result['val_accuracy']:.4f}")
        print(f"  Total Parameters: {result['total_params']:,}")
        print(f"  Epochs Trained: {result['epochs_trained']}")
    
    # Create comparison plot if plots were saved
    if save_plots and results:
        create_comparison_plot(results[:n_best])
    
    return results[:n_best]

def create_comparison_plot(results, save_dir='rnn_training_plots'):
    """
    Create a comparison plot of the best configurations
    """
    if not results:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot test accuracy comparison
    configs = [f"Config {i+1}" for i in range(len(results))]
    test_accs = [r['test_accuracy'] for r in results]
    val_accs = [r['val_accuracy'] for r in results]
    
    x_pos = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, val_accs, width, label='Validation Accuracy', alpha=0.8)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test vs Validation Accuracy Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot parameter count vs accuracy
    param_counts = [r['total_params'] for r in results]
    ax2.scatter(param_counts, test_accs, s=100, alpha=0.7, label='Test Accuracy')
    ax2.scatter(param_counts, val_accs, s=100, alpha=0.7, label='Validation Accuracy')
    
    for i, (params, test_acc, val_acc) in enumerate(zip(param_counts, test_accs, val_accs)):
        ax2.annotate(f'Config {i+1}', (params, test_acc), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Total Parameters')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Parameters vs Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    filepath = os.path.join(save_dir, 'grid_search_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Comparison plot saved: {filepath}")

def test_rnn_translation(rnn, df_tokens, n_examples=5):
    """
    Test the RNN translator with sample translations
    """
    print("="*60)
    print("TESTING RNN TRANSLATIONS")
    print("="*60)
    
    for i in range(min(n_examples, len(df_tokens))):
        fr_tokens = df_tokens.iloc[i]['tokens_fr']
        true_en_tokens = df_tokens.iloc[i]['tokens_en']
        
        predicted_tokens = rnn.translate_tokens(fr_tokens)
        
        print(f"\nExample {i+1}:")
        print(f"French: {' '.join(fr_tokens)}")
        print(f"True English: {' '.join(true_en_tokens)}")
        print(f"Predicted: {' '.join(predicted_tokens) if predicted_tokens else '[No translation]'}")

# Example usage
if __name__ == "__main__":
    # Example parameter grid for grid search
    param_grid = {
        'embedding_dim': [64, 128],
        'hidden_units': [128, 256],
        'dropout_rate': [0.1, 0.2],
        'learning_rate': [0.001, 0.0001]
    }
    
    print("RNN Translator with Grid Search ready!")
    print("Use grid_search_rnn(df_tokens, param_grid) to start hyperparameter search")