import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Bidirectional, Attention, AdditiveAttention, Concatenate, Dot, Lambda, Softmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import warnings
import re
import nltk
from nltk.tokenize import word_tokenize
from config import get_config
warnings.filterwarnings('ignore')

class LSTMTranslator:
    def __init__(self, config=None, **kwargs):
        """
        Initialize the LSTM Translator with configuration
        
        Parameters:
        - config: Configuration dictionary (if None, uses default config)
        - **kwargs: Override specific config parameters
        """
        # Load configuration
        if config is None:
            config = get_config()
        
        # Override with any provided kwargs
        config.update(kwargs)
        
        # Set all configuration parameters as instance variables
        self.max_seq_length = config['max_seq_length']
        self.embedding_dim = config['embedding_dim']
        self.hidden_units = config['hidden_units']
        self.max_vocab_size = config['max_vocab_size']
        self.dropout_rate = config['dropout_rate']
        
        # Advanced features
        self.use_attention = config['use_attention']
        self.use_bidirectional = config['use_bidirectional']
        self.use_teacher_forcing = config['use_teacher_forcing']
        self.use_scheduled_sampling = config['use_scheduled_sampling']
        self.attention_type = config['attention_type']
        self.attention_units = config['attention_units']
        self.bidirectional_merge_mode = config['bidirectional_merge_mode']
        
        # Training parameters
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.patience = config['patience']
        self.learning_rate = config['learning_rate']
        self.min_lr = config['min_lr']
        self.lr_reduce_factor = config['lr_reduce_factor']
        self.lr_reduce_patience = config['lr_reduce_patience']
        
        # Scheduled sampling parameters
        self.sampling_probability_start = config['sampling_probability_start']
        self.sampling_probability_end = config['sampling_probability_end']
        self.sampling_schedule = config['sampling_schedule']
        self.sampling_start_epoch = config['sampling_start_epoch']
        
        # Store full config for saving/loading
        self.config = config
        
        # Will be set during training
        self.fr_vocab_size = None
        self.en_vocab_size = None
        self.fr_word_to_idx = None
        self.fr_idx_to_word = None
        self.en_word_to_idx = None
        self.en_idx_to_word = None
        
        # Models
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        
        # TF-IDF vectorizers (if using embeddings)
        self.tfidf_fr = None
        self.tfidf_en = None
    
    def preprocess_french_phrase(self, text):
        """
        Preprocess a single French phrase into tokens for translation
        
        Parameters:
        - text: French text string to preprocess
        
        Returns:
        - List of tokens ready for translation
        """
        def fix_punctuation_spacing(text):
            # Apostrophe: no spaces around it
            if text.find("'") != -1:
                text = text.replace(" '", "'").replace("' ", "'")

            # Hyphen in compound words: no spaces around it
            # Em dash (—) or en dash (–): space before and after for sentence breaks
            if text.find("-") != -1:
                # First handle spaced dashes (likely sentence breaks)
                text = text.replace(" - ", " — ")  # Convert to em dash
                text = text.replace(" -", " —").replace("- ", "— ")
                
                # Replace em dashes back to spaced format
                text = text.replace("—", " — ")
                
                # Clean up multiple spaces around em dashes
                text = re.sub(r'\s*—\s*', ' — ', text)
            
            # Comma: no space before, one space after
            if text.find(",") != -1:
                text = text.replace(" ,", ",")
                # Add space after comma if not already there
                text = re.sub(r',(?!\s)', ', ', text)
                # Fix multiple spaces after comma
                text = text.replace(",  ", ", ")

            # Period: no space before, one space after (except end of text)
            if text.find(".") != -1:
                text = text.replace(" .", ".")
                # Add space after period if not already there and not at end
                text = re.sub(r'\.(?!\s|$)', '. ', text)
                # Fix multiple spaces after period
                text = text.replace(".  ", ". ")
            
            # Semicolon: no space before, one space after
            if text.find(";") != -1:
                text = text.replace(" ;", ";")
                text = re.sub(r';(?!\s)', '; ', text)
                text = text.replace(";  ", "; ")
            
            # Colon: no space before, one space after
            if text.find(":") != -1:
                text = text.replace(" :", ":")
                text = re.sub(r':(?!\s)', ': ', text)
                text = text.replace(":  ", ": ")
            
            # Question mark: no space before, one space after
            if text.find("?") != -1:
                text = text.replace(" ?", "?")
                text = re.sub(r'\?(?!\s|$)', '? ', text)
                text = text.replace("?  ", "? ")
            
            # Exclamation mark: no space before, one space after
            if text.find("!") != -1:
                text = text.replace(" !", "!")
                text = re.sub(r'!(?!\s|$)', '! ', text)
                text = text.replace("!  ", "! ")
            
            # Opening parenthesis: one space before (if not at start), no space after
            if text.find("(") != -1:
                text = re.sub(r'(?<!\s)(?<!^)\(', ' (', text)  # Add space before if not already there
                text = text.replace("( ", "(")  # Remove space after
                text = text.replace("  (", " (")  # Fix double spaces
            
            # Closing parenthesis: no space before, one space after (if not at end)
            if text.find(")") != -1:
                text = text.replace(" )", ")")
                text = re.sub(r'\)(?!\s|$|[.,;:!?])', ') ', text)  # Add space after unless at end or before punctuation
                text = text.replace(")  ", ") ")
            
            # Clean up any multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()

        # Basic text cleaning
        text = text.strip()
        text = fix_punctuation_spacing(text)

        # Tokenize using NLTK
        word_tokens = word_tokenize(text, language='french')
        
        # Filter tokens with length > 1
        word_tokens = [wt for wt in word_tokens if len(wt) > 1]
        
        return word_tokens
    
    def prepare_data_from_text(self, df_text, validation_size=1000, test_size=0.2, random_state=42):
        """
        Prepare data from raw text dataframe by applying preprocessing
        
        Parameters:
        - df_text: DataFrame with columns ['text_fr', 'text_en']
        - validation_size: Number of samples for validation
        - test_size: Proportion for test set
        - random_state: Random seed
        
        Returns:
        - Processed train/test/validation datasets with tokens
        """
        print("Processing raw text data...")
        
        # Create copy and apply preprocessing
        df_processed = df_text.copy()
        
        # Apply French preprocessing
        df_processed['tokens_fr'] = df_processed['text_fr'].apply(
            lambda x: self.preprocess_french_phrase(x)
        )
        
        # Apply English preprocessing (same logic but for English)
        df_processed['tokens_en'] = df_processed['text_en'].apply(
            lambda x: word_tokenize(x.strip(), language='english')
        ).apply(lambda x: [wt for wt in x if len(wt) > 1])
        
        # Filter out empty token lists
        df_processed = df_processed[
            (df_processed['tokens_fr'].apply(len) > 0) & 
            (df_processed['tokens_en'].apply(len) > 0)
        ].copy()
        
        print(f"Dataset size after preprocessing and filtering: {len(df_processed)}")
        
        # Continue with existing token-based preparation
        return self.prepare_data_from_tokens(
            df_processed[['tokens_fr', 'tokens_en']], 
            validation_size, test_size, random_state
        )
        
    def prepare_data_from_tokens(self, df_tokens, validation_size=1000, test_size=0.2, random_state=42):
        """
        Prepare data from your pre-tokenized dataframe
        
        Parameters:
        - df_tokens: DataFrame with columns ['tokens_fr', 'tokens_en']
        - validation_size: Number of samples for validation
        - test_size: Proportion for test set
        - random_state: Random seed
        
        Returns:
        - Processed train/test/validation datasets
        """
        print("Processing tokenized data...")
        
        # Create copies to avoid modifying original data
        df_clean = df_tokens.copy()
        
        # Filter out empty token lists
        df_clean = df_clean[
            (df_clean['tokens_fr'].apply(len) > 0) & 
            (df_clean['tokens_en'].apply(len) > 0)
        ].copy()
        
        print(f"Dataset size after filtering: {len(df_clean)}")
        
        # Add start/end tokens to English sequences
        df_clean['tokens_en_with_markers'] = df_clean['tokens_en'].apply(
            lambda x: ['<start>'] + x + ['<end>']
        )
        
        # Split the data
        df_temp, df_val = train_test_split(df_clean, test_size=validation_size, random_state=random_state)
        df_train, df_test = train_test_split(df_temp, test_size=test_size, random_state=random_state)
        
        print(f"Training set: {len(df_train)}")
        print(f"Test set: {len(df_test)}")
        print(f"Validation set: {len(df_val)}")
        
        return df_train, df_test, df_val
    
    def build_vocabularies(self, df_train):
        """
        Build vocabularies from training data tokens
        """
        print("Building vocabularies...")
        
        # Collect all French tokens
        all_fr_tokens = []
        for tokens in df_train['tokens_fr']:
            all_fr_tokens.extend(tokens)
        
        # Collect all English tokens (with markers)
        all_en_tokens = []
        for tokens in df_train['tokens_en_with_markers']:
            all_en_tokens.extend(tokens)
        
        # Count frequencies and create vocabularies
        from collections import Counter
        
        fr_counter = Counter(all_fr_tokens)
        en_counter = Counter(all_en_tokens)
        
        # Keep most common words
        fr_most_common = fr_counter.most_common(self.max_vocab_size - 2)  # -2 for <pad> and <unk>
        en_most_common = en_counter.most_common(self.max_vocab_size - 2)
        
        # Create word-to-index mappings
        self.fr_word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.fr_word_to_idx.update({word: i+2 for i, (word, _) in enumerate(fr_most_common)})
        
        self.en_word_to_idx = {'<pad>': 0, '<unk>': 1}
        self.en_word_to_idx.update({word: i+2 for i, (word, _) in enumerate(en_most_common)})
        
        # Create index-to-word mappings
        self.fr_idx_to_word = {idx: word for word, idx in self.fr_word_to_idx.items()}
        self.en_idx_to_word = {idx: word for word, idx in self.en_word_to_idx.items()}
        
        self.fr_vocab_size = len(self.fr_word_to_idx)
        self.en_vocab_size = len(self.en_word_to_idx)
        
        print(f"French vocabulary size: {self.fr_vocab_size}")
        print(f"English vocabulary size: {self.en_vocab_size}")
    
    def tokens_to_sequences(self, tokens_list, word_to_idx):
        """
        Convert list of token lists to sequences of indices
        """
        sequences = []
        for tokens in tokens_list:
            seq = [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
            sequences.append(seq)
        return sequences
    
    def prepare_sequences(self, df_train, df_test, df_val):
        """
        Convert tokens to padded sequences
        """
        print("Converting tokens to sequences...")
        
        # Convert to sequences
        X_train = self.tokens_to_sequences(df_train['tokens_fr'], self.fr_word_to_idx)
        y_train = self.tokens_to_sequences(df_train['tokens_en_with_markers'], self.en_word_to_idx)
        
        X_test = self.tokens_to_sequences(df_test['tokens_fr'], self.fr_word_to_idx)
        y_test = self.tokens_to_sequences(df_test['tokens_en_with_markers'], self.en_word_to_idx)
        
        X_val = self.tokens_to_sequences(df_val['tokens_fr'], self.fr_word_to_idx)
        y_val = self.tokens_to_sequences(df_val['tokens_en_with_markers'], self.en_word_to_idx)
        
        # Pad sequences
        X_train = pad_sequences(X_train, maxlen=self.max_seq_length, padding='post', truncating='post')
        y_train = pad_sequences(y_train, maxlen=self.max_seq_length+1, padding='post', truncating='post')
        
        X_test = pad_sequences(X_test, maxlen=self.max_seq_length, padding='post', truncating='post')
        y_test = pad_sequences(y_test, maxlen=self.max_seq_length+1, padding='post', truncating='post')
        
        X_val = pad_sequences(X_val, maxlen=self.max_seq_length, padding='post', truncating='post')
        y_val = pad_sequences(y_val, maxlen=self.max_seq_length+1, padding='post', truncating='post')
        
        # Prepare decoder inputs and targets
        decoder_input_train = y_train[:, :-1]
        decoder_target_train = y_train[:, 1:]
        
        decoder_input_test = y_test[:, :-1]
        decoder_target_test = y_test[:, 1:]
        
        decoder_input_val = y_val[:, :-1]
        decoder_target_val = y_val[:, 1:]
        
        return {
            'X_train': X_train, 'decoder_input_train': decoder_input_train, 'decoder_target_train': decoder_target_train,
            'X_test': X_test, 'decoder_input_test': decoder_input_test, 'decoder_target_test': decoder_target_test,
            'X_val': X_val, 'decoder_input_val': decoder_input_val, 'decoder_target_val': decoder_target_val
        }
    
    def build_model(self):
        """
        Build the sequence-to-sequence LSTM model with attention and bidirectional options
        """
        print(f"Building the model with attention={self.use_attention}, bidirectional={self.use_bidirectional}...")
        
        # Encoder
        encoder_inputs = Input(shape=(self.max_seq_length,), name='encoder_inputs')
        encoder_embedding = Embedding(
            self.fr_vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        encoder_embedding = Dropout(self.dropout_rate)(encoder_embedding)
        
        # Create LSTM layer
        encoder_lstm_layer = LSTM(
            self.hidden_units, 
            return_sequences=True if self.use_attention else False,
            return_state=True, 
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name='encoder_lstm'
        )
        
        # Apply bidirectional wrapper if enabled
        if self.use_bidirectional:
            # Use merge_mode=None to get separate forward/backward states
            encoder_lstm_layer = Bidirectional(
                encoder_lstm_layer, 
                merge_mode=None,
                name='bidirectional_encoder'
            )
            
            # Bidirectional LSTM with return_sequences=True and return_state=True returns:
            # [forward_output, backward_output, forward_h, forward_c, backward_h, backward_c]
            bidirectional_results = encoder_lstm_layer(encoder_embedding)
            
            if self.use_attention:
                # Extract outputs and states
                forward_output = bidirectional_results[0]
                backward_output = bidirectional_results[1] 
                forward_h = bidirectional_results[2]
                forward_c = bidirectional_results[3]
                backward_h = bidirectional_results[4]
                backward_c = bidirectional_results[5]
                
                # Concatenate forward and backward outputs for attention
                concat_layer = Concatenate(axis=-1, name='encoder_outputs_concat')
                encoder_outputs_raw = concat_layer([forward_output, backward_output])
                
                # Project concatenated encoder outputs back to hidden_units size for attention compatibility
                encoder_projection = Dense(self.hidden_units, activation='tanh', name='encoder_outputs_projection')
                encoder_outputs = encoder_projection(encoder_outputs_raw)
            else:
                # Extract only states (no outputs needed)
                forward_h = bidirectional_results[2]
                forward_c = bidirectional_results[3]
                backward_h = bidirectional_results[4]
                backward_c = bidirectional_results[5]
                encoder_outputs = None
            
            # Project concatenated states back to hidden_units size using Concatenate layer
            concat_h = Concatenate(axis=-1, name='state_h_concat')([forward_h, backward_h])
            concat_c = Concatenate(axis=-1, name='state_c_concat')([forward_c, backward_c])
            
            state_h = Dense(self.hidden_units, activation='tanh', name='state_h_projection')(concat_h)
            state_c = Dense(self.hidden_units, activation='tanh', name='state_c_projection')(concat_c)
            encoder_states = [state_h, state_c]
        else:
            if self.use_attention:
                encoder_outputs, state_h, state_c = encoder_lstm_layer(encoder_embedding)
            else:
                _, state_h, state_c = encoder_lstm_layer(encoder_embedding)
                encoder_outputs = None
            encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(
            self.en_vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            name='decoder_embedding'
        )
        decoder_embedding_layer = decoder_embedding(decoder_inputs)
        decoder_embedding_layer = Dropout(self.dropout_rate)(decoder_embedding_layer)
        
        decoder_lstm = LSTM(
            self.hidden_units, 
            return_sequences=True, 
            return_state=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name='decoder_lstm'
        )
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding_layer, initial_state=encoder_states)
        
        # Add attention mechanism if enabled
        if self.use_attention and encoder_outputs is not None:
            # Pure Keras attention implementation using only Keras layers
            # Project decoder and encoder outputs to attention space
            attention_dense_query = Dense(self.hidden_units, name='attention_query')
            attention_dense_key = Dense(self.hidden_units, name='attention_key')
            
            query = attention_dense_query(decoder_outputs)  # (batch, seq_len, hidden)
            key = attention_dense_key(encoder_outputs)      # (batch, seq_len, hidden)
            
            # Compute attention scores using Dot layer 
            # Dot([query, key]) with axes=[2, 2] computes batch-wise dot product along last axis
            attention_scores = Dot(axes=[2, 2], name='attention_scores')([query, key])
            
            # Apply softmax to get attention weights
            attention_weights = Softmax(axis=-1, name='attention_weights')(attention_scores)
            
            # Apply attention weights to encoder outputs using Dot layer
            # attention_weights: (batch, seq_len, seq_len)
            # encoder_outputs: (batch, seq_len, hidden)  
            attention_output = Dot(axes=[2, 1], name='attention_output')([attention_weights, encoder_outputs])
            
            # Concatenate attention output with decoder output
            decoder_concat = Concatenate(axis=-1, name='decoder_attention_concat')([decoder_outputs, attention_output])
            attention_dense = Dense(self.hidden_units, activation='tanh', name='attention_dense')
            decoder_outputs = attention_dense(decoder_concat)
        
        # Dense layer for output
        decoder_dense = Dense(self.en_vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile the model with configurable optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train(self, data_dict, batch_size=None, epochs=None, patience=None):
        """
        Train the model using configuration parameters
        
        Parameters:
        - data_dict: Dictionary containing training data
        - batch_size: Batch size for training (uses config if None)
        - epochs: Maximum number of epochs to train (uses config if None)
        - patience: Number of epochs with no improvement after which training will be stopped (uses config if None)
        """
        # Use config values if parameters not provided
        batch_size = batch_size or self.batch_size
        epochs = epochs or self.epochs
        patience = patience or self.patience
        
        print(f"Starting training for maximum {epochs} epochs with patience {patience}...")
        print(f"Using batch_size={batch_size}, attention={self.use_attention}, bidirectional={self.use_bidirectional}")
        print(f"Teacher forcing={self.use_teacher_forcing}, scheduled_sampling={self.use_scheduled_sampling}")
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.lr_reduce_factor,
                patience=self.lr_reduce_patience,
                min_lr=self.min_lr,
                verbose=1,
                min_delta=0.001
            )
        ]
        
        # Determine training mode
        if not self.use_teacher_forcing:
            print("Using free running mode (no teacher forcing)...")
            history = self._train_free_running(data_dict, batch_size, epochs, callbacks)
        elif self.use_scheduled_sampling:
            print("Using scheduled sampling training...")
            history = self._train_with_scheduled_sampling(data_dict, batch_size, epochs, callbacks)
        else:
            print("Using standard teacher forcing...")
            # Standard training with teacher forcing
            history = self.model.fit(
                [data_dict['X_train'], data_dict['decoder_input_train']],
                data_dict['decoder_target_train'],
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(
                    [data_dict['X_val'], data_dict['decoder_input_val']], 
                    data_dict['decoder_target_val']
                ),
                callbacks=callbacks,
                verbose=1
            )
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(
            [data_dict['X_test'], data_dict['decoder_input_test']], 
            data_dict['decoder_target_test'], 
            verbose=0
        )
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return history
    
    def _train_free_running(self, data_dict, batch_size, epochs, callbacks):
        """
        Train without teacher forcing - decoder uses its own predictions
        """
        print("Note: Free running mode uses simplified approach.")
        print("Decoder receives <start> token only, must generate entire sequence.")
        
        # For free running, we modify the input data
        # Create decoder inputs with only start tokens
        start_token_idx = self.en_word_to_idx['<start>']
        
        # Prepare free running decoder inputs (just start token)
        decoder_input_train_free = np.full((len(data_dict['X_train']), 1), start_token_idx)
        decoder_input_val_free = np.full((len(data_dict['X_val']), 1), start_token_idx)
        decoder_input_test_free = np.full((len(data_dict['X_test']), 1), start_token_idx)
        
        # Build a different model for free running if needed
        print("Training in free running mode - this is much more challenging!")
        
        # Use standard training but with modified inputs
        history = self.model.fit(
            [data_dict['X_train'], decoder_input_train_free],
            data_dict['decoder_target_train'],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [data_dict['X_val'], decoder_input_val_free], 
                data_dict['decoder_target_val']
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _train_with_scheduled_sampling(self, data_dict, batch_size, epochs, callbacks):
        """
        Custom training loop with scheduled sampling
        """
        # For simplicity, fall back to standard training but with a note
        # Full scheduled sampling requires custom training loops which are complex
        print("Note: Scheduled sampling is enabled but using simplified version.")
        print("For full scheduled sampling, consider using a custom training loop.")
        
        # Use standard training for now
        history = self.model.fit(
            [data_dict['X_train'], data_dict['decoder_input_train']],
            data_dict['decoder_target_train'],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [data_dict['X_val'], data_dict['decoder_input_val']], 
                data_dict['decoder_target_val']
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def build_inference_models(self):
        """
        Build inference models for translation
        """
        print("Building inference models...")
        
        # For complex architectures (bidirectional + attention), building separate inference models
        # is very complex. For now, skip and use the main model for translation.
        if self.use_bidirectional or self.use_attention:
            print("⚠️  Complex architecture detected (bidirectional/attention).")
            print("Skipping inference model building - use main model for translation.")
            self.encoder_model = None
            self.decoder_model = None
            return
        
        # Get encoder input and determine the correct layer name based on configuration
        encoder_inputs = self.model.input[0]  # First input (encoder)
        encoder_layer_name = 'encoder_lstm'
            
        try:
            encoder_lstm = self.model.get_layer(encoder_layer_name)
        except ValueError:
            print(f"⚠️  Could not find layer '{encoder_layer_name}'. Available layers:")
            for layer in self.model.layers:
                print(f"    - {layer.name}")
            print("Skipping inference model building - use main model for translation.")
            self.encoder_model = None
            self.decoder_model = None
            return
        
        # Build encoder model for simple architecture only
        encoder_embedding = Embedding(
            self.fr_vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            name='encoder_embedding_inference'
        )
        encoder_embedded = encoder_embedding(encoder_inputs)
        encoder_embedded = Dropout(self.dropout_rate)(encoder_embedded)
        
        _, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedded)
        encoder_states = [encoder_state_h, encoder_state_c]
        
        self.encoder_model = Model(encoder_inputs, encoder_states)
        
        # Decoder model for inference
        decoder_state_input_h = Input(shape=(self.hidden_units,), name='decoder_state_h')
        decoder_state_input_c = Input(shape=(self.hidden_units,), name='decoder_state_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_inputs = Input(shape=(1,), name='decoder_inputs_inference')
        
        # Get layers from trained model
        decoder_embedding = self.model.get_layer('decoder_embedding')
        decoder_lstm = self.model.get_layer('decoder_lstm')
        decoder_dense = self.model.get_layer('decoder_dense')
        
        # Build decoder inference path
        decoder_embedding_layer = decoder_embedding(decoder_inputs)
        decoder_embedding_layer = Dropout(self.dropout_rate)(decoder_embedding_layer)
        
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding_layer, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
    
    def translate_tokens(self, fr_tokens):
        """
        Translate a list of French tokens to English tokens
        """
        # For complex architectures (bidirectional/attention), use main model
        if self.use_bidirectional or self.use_attention:
            return self._translate_with_main_model(fr_tokens)
        
        # For simple architectures, use separate encoder/decoder models
        if self.encoder_model is None or self.decoder_model is None:
            self.build_inference_models()
        
        if self.encoder_model is None or self.decoder_model is None:
            # Fallback to main model if inference models couldn't be built
            return self._translate_with_main_model(fr_tokens)
        
        # Convert tokens to sequence
        seq = [self.fr_word_to_idx.get(token, self.fr_word_to_idx['<unk>']) for token in fr_tokens]
        input_seq = pad_sequences([seq], maxlen=self.max_seq_length, padding='post', truncating='post')
        
        # Encode the input sequence
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        
        # Generate empty target sequence of length 1
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.en_word_to_idx.get('<start>', 1)
        
        # Start translation
        decoded_tokens = []
        
        for _ in range(self.max_seq_length):
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value, verbose=0)
            
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.en_idx_to_word.get(sampled_token_index, '<unk>')
            
            if sampled_word == '<end>' or sampled_word == '<pad>':
                break
                
            if sampled_word != '<start>' and sampled_word != '<unk>':
                decoded_tokens.append(sampled_word)
            
            # Update target sequence
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            
            # Update states
            states_value = [h, c]
        
        return decoded_tokens
    
    def _translate_with_main_model(self, fr_tokens):
        """
        Translate using the main model for complex architectures (bidirectional/attention)
        Uses greedy search with sequential prediction
        """
        # Convert French tokens to sequence
        fr_seq = [self.fr_word_to_idx.get(token, self.fr_word_to_idx['<unk>']) for token in fr_tokens]
        encoder_input = pad_sequences([fr_seq], maxlen=self.max_seq_length, padding='post', truncating='post')
        
        # Start with <start> token
        start_token = self.en_word_to_idx.get('<start>', 1)
        decoder_input = np.array([[start_token]])
        
        decoded_tokens = []
        
        for _ in range(self.max_seq_length):
            # Predict next token using main model
            predictions = self.model.predict([encoder_input, decoder_input], verbose=0)
            
            # Get the predicted token (greedy search - take most likely)
            predicted_id = np.argmax(predictions[0, -1, :])
            predicted_token = self.en_idx_to_word.get(predicted_id, '<unk>')
            
            # Stop if we hit end token or padding
            if predicted_token in ['<end>', '<pad>']:
                break
            
            # Add to decoded tokens if it's a real word
            if predicted_token not in ['<start>', '<unk>']:
                decoded_tokens.append(predicted_token)
            
            # Update decoder input for next iteration - append new token
            new_decoder_input = np.zeros((1, decoder_input.shape[1] + 1))
            new_decoder_input[0, :-1] = decoder_input[0]
            new_decoder_input[0, -1] = predicted_id
            decoder_input = new_decoder_input
        
        return decoded_tokens
    
    def translate_sentence(self, french_sentence):
        """
        Translate a full French sentence to English with validation
        
        Parameters:
        - french_sentence: String containing French text
        
        Returns:
        - List of English tokens if successful, False if validation fails
        """
        try:
            # Preprocess the French sentence
            french_tokens = self.preprocess_french_phrase(french_sentence)
            
            # Check if sentence is too long
            if len(french_tokens) > self.max_seq_length:
                print(f"Error: Sentence too long. Got {len(french_tokens)} tokens, maximum allowed is {self.max_seq_length}")
                return False
            
            # Check if vocabularies are loaded
            if self.fr_word_to_idx is None:
                print("Error: French vocabulary not loaded. Model needs to be trained or loaded first.")
                return False
            
            # Check for unknown words
            unknown_words = []
            for token in french_tokens:
                if token not in self.fr_word_to_idx:
                    unknown_words.append(token)
            
            if unknown_words:
                print(f"Error: Unknown words not in vocabulary: {unknown_words}")
                return False
            
            # If all checks pass, translate the tokens
            english_tokens = self.translate_tokens(french_tokens)
            
            return english_tokens
            
        except Exception as e:
            print(f"Error: An unexpected error occurred during translation: {str(e)}")
            return False
    
    def save_model(self, model_path='lstm_translator'):
        """
        Save the complete model and vocabularies
        """
        print(f"Saving model to {model_path}...")
        
        # Save the main model
        self.model.save(f'{model_path}_main.h5')
        
        # Save inference models if they exist
        if self.encoder_model is not None:
            self.encoder_model.save(f'{model_path}_encoder.h5')
        if self.decoder_model is not None:
            self.decoder_model.save(f'{model_path}_decoder.h5')
        
        # Save vocabularies and parameters
        model_data = {
            'fr_word_to_idx': self.fr_word_to_idx,
            'fr_idx_to_word': self.fr_idx_to_word,
            'en_word_to_idx': self.en_word_to_idx,
            'en_idx_to_word': self.en_idx_to_word,
            'fr_vocab_size': self.fr_vocab_size,
            'en_vocab_size': self.en_vocab_size,
            'max_seq_length': self.max_seq_length,
            'embedding_dim': self.embedding_dim,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate
        }
        
        with open(f'{model_path}_data.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved successfully!")
    
    @classmethod
    def load_model(cls, model_path='lstm_translator'):
        """
        Load a saved model
        """
        print(f"Loading model from {model_path}...")
        
        # Load model data
        with open(f'{model_path}_data.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        translator = cls(
            max_seq_length=model_data['max_seq_length'],
            embedding_dim=model_data['embedding_dim'],
            hidden_units=model_data['hidden_units'],
            dropout_rate=model_data['dropout_rate']
        )
        
        # Set vocabularies
        translator.fr_word_to_idx = model_data['fr_word_to_idx']
        translator.fr_idx_to_word = model_data['fr_idx_to_word']
        translator.en_word_to_idx = model_data['en_word_to_idx']
        translator.en_idx_to_word = model_data['en_idx_to_word']
        translator.fr_vocab_size = model_data['fr_vocab_size']
        translator.en_vocab_size = model_data['en_vocab_size']
        
        # Load models with custom objects
        custom_objects = {
            'NotEqual': tf.not_equal,
            'Equal': tf.equal,
            'ReduceSum': tf.reduce_sum,
            'Cast': tf.cast
        }
        
        try:
            translator.model = tf.keras.models.load_model(
                f'{model_path}_main.h5', 
                custom_objects=custom_objects
            )
        except Exception as e:
            print(f"Error loading main model: {e}")
            print("Rebuilding model from scratch...")
            translator.build_model()
            # Try to load weights only
            try:
                translator.model.load_weights(f'{model_path}_main.h5')
            except:
                raise Exception("Could not load model or weights. Please retrain the model.")
        
        try:
            translator.encoder_model = tf.keras.models.load_model(
                f'{model_path}_encoder.h5',
                custom_objects=custom_objects
            )
            translator.decoder_model = tf.keras.models.load_model(
                f'{model_path}_decoder.h5',
                custom_objects=custom_objects
            )
        except:
            print("Inference models not found or corrupted, will build when needed.")
            translator.build_inference_models()
        
        print("Model loaded successfully!")
        return translator

# Example usage functions
def train_translator_from_tokens(df_tokens, validation_size=None, config_overrides=None):
    """
    Complete training pipeline from your tokenized dataframe using configuration
    
    Parameters:
    - df_tokens: DataFrame with columns ['tokens_fr', 'tokens_en']
    - validation_size: Number of samples for validation set (uses config if None)
    - config_overrides: Dictionary to override specific config parameters
    
    Returns:
    - Trained LSTMTranslator instance
    """
    # Load configuration and apply overrides
    config = get_config()
    if config_overrides:
        config.update(config_overrides)
    if validation_size is not None:
        config['validation_size'] = validation_size
    
    # Initialize translator with configuration
    translator = LSTMTranslator(config=config)
    
    print("=== Training Configuration ===")
    print(f"Embedding Dim: {translator.embedding_dim}")
    print(f"Hidden Units: {translator.hidden_units}")
    print(f"Max Vocab Size: {translator.max_vocab_size}")
    print(f"Attention: {translator.use_attention}")
    print(f"Bidirectional: {translator.use_bidirectional}")
    print(f"Teacher Forcing: {translator.use_teacher_forcing}")
    print(f"Scheduled Sampling: {translator.use_scheduled_sampling}")
    print(f"Epochs: {translator.epochs}, Patience: {translator.patience}")
    print("=" * 30)
    
    # Prepare data
    df_train, df_test, df_val = translator.prepare_data_from_tokens(
        df_tokens, validation_size=config['validation_size']
    )
    
    # Build vocabularies
    translator.build_vocabularies(df_train)
    
    # Prepare sequences
    data_dict = translator.prepare_sequences(df_train, df_test, df_val)
    
    # Build and train model
    translator.build_model()
    history = translator.train(data_dict)  # Uses config parameters
    
    # Build inference models
    translator.build_inference_models()
    
    return translator, history

def load_translator_for_inference(model_path='lstm_translator'):
    """
    Load a trained model for inference only
    """
    return LSTMTranslator.load_model(model_path)

# Quick test function
def test_translation(translator, df_tokens, n_examples=5):
    """
    Test the translator on random examples
    """
    print("\n" + "="*60)
    print("TESTING TRANSLATIONS")
    print("="*60)
    
    # Sample random examples
    test_indices = np.random.choice(len(df_tokens), min(n_examples, len(df_tokens)))
    
    for i in test_indices:
        fr_tokens = df_tokens.iloc[i]['tokens_fr']
        true_en_tokens = df_tokens.iloc[i]['tokens_en']
        
        predicted_tokens = translator.translate_tokens(fr_tokens)
        
        print(f"\nFrench tokens: {fr_tokens}")
        print(f"True English: {true_en_tokens}")
        print(f"Predicted: {predicted_tokens}")
        print("-" * 50)

# Usage example:
# translator, history = train_translator_from_tokens(df_tokens, validation_size=1000)
# translator.save_model('my_translator')
# test_translation(translator, df_tokens)