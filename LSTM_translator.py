import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import warnings
import re
import nltk
from nltk.tokenize import word_tokenize
warnings.filterwarnings('ignore')

class LSTMTranslator:
    def __init__(self, max_seq_length=50, embedding_dim=256, hidden_units=512, 
                 max_vocab_size=20000, dropout_rate=0.2):
        """
        Initialize the LSTM Translator
        
        Parameters:
        - max_seq_length: Maximum sequence length for padding
        - embedding_dim: Dimension of word embeddings
        - hidden_units: Number of LSTM hidden units
        - max_vocab_size: Maximum vocabulary size
        - dropout_rate: Dropout rate for regularization
        """
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.max_vocab_size = max_vocab_size
        self.dropout_rate = dropout_rate
        
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
        Build the sequence-to-sequence LSTM model
        """
        print("Building the model...")
        
        # Encoder
        encoder_inputs = Input(shape=(self.max_seq_length,), name='encoder_inputs')
        encoder_embedding = Embedding(
            self.fr_vocab_size, 
            self.embedding_dim, 
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        encoder_embedding = Dropout(self.dropout_rate)(encoder_embedding)
        
        encoder_lstm = LSTM(
            self.hidden_units, 
            return_state=True, 
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name='encoder_lstm'
        )
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
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
        
        # Dense layer for output
        decoder_dense = Dense(self.en_vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Define the model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, data_dict, batch_size=64, epochs=3, patience=1):
        """
        Train the model
        
        Parameters:
        - data_dict: Dictionary containing training data
        - batch_size: Batch size for training
        - epochs: Maximum number of epochs to train
        - patience: Number of epochs with no improvement after which training will be stopped
        """
        print(f"Starting training for maximum {epochs} epochs with patience {patience}...")
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001  # Minimum change to qualify as an improvement
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(1, patience-1),  # Reduce LR before early stopping
                min_lr=0.0001,
                verbose=1,
                min_delta=0.001
            )
        ]
        
        # Train the model
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
    
    def build_inference_models(self):
        """
        Build inference models for translation
        """
        print("Building inference models...")
        
        # Get encoder input and LSTM layer
        encoder_inputs = self.model.input[0]  # First input (encoder)
        encoder_lstm = self.model.get_layer('encoder_lstm')
        
        # Build encoder model - need to recreate the encoder path
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
        if self.encoder_model is None or self.decoder_model is None:
            self.build_inference_models()
        
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
def train_translator_from_tokens(df_tokens, validation_size=1000):
    """
    Complete training pipeline from your tokenized dataframe
    
    Parameters:
    - df_tokens: DataFrame with columns ['tokens_fr', 'tokens_en']
    - validation_size: Number of samples for validation set
    
    Returns:
    - Trained LSTMTranslator instance
    """
    # Initialize translator
    translator = LSTMTranslator(
        max_seq_length=50,
        embedding_dim=256,
        hidden_units=512,
        dropout_rate=0.2
    )
    
    # Prepare data
    df_train, df_test, df_val = translator.prepare_data_from_tokens(
        df_tokens, validation_size=validation_size
    )
    
    # Build vocabularies
    translator.build_vocabularies(df_train)
    
    # Prepare sequences
    data_dict = translator.prepare_sequences(df_train, df_test, df_val)
    
    # Build and train model
    translator.build_model()
    history = translator.train(data_dict, batch_size=64, epochs=5, patience=2)
    
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