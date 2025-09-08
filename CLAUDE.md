# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rosetta-Stone is a French-to-English neural machine translation project using LSTM-based sequence-to-sequence models. The project consists of data preprocessing, model training, and translation capabilities.

## Development Setup

### Environment Management
- Uses UV for dependency management (Python 3.12.8)
- Install dependencies: `uv sync`
- Activate environment: `source .venv/bin/activate` or use `uv run <command>`

### Common Commands
- Run training script: `uv run python LSTM_translator.py`
- Start Jupyter notebook: `uv run jupyter notebook text_checks.ipynb`
- Install new dependency: `uv add <package_name>`

## Project Architecture

### Core Components

1. **LSTMTranslator Class** (`LSTM_translator.py`)
   - Main neural machine translation model
   - Encoder-decoder architecture with attention mechanism
   - Handles vocabulary building, sequence preparation, and model training
   - Key methods:
     - `prepare_data_from_tokens()`: Processes tokenized data
     - `build_vocabularies()`: Creates word-to-index mappings
     - `build_model()`: Constructs the LSTM seq2seq model
     - `train_model()`: Training loop with validation
     - `translate()`: Inference method

2. **Data Preprocessing** (`text_checks.ipynb`)
   - Text cleaning and tokenization for French/English pairs
   - TF-IDF vectorization for analysis
   - Custom punctuation spacing normalization
   - Outputs preprocessed data to `data/cleaned_texts.csv`

### Data Structure
- **Input datasets**: 
  - `data/small_vocab_fr.txt`: French sentences
  - `data/small_vocab_en.txt`: English sentences (parallel corpus)
- **Processed data**: `data/cleaned_texts.csv` with tokenized pairs
- **Model artifacts**: Saved vocabularies and trained models

### Key Dependencies
- TensorFlow 2.20.0+ for neural networks
- NLTK for tokenization and preprocessing  
- spaCy for advanced NLP features
- scikit-learn for data splitting and vectorization
- pandas/numpy for data manipulation

## Model Configuration

### Default Hyperparameters
- Max sequence length: 50 tokens
- Embedding dimension: 256
- LSTM hidden units: 512
- Max vocabulary size: 20,000
- Dropout rate: 0.2

### Training Process
1. Load and preprocess parallel text data
2. Build French/English vocabularies with special tokens (`<pad>`, `<unk>`, `<start>`, `<end>`)
3. Convert tokens to padded sequences
4. Train encoder-decoder LSTM with teacher forcing
5. Create inference models for translation

## Working with the Codebase

### Text Preprocessing Pipeline
The preprocessing in `text_checks.ipynb` includes:
- Punctuation spacing normalization
- Tokenization using NLTK
- Optional stopword removal
- TF-IDF analysis for vocabulary insights

### Model Training Workflow
1. Use `text_checks.ipynb` to preprocess raw text data
2. Load processed data in `LSTMTranslator`
3. Build vocabularies and prepare sequences
4. Train the seq2seq model
5. Save model weights and vocabularies for inference

### Adding New Features
- Extend `LSTMTranslator` class for new architectures
- Modify preprocessing functions in the notebook for different data formats
- Update vocabulary building for new languages or special tokens