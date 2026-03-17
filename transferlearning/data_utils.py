"""Data utilities for text preprocessing and loading."""

import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextDataLoader:
    """Load and preprocess text data for language modeling."""
    
    def __init__(self, vocab_size=5000, max_seq_length=50):
        """
        Initialize the data loader.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_seq_length: Maximum sequence length for padding
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer = None
        self.word_index = None
        
    def load_text(self, file_path):
        """Load text from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    def prepare_text(self, text):
        """
        Prepare text by tokenizing and creating sequences.
        
        Args:
            text: Raw text string
            
        Returns:
            encoded_sequences: List of encoded sequences
            tokenizer: Fitted tokenizer object
        """
        # Convert to lowercase and clean
        text = text.lower()
        
        # Split into sentences
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Tokenize
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<unk>')
        self.tokenizer.fit_on_texts(sentences)
        self.word_index = self.tokenizer.word_index
        
        # Encode sequences
        encoded_sequences = self.tokenizer.texts_to_sequences(sentences)
        
        # Pad sequences
        padded_sequences = pad_sequences(
            encoded_sequences,
            maxlen=self.max_seq_length,
            padding='post'
        )
        
        return padded_sequences
    
    def create_training_data(self, sequences, lookback=1):
        """
        Create training data from sequences.
        
        Args:
            sequences: Padded sequences
            lookback: Number of previous time steps to use as input
            
        Returns:
            X, y: Training features and labels
        """
        X, y = [], []
        
        for seq in sequences:
            for i in range(len(seq) - lookback):
                X.append(seq[i:i + lookback])
                y.append(seq[i + lookback])
        
        return np.array(X), np.array(y)
    
    def decode_sequence(self, sequence):
        """Decode a sequence back to text."""
        reverse_word_index = {v: k for k, v in self.word_index.items()}
        words = [reverse_word_index.get(i, '<unk>') for i in sequence if i != 0]
        return ' '.join(words)
