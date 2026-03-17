"""Training script for bidirectional RNN language model."""

import os
import numpy as np
from data_utils import TextDataLoader
from model import BidirectionalLMModel
import matplotlib.pyplot as plt


def train_language_model():
    """Train the bidirectional RNN language model."""
    
    print("=" * 60)
    print("Bidirectional RNN Language Model Training")
    print("=" * 60)
    
    # Configuration
    VOCAB_SIZE = 5000
    MAX_SEQ_LENGTH = 50
    EMBEDDING_DIM = 128
    LSTM_UNITS = 256
    EPOCHS = 30
    BATCH_SIZE = 32
    
    # Create sample text data
    sample_text = """
    Natural language processing is a fascinating field of artificial intelligence.
    It deals with the interaction between computers and human languages.
    Deep learning has revolutionized the field of natural language processing.
    Recurrent neural networks are powerful for sequential data processing.
    Bidirectional RNNs process sequences in both forward and backward directions.
    This allows the model to have context from both past and future tokens.
    Language models are trained to predict the next word in a sequence.
    They can be used for various applications like machine translation and text generation.
    The transformer architecture has become very popular recently.
    It uses attention mechanisms instead of recurrent layers.
    However, RNNs and LSTMs are still widely used in many applications.
    Text data needs to be tokenized and encoded before feeding to neural networks.
    Embedding layers map discrete tokens to continuous vectors.
    These embeddings capture semantic relationships between words.
    Training neural networks requires careful tuning of hyperparameters.
    Validation data helps us monitor model performance during training.
    """
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    with open('data/sample_text.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print("\n1. Loading and preparing data...")
    # Initialize data loader
    data_loader = TextDataLoader(vocab_size=VOCAB_SIZE, max_seq_length=MAX_SEQ_LENGTH)
    
    # Prepare text
    sequences = data_loader.prepare_text(sample_text)
    print(f"   - Loaded {len(sequences)} sequences")
    print(f"   - Vocabulary size: {len(data_loader.word_index)}")
    
    # Create training data
    X, y = data_loader.create_training_data(sequences, lookback=1)
    print(f"   - Created {len(X)} training samples")
    
    print("\n2. Building model...")
    # Build model
    model = BidirectionalLMModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        dropout_rate=0.3
    )
    model.build(input_length=1)
    model.summary()
    
    print("\n3. Training model...")
    # Train model
    history = model.train(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    
    print("\n4. Saving model...")
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/bidirectional_rnn_lm.h5')
    
    print("\n5. Plotting training history...")
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300)
    print("   - Saved training plots to results/training_history.png")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return model, data_loader, X, y


if __name__ == '__main__':
    model, data_loader, X, y = train_language_model()
