# Bidirectional RNN Language Model

A simple educational project demonstrating language modeling using bidirectional RNNs (specifically, bidirectional LSTMs).

## Project Overview

This project implements a bidirectional RNN-based language model that:
- **Processes text bidirectionally**: Reads sequences in both forward and backward directions
- **Captures context**: Combines information from past and future tokens
- **Predicts next words**: Estimates the probability of the next word in a sequence
- **Demonstrates LSTM capabilities**: Uses long short-term memory networks for better sequence learning

## Architecture

### Model Components:

```
Input Layer
    ↓
Embedding Layer (128 dims)
    ↓
Bidirectional LSTM (256 units, return sequences)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (128 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (Softmax, vocab_size)
```

### Key Features:

1. **Bidirectional Processing**: 
   - Forward LSTM: Processes input left to right
   - Backward LSTM: Processes input right to left
   - Concatenates both representations for richer context

2. **Embedding Layer**: 
   - Converts tokens to dense vectors (128-dimensional)
   - Learns semantic relationships between words

3. **Stacked LSTMs**: 
   - Two bidirectional LSTM layers for deeper learning
   - Return sequences in first layer for multi-step processing

4. **Dropout Regularization**: 
   - Prevents overfitting
   - Applied after embedding and each LSTM layer

## Installation

### Requirements:
- Python 3.8+
- TensorFlow 2.14+
- NumPy
- Matplotlib (for visualization)
- Pandas
- scikit-learn

### Setup:

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

```bash
python train.py
```

This will:
- Load sample text data
- Prepare sequences for training
- Build and compile the bidirectional RNN model
- Train for 30 epochs
- Save the trained model to `models/bidirectional_rnn_lm.h5`
- Generate training history plots

### 2. Making Predictions

```bash
python predict.py
```

This will:
- Load the trained model
- Make predictions on sample inputs
- Show top predicted next words with probabilities
- Demonstrate bidirectional context understanding

## File Structure

```
bidirectional_rnn_lm/
├── train.py              # Training script
├── predict.py            # Inference and prediction script
├── model.py              # Bidirectional RNN model definition
├── data_utils.py         # Data loading and preprocessing utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Input text data
├── models/              # Trained model checkpoints
└── results/             # Training plots and results
```

## How Bidirectional RNNs Work

### Forward Pass (Left to Right):
```
Input: "The quick brown fox"
       ↓     ↓      ↓     ↓
LSTM→ LSTM → LSTM → LSTM
(processes words sequentially left to right)
```

### Backward Pass (Right to Left):
```
Input: "The quick brown fox"
       ↓     ↓      ↓     ↓
LSTM ← LSTM ← LSTM ← LSTM
(processes words sequentially right to left)
```

### Concatenation:
The outputs from both directions are concatenated, giving the model information about:
- **Past context**: What came before (from forward LSTM)
- **Future context**: What comes after (from backward LSTM)

This is particularly useful for tasks like:
- Named entity recognition
- Part-of-speech tagging
- Text classification
- Language modeling

## Model Training Details

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam (learning rate: 0.001)
- **Metrics**: Accuracy
- **Epochs**: 30
- **Batch Size**: 32
- **Validation Split**: 20%

## Results

After training, the model learns to:
1. Understand semantic relationships between words
2. Predict plausible next words based on context
3. Leverage bidirectional information for better predictions

Training plots are saved to `results/training_history.png` showing:
- Training and validation loss curves
- Training and validation accuracy curves

## Key Advantages of Bidirectional RNNs

1. **Richer Context**: Accesses information from both directions
2. **Better Performance**: Often outperforms unidirectional RNNs on sequence tasks
3. **Semantic Understanding**: Better captures word relationships
4. **Improved Predictions**: More accurate next-word predictions
5. **Suitable for NLP**: Natural fit for language understanding tasks

## Limitations and Considerations

1. **Cannot predict in real-time**: Requires full sequences (not suitable for live streaming)
2. **Higher computational cost**: Processes data twice (forward and backward)
3. **More parameters**: Bidirectional layers have 2x the parameters
4. **Memory requirements**: Needs to store activations for both directions

## Extensions and Improvements

Potential enhancements:
- Implement attention mechanisms
- Use pre-trained embeddings (Word2Vec, GloVe)
- Add variational dropout
- Implement beam search for better text generation
- Increase vocabulary size and training data
- Fine-tune on domain-specific text
- Implement character-level modeling

## References

- Hochreiter & Schmidhuber (1997) - LSTM paper
- Schuster & Paliwal (1997) - Bidirectional RNN paper
- TensorFlow/Keras documentation
- Deep Learning (Goodfellow, Bengio, Courville)

## Author

Created as an educational project on bidirectional RNNs for language modeling.

## License

MIT License - Feel free to use and modify for educational purposes.
