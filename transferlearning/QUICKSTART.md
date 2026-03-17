# Bidirectional RNN Language Model - Quick Start Guide

## Project Summary

This is a complete educational project implementing a **Bidirectional RNN (Recurrent Neural Network)** for language modeling using TensorFlow/Keras.

## What is a Bidirectional RNN?

A bidirectional RNN processes data in two directions simultaneously:
- **Forward**: Left to right through the sequence
- **Backward**: Right to left through the sequence

This allows the model to understand context from both past and future tokens, leading to better language understanding.

## Project Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script - trains the model on sample text |
| `predict.py` | Inference script - makes predictions with the trained model |
| `model.py` | Model architecture definition (Bidirectional LSTM) |
| `data_utils.py` | Data loading and preprocessing utilities |
| `requirements.txt` | Python package dependencies |
| `README.md` | Detailed documentation |

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train.py
```

Expected output:
- Creates `models/bidirectional_rnn_lm.h5` (trained model)
- Creates `results/training_history.png` (training plots)
- Shows training progress with loss and accuracy metrics

Training takes ~5-10 minutes depending on your hardware.

### Step 3: Make Predictions
```bash
python predict.py
```

Expected output:
- Tests predictions on sample texts
- Shows top predicted next words with probabilities
- Demonstrates bidirectional context understanding

## Model Architecture Overview

```
Input Sequence
    ↓
[Embedding Layer] - Maps words to 128-dim vectors
    ↓
[Bidirectional LSTM #1] - 256 units, processes both directions
    ↓
[Dropout] - Prevents overfitting
    ↓
[Bidirectional LSTM #2] - 128 units
    ↓
[Dropout]
    ↓
[Dense Layer] - 128 units with ReLU activation
    ↓
[Dropout]
    ↓
[Output Layer] - Softmax, vocabulary size
    ↓
Probability Distribution over next word
```

## Key Concepts Demonstrated

### 1. Bidirectional Processing
- LSTM processes sequences forward AND backward
- Concatenates both representations
- Gives the model past and future context

### 2. LSTM Units
- Long Short-Term Memory units solve the vanishing gradient problem
- Good for learning long-term dependencies in text

### 3. Embedding Layer
- Converts discrete word indices to continuous vectors
- Learns semantic relationships between words
- Improves model efficiency

### 4. Sequence-to-Sequence Prediction
- Takes a sequence of words as input
- Predicts the next word in the sequence
- Outputs probability distribution over vocabulary

## How to Use in Your Code

### Training a Custom Model:
```python
from model import BidirectionalLMModel
from data_utils import TextDataLoader

# Load and prepare data
loader = TextDataLoader(vocab_size=5000)
sequences = loader.prepare_text(your_text)
X, y = loader.create_training_data(sequences)

# Build and train model
model = BidirectionalLMModel(vocab_size=5000)
model.build(input_length=1)
history = model.train(X, y, epochs=30, batch_size=32)
model.save('my_model.h5')
```

### Making Predictions:
```python
from predict import TextGenerator

generator = TextGenerator(model, data_loader)
predictions, extended_text = generator.predict_next_word("hello world")
```

## Understanding the Output

When you run `predict.py`, you'll see:

```
Input: 'natural language'
Top predictions:
  Next word 1: 'processing' (0.8234)
```

This means:
- Given input "natural language"
- The model predicts "processing" as most likely next word
- With 82.34% confidence

## Advantages of This Approach

✅ **Better context understanding** - Bidirectional processing  
✅ **Simpler than Transformers** - No attention mechanisms needed  
✅ **Educational** - Great for learning RNN fundamentals  
✅ **Practical** - Works well for many NLP tasks  

## Next Steps to Improve

1. **Larger Dataset** - Train on Wikipedia or books
2. **Attention Mechanism** - Add attention layers
3. **Beam Search** - Better text generation
4. **Word Embeddings** - Use pre-trained embeddings (GloVe, FastText)
5. **Character-level Model** - Predict characters instead of words
6. **Ensemble Methods** - Combine multiple models

## Common Issues and Solutions

**Q: Training is slow**
- A: Use a GPU (CUDA/cuDNN)
- A: Reduce vocabulary size or sequence length
- A: Use smaller batch size

**Q: Low accuracy**
- A: Train for more epochs
- A: Use more training data
- A: Increase model complexity (more LSTM units)

**Q: OutOfMemory error**
- A: Reduce batch size
- A: Reduce sequence length
- A: Use smaller vocabulary

## Learning Resources

- **Bidirectional RNNs**: Schuster & Paliwal (1997)
- **LSTM**: Hochreiter & Schmidhuber (1997)
- **Keras Documentation**: https://keras.io
- **TensorFlow Documentation**: https://tensorflow.org

## Author

Created as an educational resource for learning bidirectional RNNs.

## License

MIT - Free to use and modify
