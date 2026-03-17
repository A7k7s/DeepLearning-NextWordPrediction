"""Inference and text generation using the trained bidirectional RNN model."""

import numpy as np
import os
from data_utils import TextDataLoader
from model import BidirectionalLMModel
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextGenerator:
    """Generate text using the bidirectional RNN model."""
    
    def __init__(self, model, data_loader):
        """
        Initialize the text generator.
        
        Args:
            model: Trained BidirectionalLMModel
            data_loader: TextDataLoader with fitted tokenizer
        """
        self.model = model
        self.data_loader = data_loader
    
    def predict_next_word(self, input_text, num_predictions=1):
        """
        Predict the next word(s) given input text.
        
        Args:
            input_text: Input text string
            num_predictions: Number of next words to predict
            
        Returns:
            predictions: List of predicted words with probabilities
        """
        predictions = []
        current_text = input_text
        
        for _ in range(num_predictions):
            # Tokenize input text
            encoded = self.data_loader.tokenizer.texts_to_sequences([current_text])
            padded = pad_sequences(
                encoded,
                maxlen=self.data_loader.max_seq_length,
                padding='post'
            )
            
            # Get prediction
            pred_probs = self.model.predict(padded, verbose=0)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(pred_probs)[-3:][::-1]
            top_words = []
            
            for idx in top_indices:
                if idx > 0:  # Skip padding index
                    word = None
                    for w, i in self.data_loader.word_index.items():
                        if i == idx:
                            word = w
                            break
                    if word:
                        top_words.append({
                            'word': word,
                            'probability': float(pred_probs[idx])
                        })
            
            predictions.append(top_words)
            
            # Use the most likely word for next iteration
            if top_words:
                current_text += ' ' + top_words[0]['word']
        
        return predictions, current_text
    
    def analyze_bidirectional_processing(self, input_text):
        """
        Demonstrate bidirectional context processing.
        
        Args:
            input_text: Input text to analyze
        """
        encoded = self.data_loader.tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(
            encoded,
            maxlen=self.data_loader.max_seq_length,
            padding='post'
        )
        
        print("\nBidirectional RNN Analysis")
        print("=" * 60)
        print(f"Input text: '{input_text}'")
        print(f"Encoded sequence: {padded[0][:len(padded[0]) - np.sum(padded[0] == 0)]}")
        print("\nBidirectional RNN processes:")
        print("1. Forward direction: Left to right")
        print("2. Backward direction: Right to left")
        print("3. Concatenates both representations for context")
        
        # Get predictions
        pred_probs = self.model.predict(padded, verbose=0)[0]
        top_5_indices = np.argsort(pred_probs)[-5:][::-1]
        
        print("\nTop 5 predicted next words:")
        for rank, idx in enumerate(top_5_indices, 1):
            if idx > 0:
                for word, i in self.data_loader.word_index.items():
                    if i == idx:
                        print(f"{rank}. '{word}' - Probability: {pred_probs[idx]:.4f}")
                        break


def main():
    """Main inference function."""
    
    print("=" * 60)
    print("Bidirectional RNN Language Model - Inference")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('models/bidirectional_rnn_lm.h5'):
        print("\nError: Model not found. Please train the model first using train.py")
        return
    
    # Initialize model and data loader
    print("\n1. Loading model and tokenizer...")
    model = BidirectionalLMModel(vocab_size=5000)
    model.load('models/bidirectional_rnn_lm.h5')
    
    # Recreate tokenizer (in production, you'd save this separately)
    data_loader = TextDataLoader(vocab_size=5000, max_seq_length=50)
    
    # Prepare sample text to fit tokenizer
    sample_text = """
    Natural language processing is a fascinating field of artificial intelligence.
    It deals with the interaction between computers and human languages.
    Deep learning has revolutionized the field of natural language processing.
    Recurrent neural networks are powerful for sequential data processing.
    Bidirectional RNNs process sequences in both forward and backward directions.
    """
    data_loader.prepare_text(sample_text)
    
    # Initialize text generator
    generator = TextGenerator(model, data_loader)
    
    # Test predictions
    print("\n2. Testing predictions...")
    test_inputs = [
        "natural language",
        "deep learning",
        "neural networks",
        "recurrent"
    ]
    
    for test_input in test_inputs:
        predictions, extended_text = generator.predict_next_word(test_input, num_predictions=2)
        print(f"\nInput: '{test_input}'")
        print(f"Top predictions:")
        for i, pred_list in enumerate(predictions):
            if pred_list:
                print(f"  Next word {i+1}: '{pred_list[0]['word']}' ({pred_list[0]['probability']:.4f})")
    
    # Analyze bidirectional processing
    print("\n3. Demonstrating bidirectional processing...")
    generator.analyze_bidirectional_processing("language processing")
    
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
