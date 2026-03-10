import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, SimpleRNN, Dense
from tensorflow.keras.models import Sequential

# Load dataset
with open("../dataset/text_data.txt", "r") as file:
    text = file.read().split("\n")

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []

for line in text:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

# Padding
max_len = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding="pre")

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Model
model = Sequential([
    Embedding(total_words, 64, input_length=max_len-1),
    Bidirectional(SimpleRNN(64, return_sequences=True)),
    Bidirectional(SimpleRNN(32)),
    Dense(total_words, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X, y, epochs=200, verbose=1)

# Test prediction
seed_text = "machine learning"

token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=max_len-1, padding="pre")

prediction = model.predict(token_list)
predicted_word = tokenizer.index_word[np.argmax(prediction)]

print("Next word prediction:", predicted_word)