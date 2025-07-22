# 🧠 LSTM Hands-on Project: Next-Word Prediction

This project implements a basic LSTM model using Keras to perform **next-word prediction**. It covers the complete pipeline — from text preprocessing to training and inference — giving a hands-on introduction to sequence modeling using LSTM (Long Short-Term Memory) networks.

---

## 🔧 Technologies & Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Tokenizer & Sequence Padding (Keras preprocessing)
- LSTM (Sequential Model)
- Embedding Layers

---
#🧮 Step-by-Step Implementation
## Step 1️⃣: Import Required Libraries
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

## Step 2️⃣: Tokenize the Corpus
We use Keras' Tokenizer to convert each word into a unique integer.

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # +1 for padding
word_index: Dictionary mapping words to their indices

total_words: Size of vocabulary

## Step 3️⃣: Generate N-Gram Sequences
Each sentence is converted into multiple n-gram sequences for training:

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
Example:
["AI will change the world"] → [AI will], [AI will change], ..., [AI will change the world]

## Step 4️⃣: Pad Sequences
All sequences are padded with zeros to the same length (pre-padding):

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

## Step 5️⃣: Split Features (X) and Labels (y)
The last word in each sequence is treated as the label (target), and the remaining words as input.

X = input_sequences[:, :-1]
y = input_sequences[:, -1]


## Step 6️⃣: Build the LSTM Model

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=10, input_length=max_sequence_len - 1))
model.add(LSTM(50))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
Embedding: Converts integers to dense vectors

LSTM: Learns sequence relationships

Dense: Outputs probability distribution over vocabulary

## Step 7️⃣: Train the Model
model.fit(X, y, epochs=100, verbose=1)


## Step 8️⃣: Predict the Next Words
We define a function that takes a seed text and generates next words:

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                seed_text += " " + word
                break
    return seed_text

# Example usage
print(generate_text("Data is", 2, model, max_sequence_len))
📉 LSTM Model Limitations
Struggles with long-term dependencies

Cannot parallelize training like Transformers

Slow training on large datasets

Sensitive to sequence length and padding

Outperformed by newer models like BERT, GPT, etc.


## 📚 Dataset

We use a **custom corpus** — a list of sentences related to AI, machine learning, and deep learning.

```python
corpus = [
    "AI will change the world",
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Data is the new oil",
    "Supervised learning requires labels",
    "Unsupervised learning finds patterns",
    "Reinforcement learning needs rewards",
    "Neural networks mimic the human brain",
    "Natural language processing understands text",
    "Convolutional neural networks process images",
    "LSTM handles sequences and memory",
    "Tokenization splits text into words",
    "Padding ensures equal length sequences",
    "Embeddings map words to vectors",
    "Activation functions introduce non-linearity",
    "Backpropagation updates weights in training",
    "Overfitting means the model memorized",
    "Regularization helps prevent overfitting",
    "Dropout randomly deactivates neurons",
    "Early stopping halts training early",
    "Label encoding converts categories into numbers"
]

