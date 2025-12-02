# ULTRA SIMPLE TEXT GENERATOR
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("="*50)
print("ULTRA SIMPLE TEXT GENERATOR")
print("="*50)

# 1. Simple text
text = "hello world hello universe hello everyone "
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

print(f"Unique characters: {chars}")
print(f"Text length: {len(text)}")

# 2. Prepare data
maxlen = 5  # Sequence length
step = 1    # Step size
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print(f"\nCreated {len(sentences)} sequences")
print(f"Example: '{sentences[0]}' -> '{next_chars[0]}'")

# 3. Vectorize
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# 4. Simple model
model = keras.Sequential([
    keras.layers.LSTM(32, input_shape=(maxlen, len(chars))),
    keras.layers.Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# 5. Train quickly
model.fit(X, y, batch_size=16, epochs=50, verbose=0)

# 6. Generate text
def generate_simple(start_text, length=20):
    generated = start_text
    for _ in range(length):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated[-maxlen:]):
            x[0, t, char_to_idx[char]] = 1
        
        preds = model.predict(x, verbose=0)[0]
        next_idx = np.argmax(preds)
        next_char = idx_to_char[next_idx]
        generated += next_char
    
    return generated

# 7. Test
print("\n" + "="*50)
print("GENERATED TEXT EXAMPLES")
print("="*50)

print(f"\nStarting with 'hello':")
print(generate_simple("hello", 30))

print(f"\nStarting with 'world':")
print(generate_simple("world", 30))

print("\n✓ Text generator working!")