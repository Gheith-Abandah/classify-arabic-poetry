from __future__ import print_function
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Embedding
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re

print('Experiment: Classify poems, batch size 64')

batch_size = 64 # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.
print('Bach size', batch_size)
print('Epochs', epochs)
print('Latent dim', latent_dim)

data_path = 'APCD_plus_porse_all.csv'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[: len(lines) - 1]:
    input_text, target_text = re.split(',', line)
    for shatter in input_text.split('¤'):
        input_texts.append(shatter.strip())
        target_texts.append(target_text)
        for char in shatter:
            if char not in input_characters:
                input_characters.add(char)

print('Number of samples:', len(input_texts))
max_seq_length = max([len(txt) for txt in input_texts])
print('Max sequence length:', max_seq_length)

input_characters = sorted(list(input_characters))
num_tokens = len(input_characters)
print('Number of tokens:', num_tokens)

input_token_index = dict( [(char, i) for i, char in enumerate(input_characters)])

input_data = np.zeros((len(input_texts), max_seq_length), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        input_data[i, t] = input_token_index[char] + 1.

encoder=OneHotEncoder(sparse=False)
out= np.array(target_texts).reshape(-1, 1)
output_data=encoder.fit_transform(out)
classes = output_data.shape[1]
print('Number of classes:', classes)

model = Sequential()
model.add(Embedding(num_tokens+1, 32, input_length=max_seq_length, mask_zero=True))
model.add(Bidirectional(LSTM(latent_dim, input_shape=(None,num_tokens), return_sequences=True,
            dropout=0.1, recurrent_dropout=0.3),
            merge_mode='concat'))
model.add(Bidirectional(LSTM(latent_dim, return_sequences=True,
            dropout=0.1, recurrent_dropout=0.3),
            merge_mode='concat'))
model.add(Bidirectional(LSTM(latent_dim, return_sequences=True,
            dropout=0.1, recurrent_dropout=0.3),
            merge_mode='concat'))
model.add(Bidirectional(LSTM(latent_dim,
            dropout=0.1, recurrent_dropout=0.3),
            merge_mode='concat'))
model.add(Dense(output_data.shape[1], activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = "training/cp.ckpt"
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5),
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy'),
]

test_samples = 163917

print('Train samples', len(input_texts)-test_samples)
print('Test samples', test_samples)

history = model.fit(input_data[test_samples:], output_data[test_samples:], batch_size=batch_size, epochs=epochs,
                    validation_split=0.15, verbose=1, callbacks=callbacks_list)

model.load_weights(checkpoint_path)

scores = model.evaluate(input_data[0:test_samples], output_data[0:test_samples])
print('Loss on the test set', scores[0])
print('Accuracy on the test set', scores[1])
