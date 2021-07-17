# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensroflow.keras import layers

from numpy import array
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layrs import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# source text
data = """ Piford Technologies is a leading Software Development Comany\n
Piford Technologies provide trainings to working professionals and students\n
We are product based and service based company\n
we have one of our office in IT Park, Mohali\n """

# integer encode text
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([data])
encoded_data = tokenizer.texts_to_sequences([data])[0]
encoded_data

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) +1 # 0 is reserved for padding so that's why we added 1

print('Vocabulary Size: %d' % vocab_size)

# create word -> word sequences
sequences = list()
for i in range(1,len(encoded_data)):
    sequence = encoded_data[i-1:i+1]
    sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))
#split into x and y elements

sequences
# sequences[:5] # [input,output]

sequences = array(sequences)
X, y = sequences[:,0],sequences[:,1]

X[:5]
y[:5]

# oen hot encode outputs
y = to_categorical(y, num_classes=vocab_size)
# define model
y[:5]

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(X, y, epochs=100)

# generate a sequence from the model
def generate_seq(model, tokenizer, enter_text, n_pred):
    in_text, result = enter_text, enter_text
    # generate a fixed number of words
    for _ in range(n_pred):
        
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = array(encoded)
        
        # predict a word in teh vocabulary
        yhat = model.predict_classes(encoded)
        
        # map predicted wrod index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text, result = out_word, result + '' + out_word
    return result

# evaluate
print(generate_seq(model, tokenizer, 'Piford', 6))

# evaluate
print(generate_seq(model, tokenizer, 'service', 6))


        