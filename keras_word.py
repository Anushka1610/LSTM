# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 19:04:39 2018

@author: srajend2
"""

# -*- coding: utf-8 -*-


from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import matplotlib.pyplot as plt

path="data/sp.txt"

'''path = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')'''
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

words = set(open(path).read().lower().split())
print("words",type(words))
print("total number of unique words",len(words))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

print("word_indices", type(word_indices), "length:",len(word_indices) )
print("indices_words", type(indices_word), "length", len(indices_word))
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 30
step = 1
sentences = []
next_words= []
sentences1 = []
list_words = []
sentences2=[]
list_words=text.lower().split()

for i in range(0, len(list_words) - maxlen, step):
    sentences2 = ' '.join(list_words[i: i + maxlen])
    sentences.append(sentences2)
    next_words.append(list_words[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')

print("DEBUG: shape is {} by {} by {}".format(len(sentences), maxlen, len(words)))
print("DEBUG: size is {}".format((len(sentences) * maxlen * len(words))))
x = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split()):
        x[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(200,return_sequences=True, input_shape=(maxlen, len(words))))
model.add(Dropout(0.3))
model.add(LSTM(200, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(len(words), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(list_words) - maxlen - 1)
    for diversity in [ 1.0]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = list_words[start_index: start_index + maxlen]
        generated += ' '.join(sentence)
        print('----- Generating with seed: "' , sentence , '"')
        sys.stdout.write(generated)

        for i in range(150):
            x_pred = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]
            '''sentence = sentence[1:] + next_word

            sys.stdout.write(next_word)
            sys.stdout.flush()'''
            generated += next_word
            del sentence[0]
            sentence.append(next_word)
            sys.stdout.write(' ')
            sys.stdout.write(next_word)
            sys.stdout.flush()

        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

history= model.fit(x, y,
          batch_size=128,
          epochs=50,validation_split=0.2,
          callbacks=[print_callback])
val_loss=history.history['val_loss']
print(history.history.keys())
loss= history.history['loss']
print(loss)


plt.figure()
plt.plot((np.exp(loss)))
plt.xlabel("Epochs")
plt.ylabel("Perplexity")
plt.title("Perplexity_Train vs Epochs")
plt.savefig("Perplexity_train_word")
plt.show()

plt.figure()
plt.plot((np.exp(val_loss)))
plt.xlabel("Epochs")
plt.ylabel("Perplexity")
plt.title("Perplexity_Validation vs Epochs")
plt.savefig("perplexity_val_word")
plt.show()
