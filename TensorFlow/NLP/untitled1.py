import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras.preprocessing import text as tx

a = "that is a good way of seeing it"
b = tx.text_to_word_sequence(a,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ')
c = tx.one_hot(a,1000000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ')
d = tx.hashing_trick(a,1300000,"md5",filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=' ')
e = tx.Tokenizer()
f = e.fit_on_texts([a])