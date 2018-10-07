# -*- coding: utf-8 -*-
"""
This is a model training program to train subject classification
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import concatenate
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, BatchNormalization
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import jieba
import codecs
import pickle as pkl
import matplotlib.pyplot as plt

def cut(string, stop_words=None):
    """
    分词
    :param string: 待分词的句子
    :return: 分词所得的列表
    """
    words = list(jieba.cut(string.strip()))
    words_return = []
    if stop_words:
        for word in words:
            if word not in stop_words:
                words_return.append(word)
    return words_return

def get_stop_word_list(filename="../data/hlt_stop_words.txt"):
    """
    返回停词表
    :param filename: 停词表位置
    :return: <List> 停词表
    """
    stop_words = []
    with codecs.open(filename, "r", "utf=8") as stop_word_file:
        for line in stop_word_file:
            stop_words.append(line.strip())
    return stop_words

def f1_score_metrics(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def pkl_write(obj, filename):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)

def pkl_read(filename):
    with open(filename, 'rb') as f:
        model = pkl.load(f)
    if model:
        return model
    else:
        raise Exception('model load error')

try:
    train_data = pd.read_pickle('./../output/data/train_data_after_cut.pkl')
    test_data = pd.read_pickle('./../output/data/test_data_after_cut.pkl')

except Exception as e:
    print(e)
    train_data = pd.read_csv('./../data/train/train.csv')
    test_data = pd.read_csv('./../data/test_public/test_public.csv')

    stop_words = get_stop_word_list()
    cut_words_train = []
    for ind in train_data.index:
        sentence = train_data.loc[ind, "content"]
        words = cut(sentence, stop_words=stop_words)
        cut_words_train.append(words)
        train_data.loc[ind, "content"] = " ".join(words)
        print("\rProcess: {:5d}/{:5d}".format(ind, train_data.shape[0]-1), end="")

    print('\n')

    cut_words_test = []
    for ind in test_data.index:
        sentence = test_data.loc[ind, "content"]
        words = cut(sentence, stop_words=stop_words)
        cut_words_test.append(words)
        test_data.loc[ind, "content"] = " ".join(words)
        print("\rProcess: {:5d}/{:5d}".format(ind, test_data.shape[0]-1), end="")

    pd.to_pickle(train_data, './../output/data/train_data_after_cut.pkl')
    pd.to_pickle(test_data, './../output/data/test_data_after_cut.pkl')

X_train, X_test,  y_train, y_test = train_test_split(train_data.content.values, 
                                                     train_data.subject.values,
                                                     test_size=0.2,
                                                     random_state=10)
# y_labels = list(y_train.value_counts().index)
# le = LabelEncoder()
# le.fit(y_labels)
# num_labels = len(y_labels)
# y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_classes=num_labels)
# y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_classes=num_labels)

le = LabelEncoder()
le.fit(train_data.subject.values)
num_labels = len(le.classes_)
y_train = to_categorical(le.transform(y_train), num_classes=num_labels)
y_test = to_categorical(le.transform(y_test), num_classes=num_labels)
print("number of labels: ", num_labels)

tokenizer = Tokenizer(split=" ")
tokenizer.fit_on_texts(train_data.content.values)
vocab = tokenizer.word_index

X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)
# # One-hot
# x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
# x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')
# # 序列模式
x_train = pad_sequences(X_train_word_ids, maxlen=64)
x_test = pad_sequences(X_test_word_ids, maxlen=64)

main_input = Input(shape=(64,), dtype='float64')
embedder = Embedding(len(vocab) + 1, 256, input_length = 64)
embed = embedder(main_input)

block1 = Convolution1D(128, 1, padding='same')(embed)

conv2_1 = Convolution1D(256, 1, padding='same')(embed)
bn2_1 = BatchNormalization()(conv2_1)
relu2_1 = Activation('relu')(bn2_1)
block2 = Convolution1D(128, 3, padding='same')(relu2_1)

conv3_1 = Convolution1D(256, 3, padding='same')(embed)
bn3_1 = BatchNormalization()(conv3_1)
relu3_1 = Activation('relu')(bn3_1)
block3 = Convolution1D(128, 5, padding='same')(relu3_1)

block4 = Convolution1D(128, 3, padding='same')(embed)

inception = concatenate([block1, block2, block3, block4], axis=-1)

flat = Flatten()(inception)
fc = Dense(128)(flat)
drop = Dropout(0.5)(fc)
bn = BatchNormalization()(drop)
relu = Activation('relu')(bn)
main_output = Dense(10, activation='softmax')(relu)
model = Model(inputs = main_input, outputs = main_output)

# model = Sequential()
# model.add(Embedding(len(vocab)+1, 300, input_length=20))
# model.add(LSTM(256, dropout=0.6, recurrent_dropout=0.5))
# model.add(Dense(num_labels, activation='softmax'))

# model = Sequential()
# model.add(Embedding(len(vocab)+1, 300, input_length=500))
# model.add(Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
# model.add(Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1)))
# model.add(Dense(num_labels, activation='softmax'))

# model = Sequential()
# model.add(Dense(512, input_shape=(len(vocab)+1, ), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_labels, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', f1_score_metrics]
)
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=25,
    validation_data=(x_test, y_test),
    callbacks=[TensorBoard(log_dir='./tmp/logs/keras_inception_logs/')]
)

model.save('./../output/model/inception_keras.h5')
# pkl_write(model, './../output/model/mlp_keras.pkl')

def plot_acc_loss(history):
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="g", label="Train")
    plt.plot(history.history["val_acc"], color="b", label="Test")
    plt.legend(loc="best")
    
    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Test")
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.show()

plot_acc_loss(history)