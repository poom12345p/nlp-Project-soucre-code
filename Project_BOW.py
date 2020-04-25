import csv
import numpy as np
import deepcut
from gensim.models import Word2Vec
from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, GRU, LSTM, Dropout, Bidirectional
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import confusion_matrix

#------------------------- Read data ------------------------------
file = open('input_pre.txt', 'r',encoding = 'utf-8-sig')
data = np.asarray([[x for x in line.split('::')] for line in file.readlines()])
sentences_ar = [d[1] for d in data]
sentences_ar = [s.replace('\n','') for s in sentences_ar]
words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences_ar]

sentencesLabels = [int(d[0]) for d in data]
shuffle(sentencesLabels)

sentences=[]

file = open('ans_pre.txt', 'r',encoding = 'utf-8-sig')
data2 = np.asarray([[x for x in line.split('::')] for line in file.readlines()])
labelsS = [d[1] for d in data2]
labelsS  = [s.replace('\n','') for s in labelsS ]

labels_ar = []
for l in labelsS:
        if l == 'H':
            labels_ar.append(0)
        elif l == 'M':
            labels_ar.append(1)
        elif l== 'P':
            labels_ar.append(2)

labels= []

for i in sentencesLabels:
    sentences.append(sentences_ar[(i-1)])
    labels.append(labels_ar[(i-1)])

max_sentence_length = max([len(s) for s in words])
words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
#---------------------------------------------------------------------------------

#wvmodel = Word2Vec(words, size=word_vector_length, window=5, min_count=1, sg=1)
vocab = set([w for s in words for w in s])

pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
count = 0
vocab_vec = {}
for line in pretrained_word_vec_file:
    if count > 0:
        line = line.split()
        if(line[0] in vocab):
            vocab_vec[line[0]] = line[1:]
    count = count + 1
#----------------------Extract bag-of-words--------------------------------------------
word_vectors = np.zeros((len(words),len(vocab)))
for i in range(0,len(words)):
    count = 0
    for j in range(0,len(words[i])):
        k = 0
        for w in vocab:
            if(words[i][j] == w):
                word_vectors[i][k] = word_vectors[i][k]+1
                count = count+1
            k = k+1
    word_vectors[i] = word_vectors[i]/count

    inputLayer = Input(shape=(len(vocab),))
h1 = Dense(64, activation='tanh')(inputLayer)
h2 = Dense(64, activation='tanh')(h1)
outputLayer = Dense(3, activation='softmax')(h2)
model = Model(inputs=inputLayer, outputs=outputLayer)
#-----------------------------------------------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(word_vectors, to_categorical(labels), epochs=120, batch_size=50, validation_split = 0.2)

model.save('Project_model_Bow.h5')
#------------------Evaluate by test set---------------------------  


file = open('input.txt', 'r',encoding = 'utf-8-sig')
data = np.asarray([[x for x in line.split('::')] for line in file.readlines()])
sentences_ar = [d[1] for d in data]
sentences_ar = [s.replace('\n','') for s in sentences_ar]
words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences_ar]

word_vectors2 = np.zeros((len(words),len(vocab)))
for i in range(0,len(words)):
    count = 0
    for j in range(0,len(words[i])):
        k = 0
        for w in vocab:
            if(words[i][j] == w):
                word_vectors2[i][k] = word_vectors2[i][k]+1
                count = count+1
            k = k+1
    word_vectors2[i] = word_vectors2[i]/count


y_pred = model.predict(word_vectors2)


print(y_pred)
# file = open('ans2.txt', 'r',encoding = 'utf-8-sig')
# data2 = np.asarray([[x for x in line.split('::')] for line in file.readlines()])
# labelsS = [d[1] for d in data2]
# labelsS  = [s.replace('\n','') for s in labelsS ]
# labels_ar = []
# for l in labelsS:
#         if l == 'H':
#             labels_ar.append(0)
#         elif l == 'M':
#             labels_ar.append(1)
#         elif l== 'P':
#             labels_ar.append(2)

file = open('ans.txt', 'w',encoding = 'utf-8-sig')
for i in range(0, len(y_pred)):
    #max_value = max(y_pred[i])
    max_index = y_pred[i].argmax()
    #print(max_index)
    S = ''
    if  max_index == 0:
            S='H'
           # print('append 0')
    elif  max_index == 1:
           S='M'
           # print('append 1')
    elif  max_index== 2:
            S='P'
           # print('append 2')

    file.write("{0}::{1}\n" .format((i+1),(S)))
file.close()

# cm = confusion_matrix(labels_ar, y_pred.argmax(axis=1))
# print('Confusion Matrix')
# print(cm)
# score = model.evaluate(word_vectors , to_categorical(labels_ar))

# print(score)