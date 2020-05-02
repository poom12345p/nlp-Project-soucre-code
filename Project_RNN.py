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
sentences=[]
words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences_ar]
sentencesLabels = [int(d[0]) for d in data]
shuffle(sentencesLabels)
# for d in data:
#     print(d)

# sentencesLabels = [int(d[0]) for d in data]
#print(len(sentencesLabels))
print(sentences_ar[2362])
print(sentencesLabels[0])

file = open('ans_pre.txt', 'r',encoding = 'utf-8-sig')
data2 = np.asarray([[x for x in line.split('::')] for line in file.readlines()])
labelsS = [d[1] for d in data2]
labelsS  = [s.replace('\n','') for s in labelsS ]
labels_ar = []
labels= []
for l in labelsS:
        if l == 'H':
            labels_ar.append(0)
           # print('append 0')
        elif l == 'M':
            labels_ar.append(1)
           # print('append 1')
        elif l== 'P':
            labels_ar.append(2)
           # print('append 2')


for i in sentencesLabels:
   # print((i-1))
    sentences.append(sentences_ar[(i-1)])
    labels.append(labels_ar[(i-1)])

max_sentence_length = max([len(s) for s in words])

words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]

print(sentences)
print( labels)
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
#---------------------------------------------------------------------------------
# word_vector_length = 16

# word_vectors = np.zeros((len(words),max_sentence_length,word_vector_length))
# sample_idx = 0
# for s in words: #for each sentence
#     word_idx = max_sentence_length - len(s) #ประโยที่สั่นกว่าความยาวสูงสุด word vector ของต้นประโยคจะเป็น 0
#     for w in s: #for each word in a sentence
#         word_vectors[sample_idx,word_idx,:] = wvmodel.wv[w]
#         word_idx = word_idx+1
#     sample_idx = sample_idx+1
#-----------------------------pretrained----------------------------------
word_vector_length = 300
word_vectors = np.zeros((len(words),max_sentence_length,word_vector_length))
sample_count = 0
for s in words:
    word_count = 0
    for w in s:
        try:
            word_vectors[sample_count,max_sentence_length-word_count-1,:] = vocab_vec[w]
            word_count = word_count+1
        except:#pass don't find word
            pass
    sample_count = sample_count+1

print(word_vectors.shape)

print(to_categorical(labels))
#----------------------Extract bag-of-words--------------------------------------------
# word_vectors = np.zeros((len(words),len(vocab)))
# for i in range(0,len(words)):
#     count = 0
#     for j in range(0,len(words[i])):
#         k = 0
#         for w in vocab:
#             if(words[i][j] == w):
#                 word_vectors[i][k] = word_vectors[i][k]+1
#                 count = count+1
#             k = k+1
#     word_vectors[i] = word_vectors[i]/count

#     inputLayer = Input(shape=(len(vocab),))
# h1 = Dense(64, activation='tanh')(inputLayer)
# h2 = Dense(64, activation='tanh')(h1)
# outputLayer = Dense(3, activation='softmax')(h2)
# model = Model(inputs=inputLayer, outputs=outputLayer)

# print(word_vectors.shape)
#------------------Create and train RNN---------------------------    

inputLayer = Input(shape=(max_sentence_length,word_vector_length,))
#rnn = SimpleRNN(32, activation='relu')(inputLayer) #the number of nodes in hidden layer = 32
rnn = GRU(32, activation='relu')(inputLayer)
# rnn = LSTM(40, activation='relu')(inputLayer)
rnn =Dropout(0.2)(rnn)
outputLayer = Dense(3, activation='softmax')(rnn) #for 3 classes
model = Model(inputs=inputLayer, outputs=outputLayer)
#------------------Bidirectional-------------------------
# LSTM_SIZE = 100
# inputLayer =Input(shape=(max_sentence_length,word_vector_length,))
# lstmLayer = Bidirectional(LSTM(LSTM_SIZE, activation='relu'))(inputLayer) #Bidirectional LSTM
# outputLayer = Dense(3, activation='softmax')(lstmLayer)
# model = Model(inputs=inputLayer, outputs=outputLayer)
#-----------------------------------------------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(word_vectors, to_categorical(labels), epochs=100, batch_size=50, validation_split = 0.2)

model.save('Project_model_RNN.h5')
#------------------Evaluate by test set---------------------------  


file = open('input.txt', 'r',encoding = 'utf-8-sig')
data = np.asarray([[x for x in line.split('::')] for line in file.readlines()])
sentences_ar = [d[1] for d in data]
sentences_ar = [s.replace('\n','') for s in sentences_ar]
words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences_ar]
# word_vectors2 = np.zeros((len(words),max_sentence_length,word_vector_length))
# sample_count = 0
# for s in words:
#     word_count = 0
#     for w in s:
#         try:
#             word_vectors2[sample_count,max_sentence_length-word_count-1,:] = vocab_vec[w]
#             word_count = word_count+1
#         except:#pass don't find word
#             pass
#     sample_count = sample_count+1

word_vectors2 = np.zeros((len(words),max_sentence_length,word_vector_length))
sample_count = 0
for s in words:
    word_count = 0
    for w in s:
        try:
            word_vectors2[sample_count,max_sentence_length-word_count-1,:] = vocab_vec[w]
            word_count = word_count+1
        except:#pass don't find word
            pass
    sample_count = sample_count+1

y_pred = model.predict(word_vectors2)


print(y_pred)

file = open('ans_test.txt', 'r',encoding = 'utf-8-sig')
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
file.close()   

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

# print(y_pred[i])
# print(labelsS[i])
file.close()
cm = confusion_matrix(labels_ar, y_pred.argmax(axis=1))
print('Confusion Matrix')
print(cm)
score = model.evaluate(word_vectors2 , to_categorical(labels_ar))

print(score)
