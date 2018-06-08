# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os 
import yaml 
import multiprocessing
import jieba
from sklearn.model_selection import train_test_split
from keras import callbacks,utils
from keras.preprocessing import text,sequence
from keras.models import Sequential,model_from_yaml,load_model
from keras.layers.embeddings import Embedding
from keras.layers import Dense,Dropout,Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD,Adam

# 基于LSTM的文本分类（positive/negative）

np.random.seed(7)
max_sequence_length=100
embedding_dim=200
window_size=7
batch_size=128
cpu_count=multiprocessing.cpu_count()  #多进程

def load_data():
    '''
    load data
    '''
    print('loading data...')
    pos=pd.read_excel("E:/dataset/words_classification/dataset/pos.xls",header=None,index=None)
    neg=pd.read_excel("E:/dataset/words_classification/dataset/neg.xls",header=None,index=None)

    data=np.concatenate((pos[0],neg[0]))
    labels=np.concatenate((np.ones(len(pos)),np.zeros(len(neg))))

    return data,labels

def train_word2vect(train_inputtexts):
    '''
    convert texts to vector
    '''
    texts=[]
    for seq in train_inputtexts:
        seg_seq=jieba.lcut(seq.replace('\n',''))
        d=" ".join(seg_seq)
        texts.append(d)
    
    # word_count,词出现次数的字典（在所有文本中出现的次数）
    # word_doc,词出现在文档数的字典（在几个文档中出现）
    # word_index,词映射到索引的字典
    # document_count,文档数的字典
    tokenizer=text.Tokenizer()
    tokenizer.fit_on_texts(texts)  #用以训练的文本列表，无返回值
    text_sequences=tokenizer.texts_to_sequences(texts)
    word_index=tokenizer.word_index
    # pad_sequences,处理序列，填充与截断
    data=sequence.pad_sequences(text_sequences,maxlen=max_sequence_length)


    '''
    # GloVe预训练，从GloVe文件中解析出每个词和它所对应的词向量，并用字典的方式存储
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    # 根据得到的字典生成语料所定义的词向量矩阵
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector 
    '''

    return word_index,data

def train_model(input_dim,x_train,y_train,x_test,y_test):
    print(input_dim)
    print('building model...')

    model=Sequential()
    # input_dim,字典长度，输入数据的最大下标+1
    model.add(Embedding(input_dim,embedding_dim,input_length=max_sequence_length))
    model.add(LSTM(256,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    print('compile model...')
    sgd=Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

    tbCallBack=callbacks.TensorBoard(log_dir='E:/ML/.vscode/logs/text_p_n/text_p_n_1',histogram_freq=0,write_graph=True)
    best_model=callbacks.ModelCheckpoint('E:/ML/models/nlp/text_p_n/word2vec_pn_1.h5',monitor='val_loss',verbose=0,save_best_only=True)

    print('train model...')
    model.fit(x_train,y_train,batch_size=batch_size,epochs=3,verbose=1,validation_data=(x_test,y_test),callbacks=[tbCallBack,best_model])

    print('evaluata...')
    score,accuracy=model.evaluate(x_test,y_test,batch_size=batch_size)
    print('\nTest score:',score)
    print('Test accuracy:',accuracy)

    yaml_string=model.to_yaml()
    with open('E:/ML/models/nlp/text_p_n/word2vec_pn_1.yaml','w') as outfile:
        outfile.write(yaml_string)

    # model.save_weights('E:/ML/models/nlp/Word2vec_pn.h5')

def train():

    inputtexts,labels=load_data()

    word_index,data=train_word2vect(inputtexts)

    input_dim=len(word_index)+1
    x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.3)

    train_model(input_dim,x_train,y_train,x_test,y_test)

def predict_pn():

    with open('E:/ML/models/nlp/text_p_n/word2vec_pn_1.yaml') as yamlfile:
        loaded_model_yaml=yamlfile.read()

    model=model_from_yaml(loaded_model_yaml)
    model.load_weights('E:/ML/models/nlp/text_p_n/word2vec_pn_1.h5')

    sgd=Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])


    pre_input_texts = ["很好很满意","特别好，服务态度很不错，售后很及时","不好不满意","质量问题严重","商家态度很差","售后很渣，差评"]
    pre_index,pre_texts=train_word2vect(pre_input_texts)

    pre_result=model.predict_classes(pre_texts)
    print(pre_result)
    labels=[int(round(x[0])) for x in pre_result]
    label2word={1:'pos',0:'neg'}
    for i in range(len(pre_result)):
        print('{0}------{1}'.format(pre_input_texts[i],label2word[labels[i]]))

if __name__=='__main__':
    train()
    # predict_pn()