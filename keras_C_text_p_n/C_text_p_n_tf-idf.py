import numpy as np
import collections
import re
import jieba
import math
import pickle as pkl 
import pandas as pd
from gensim import corpora, models, similarities
import time
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import train_test_split


def data():

    data = []
    pos = pd.read_excel(
        "E:/dataset/words_classification/dataset/pos.xls",
        header=None,
        index=None)
    neg = pd.read_excel(
        "E:/dataset/words_classification/dataset/neg.xls",
        header=None,
        index=None)

    stop_words = []
    with open(
            'E:/dataset/NLP/stopwords/stopwords_TUH.txt', 'r',
            encoding='gbk') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)

    pos_split = []
    for index, seq in enumerate(pos[0]):
        pos_seq = list(jieba.cut(seq, cut_all=False))
        pos_line = []
        for word in pos_seq:
            if word not in stop_words:
                pos_line.append(word)
        pos_split.append(pos_line)
    neg_split = []
    for index, seq in enumerate(neg[0]):
        neg_seq = list(jieba.cut(seq, cut_all=False))
        neg_line = []
        for word in neg_seq:
            if word not in stop_words:
                neg_line.append(word)
        neg_split.append(neg_line)

    data = np.concatenate((pos_split, neg_split))

    # 分词词典
    data_dic = corpora.Dictionary(data)
    data_dic.save('E:/dataset/words_classification/dataset/tf-idf_model/data_dict')

    # 转为频率表示的稀疏向量
    corpus = [data_dic.doc2bow(text) for text in data]
    corpora.MmCorpus.serialize('E:/dataset/words_classification/dataset/tf-idf_model/data_corpus',corpus)#保存生成的语料  

    # tf-idf
    tfidf_model = models.TfidfModel(corpus=corpus, dictionary=data_dic)
    # corpus_tfidf = tfidf_model[corpus]
    corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    tfidf_model.save('E:/dataset/words_classification/dataset/tf-idf_model/data_tf-idf.tfidf') 
    
    # lsi
    lsi_model = models.LsiModel(
        corpus=corpus, id2word=data_dic, num_topics=100)
    # corpus_lsi = lsi_model[tfidf_corpus]
    corpus_lsi = [lsi_model[doc] for doc in corpus]
    lsi_model.save("E:/dataset/words_classification/dataset/tf-idf_model/data_lsi") 

    return corpus_lsi


def new_matrix(lsi_corpus_total):
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in lsi_corpus_total:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    lsi_matrix = csr_matrix((data, (rows, cols))).toarray()
    rarray = np.random.random(size=line_count)

    pos = pd.read_excel(
        "E:/dataset/words_classification/dataset/pos.xls",
        header=None,
        index=None)
    neg = pd.read_excel(
        "E:/dataset/words_classification/dataset/neg.xls",
        header=None,
        index=None)

    print(lsi_matrix.shape)
    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    x_train, x_test, y_train, y_test = train_test_split(
        lsi_matrix, labels, test_size=0.2)

    return x_train, x_test, y_train, y_test


# 分类
def train(x_train, x_test, y_train, y_test):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    lda_res = lda.fit(x_train, y_train)
    train_pred = lda_res.predict(x_train)  # 训练集的预测结果
    test_pred = lda_res.predict(x_test)  # 检验集的预测结果
    print('Test Accuracy: %.2f' % lda_res.score(x_test, y_test))

    # clf = SVC()  # 使用RBF核
    clf = LinearSVC()  # 使用线性核
    clf_res = clf.fit(x_train, y_train)
    train_pred = clf_res.predict(x_train)
    test_pred = clf_res.predict(x_test)
    print('Test Accuracy: %.2f' % clf_res.score(x_test, y_test))
    
    with open('E:/dataset/words_classification/dataset/tf-idf_model/predictor','wb') as f:
        pkl.dump(clf_res,f) 


def predict():
    dictionary=corpora.Dictionary.load('E:/dataset/words_classification/dataset/tf-idf_model/data_dict')
    # corpus=corpora.MmCorpus.load('E:/dataset/words_classification/dataset/tf-idf_model/data_corpus')
    tfidf_model = models.TfidfModel.load('E:/dataset/words_classification/dataset/tf-idf_model/data_tf-idf.tfidf') 
    lsi_model = models.LsiModel.load("E:/dataset/words_classification/dataset/tf-idf_model/data_lsi")  
    with open('E:/dataset/words_classification/dataset/tf-idf_model/predictor','rb') as f:
        predictor=pkl.load(f)  
    
    # 多条一起分类，dictionary的时候有问题
    # stop_words = []
    # with open(
    #         'E:/dataset/NLP/stopwords/stopwords_TUH.txt', 'r',
    #         encoding='gbk') as f:
    #     line = f.readline()
    #     while line:
    #         stop_words.append(line[:-1])
    #         line = f.readline()
    # stop_words = set(stop_words)
    
    # test_doc=["很好很满意","特别好，服务态度很不错，售后很及时","不好不满意","质量问题严重","商家态度很差","售后很渣，差评"]
    # try_doc=[]
    # for i in test_doc:
    #     doc=list(jieba.cut(i,cut_all=False))
    #     doc_line = []
    #     for word in doc:
    #         if word not in stop_words:
    #             doc_line.append(word)
    #     try_doc.append(doc_line)
    

    try_doc="我们一家人都十分满意，好评"
    try_doc=list(jieba.cut(try_doc,cut_all=False))
    try_bow=dictionary.doc2bow(try_doc)
    try_tfidf=tfidf_model[try_bow]
    try_lsi=lsi_model[try_tfidf]
    data = []
    cols = []
    rows = []
    for item in try_lsi:
        data.append(item[1])
        cols.append(item[0])
        rows.append(0)
    try_matrix = csr_matrix((data,(rows,cols))).toarray()
    x = predictor.predict(try_matrix)
    # print('分类结果为：{%d}'.format(x))
    print(x)

    # labels=[int(round(x[0])) for x in pre_result]
    # label2word={1:'pos',0:'neg'}
    # for i in range(len(pre_result)):
    #     print('{0}------{1}'.format(pre_input_texts[i],label2word[labels[i]]))

if __name__ == '__main__':
    # corpus_lsi = data()
    # x_train, x_test, y_train, y_test = new_matrix(corpus_lsi)
    # train(x_train, x_test, y_train, y_test)
    predict()