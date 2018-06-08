import pandas as pd
import numpy as np
import jieba
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
import logging
import os,re
import gensim
from functools import reduce
from scipy.spatial.distance import pdist


def easy():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    data = []
    with open(
            'E:/dataset/words_classification/dataset/all.txt',
            'r',
            encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            data.append(line)
    print(len(data))

    sentences = gensim.models.doc2vec.TaggedLineDocument(
        'E:/dataset/words_classification/dataset/all.txt')
    model = gensim.models.Doc2Vec(sentences, size=100, window=5)
    
    model.save('E:/dataset/words_classification/dataset/doc2vect.model')
    print(len(model.docvecs))

    with open(
            'E:/dataset/words_classification/dataset/all_doc_model',
            'w',
            encoding='utf-8') as f:
        for idx, docvec in enumerate(model.docvecs):
            for value in docvec:
                f.write(str(value) + ' ')
            f.write('\n')
            if idx == len(model.docvecs) - 1:
                break
    return model

def train():
    # 取出对应的文档向量
    data = []

    with open(
            'E:/dataset/words_classification/dataset/all_doc_model',
            'r',
            encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            vector = []
            for i in line.split(' '):
                if i:
                    vector.append(i)
            data.append(vector)

    pos = pd.read_excel(
        "E:/dataset/words_classification/dataset/pos.xls",
        header=None,
        index=None)
    neg = pd.read_excel(
        "E:/dataset/words_classification/dataset/neg.xls",
        header=None,
        index=None)

    data = np.array(data)
    print(data.shape)
    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)

    clf = SVC(C=2, probability=True)
    clf.fit(x_train, y_train)
    print('Test Accuracy: %.2f' % clf.score(x_test, y_test))

def getvect_doc(doc):
    model=gensim.models.Word2Vec.load('E:/dataset/words_classification/dataset/doc2vect.model')
    # 停用词
    stop_words = []
    with open(
            'E:/dataset/NLP/stopwords/stopwords_TUH.txt', 'r',
            encoding='gbk') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
 
    reg_html = re.compile(r'<[^>]+>', re.S)
    doc = reg_html.sub('', doc)
    doc = re.sub('[０-９]', '', doc)
    doc = re.sub('\s', '', doc)
    word_list = list(jieba.cut(doc))
    out_str = ''
    for word in word_list:
        if word not in stop_words:
            out_str += word
            out_str += ' '
    segments = out_str.split(sep=" ")

    vect=model.infer_vector(segments)

    return vect

def predict():
    x="我们都喜欢，很满意"
    y="十分好，特别喜爱"
    x=getvect_doc(x)
    y=getvect_doc(y)
    cos=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

    X=np.vstack([x,y])
    d2=1-pdist(X,'cosine')
    print(cos)
    print(d2)


if __name__ == '__main__':
    # easy()
    train()
    # predict()  #相似度
    