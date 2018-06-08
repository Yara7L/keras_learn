from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import pandas as pd
import numpy as np
import jieba
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging


#  用文档中各个词语的词向量的加和平均来表示该文档的文档向量。
#  新文档中有的词不在词向量字典中（min_count=1，不应该呀），报错。


def load_data():
    '''
    load data
    '''
    print('loading data...')
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

    labels = np.concatenate((np.ones(len(pos_split)),
                             np.zeros(len(neg_split))))
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)

    print('==================================')
    # print(len(data))

    with open(
            "E:/dataset/words_classification/dataset/all.txt",
            'w',
            encoding='utf-8') as fW:
        # print(len(data))
        for i in range(len(data)):
            fW.write(str(data[i]))
            fW.write('\n')

    return x_train, x_test, y_train, y_test


def train(x_train):
    '''
    两种方式皆可
    '''
    n_dim = 300
    w2v = Word2Vec(size=n_dim, min_count=10)

    w2v.build_vocab(x_train)
    w2v.train(x_train)
    return w2v


def build_w2v(text, size, model):
    '''
    word2vec=>加和处理为文档向量
    '''
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    # print(model['好'])
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except:
        # except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def getvect(doc):
    model=gensim.models.Word2Vec.load('E:/dataset/words_classification/dataset/model_vocab')
    
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

    vect_list = []
    count=0
    for w in segments:
        count=count+1
        print(w)
        try:
            vect_list.append(model.wv[w])
        except:
            continue
    vect_list = np.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect

def similar():
    model = Word2Vec.load('E:/dataset/words_classification/dataset/model_vocab')

    print(model['好'])

    print(model.most_similar(positive=['好'], topn=2))
    
    indexes=model.most_similar(u'加',topn=10)
    for index in indexes:
        print(index)

def train_classifer():
    x_train, x_test, y_train, y_test = load_data()

    train_vecs = np.concatenate([build_w2v(z, 100, model) for z in x_train])
    train_vecs = scale(train_vecs)

    test_vecs = np.concatenate([build_w2v(z, 100, model) for z in x_test])
    test_vecs = scale(test_vecs)

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)
    print('Test Accuracy: %.2f' % lr.score(test_vecs, y_test))

if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = []
    with open(
            "E:/dataset/words_classification/dataset/all.txt",
            'r',
            encoding='utf-8') as f:
        for line in f:
            sentences.append(line)
    
    # train_method_1
    # word2vec会在整个句子序列上跑两遍, 第一遍会收集单词及其词频走一个内部字典树结构. 第二遍才会训练神经网络.
    model = Word2Vec(sentences, size=100, window=3, min_count=3, workers=4)
    model.save('E:/dataset/words_classification/dataset/model')

    # train_method_2
    # n_dim = 300
    # # 只遍历一遍数据
    # model = Word2Vec(size=n_dim, min_count=1)
    # model.build_vocab(sentences)
    # model.train(sentences, total_examples=model.corpus_count, epochs=5)
    # model.save('E:/dataset/words_classification/dataset/model_vocab')

    # train_method_3
    # 模型存储为bin
    # Word2Vec('E:/dataset/words_classification/dataset/all.txt','E:/dataset/words_classification/dataset/all_vector.bin',size=300,verbose=True)
    # model=word2vec.load('E:/dataset/words_classification/dataset/all_vector.bin')
    