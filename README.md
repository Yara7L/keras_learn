## keras的一些例子 问题待解决的文件在最开始已标记
### keras_dc_ny系列  基于CNN的猫狗分类
其中每个.py文件微调了一些超参（网络结构，卷积核，lr，优化，L2）。综合来看，每一个差别都不是特别明显，加了L2正则能取得较好的泛化效果。

### vgg_dc  在VGG16上猫狗分类
选择了小部分数据集进行图片生成增强，在VGG16上，得到Bottleneck特征的训练，或者finetune处理

### gan_mnist  生成手写数字

### keras_C_text_p_n系列  多种情感分析的方法
CNN：利用keras.processing的text和sequence的进行文字到数值型的处理，建立CNN（embedding+3(Conv1D+MaxPooling1D+Dropout）+Flatten+Dense+sigmoid）模型分类.Accuracy约91%.
LSTM：建立LSTM（Embedding+LSTM+sigmoid）LSTM+Dense+sigmoid分类，Accuracy约91%.
TF-IDF：生成词典,生成TF-IDF向量,生成LSI模型,分类器SVC（linear）训练,对新文本进行分类，Accuracy约86%.
Word2Vec：建立Word2Vec模型得到词向量，文档向量等于各个词向量的加和平均（词典太小，许多单词不在词典内），再利用SVC进行分类. 
Doc2Vec:建立Doc2Vec模型得到文档向量，直接用该向量进行文档分类.Accuracy约69%.
### keras_cifar-10.py  cifar-10分类  CNN
