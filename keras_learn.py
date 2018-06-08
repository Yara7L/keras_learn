import numpy as numpy

a=numpy.array([[1,2],[3,4]])
sum0=numpy.sum(a,axis=0)#行
sum1=numpy.sum(a,axis=1)#列
print(sum0)
print(sum1)

import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD

model=Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(10,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
'''
data=numpy.random.random((1000,100))
labels=numpy.random.randint(2,size=(1000,1))
model.fit(data,labels,epochs=10,batch_size=32)
'''
data2=numpy.random.random((1000,100))
labels2=numpy.random.randint(10,size=(1000,1))

one_hot_labels=keras.utils.to_categorical(labels2,num_classes=10)
model.fit(data2,one_hot_labels,epochs=10,batch_size=32)

#基于多层感知器的softmax多分类
x_train=numpy.random.random((1000,20))
y_train=keras.utils.to_categorical(numpy.random.randint(10,size=(1000,1)),num_classes=10)
x_test=numpy.random.random((100,20))
y_test=keras.utils.to_categorical(numpy.random.randint(10,size=(100,1)),num_classes=10)

model=Sequential()
model.add(Dense(64,activation='relu',input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

sgd=SGD(lr=0.01,decay=1e-5,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=20,batch_size=128)
score=model.evaluate(x_test,y_test,batch_size=128)

#multi-layer perceptron
x_train1=numpy.random.random((1000,20))
y_train1=numpy.random.randint(2,size=(1000,1))
x_test1=numpy.random.random((100,20))
y_test1=numpy.random.randint(2,size=(100,1))

model1=Sequential()
model1.add(Dense(64,input_dim=20,activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(64,activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1,activation='sigmoid'))

model1.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model1.fit(x_train1,y_train1,epochs=5,batch_size=128)
score1=model1.evaluate(x_test1,y_test1,batch_size=128)
print(score1)

print("output_shape----------------")
print(model1.output_shape)
print("summary---------------------")
print(model1.summary())
print("config----------------------")
print(model1.get_config())
print("weights---------------------")
print(model1.get_weights())
