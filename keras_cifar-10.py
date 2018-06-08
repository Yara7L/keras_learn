# from _future_ import print_function
import numpy as np
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPool2D
from keras.utils import np_utils
from keras import backend as k
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf  
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

np.random.seed(0)
print("Initialized!")

batch_size=2
nb_classes=10
nb_epoch=30
img_rows,img_cols=32,32
nb_filters=[32,32,64,64]
pool_size=(2,2)
kernel_size=(3,3)

(X_train,y_train),(X_test,y_test)=cifar10.load_data()
X_train_mean=np.mean(X_train.astype("float32"),axis=0)
X_test_mean=np.mean(X_test.astype("float32"),axis=0)
X_train_std=np.std(X_train.astype("float32"),axis=0)
X_test_std=np.std(X_test.astype("float32"),axis=0)
X_train=X_train-X_train_mean
X_train=X_train/X_train_std
X_test=X_test-X_test_mean
X_test=X_test/X_test_std

y_train=y_train
y_test=y_test

input_shape=(img_rows,img_cols,3)
y_train=np_utils.to_categorical(y_train,nb_classes)
y_test=np_utils.to_categorical(y_test,nb_classes)

datagen=ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

datagen.fit(X_train)

model=Sequential()
model.add(Conv2D(nb_filters[0],kernel_size,padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(nb_filters[2],kernel_size,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters[3],kernel_size))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

adam=Adam(lr=0.0003)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

best_model=ModelCheckpoint("E:/ML/models/cifar10_best.h5",monitor='val_loss',verbose=0,save_best_only=True,mode='min')
tbcallbacks=TensorBoard(log_dir='E:/ML/.vscode/logs',histogram_freq=1,write_graph=False,write_images=True)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                        steps_per_epoch=128,
                        epochs=nb_epoch, verbose=1,
                        validation_data=(X_test, y_test), 
                        callbacks=[best_model,tbcallbacks])                        
score=model.evaluate(X_test,y_test,verbose=0)
print('Test score',score[0])
print("Accuracy:%.2f%%" % (score[1]*100))
print("Compiled!")
