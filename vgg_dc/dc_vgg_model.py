import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications
'''
maybe can do
'''

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)
    model=applications.VGG16(include_top=False,weights='imagenet')
    
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            'E:/dataset/cat_dog/train_gen',
            target_size=(224, 224),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, 2000)
    np.save(open('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            'E:/dataset/cat_dog/validation_less',
            target_size=(224, 224),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, 400)
    np.save(open('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model():

    trian_data=np.load(open('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_features_train.npy'))
    train_labels=np.array([0]*1000+[1]*1000)

    validation_data=np.load(open('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_features_validation.npy'))
    validation_labels=np.array([0]*400+[1]*400)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_fc_model2.h5')


save_bottlebeck_features()
train_top_model()

import matplotlib.pyplot as plt
fig=plt.figure(figsize=(15,5))
plt.plot(fitted_model.history['loss'],'g',label="train_losses")
plt.plot(fitted_model.history['val_loss'],'r',label="val_losses")
plt.grid(True)
plt.title('Training loss vs. Validation loss--VGG16')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
print("6==========")
fig=plt.figure(figsize=(15,5))
plt.plot(fitted_model.history['accuracy'],'g',label="train_accuracy")
plt.plot(fitted_model.history['val_accuracy'],'r',label="val_accuracy")
plt.grid(True)
plt.title('Training accuracy vs. Validation accuracy--VGG16')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()