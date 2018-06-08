import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras import applications,optimizers

'''
调整权重，fine-tune
只fine-tune最后的卷积块，底层卷积模块学习到的特征更加一般，更加不具有抽象性；
保持后两个卷积块不动，只fine-tune后面的卷积块（学习别的特征）
在很低的学习率下进行，SGD优化（不选择自适应学习率的优化算法RMSProp），以免破坏特征。
与VGG16连接处有问题。
'''

model=applications.VGG16(weights='imagenet',include_top=False)

# VGG16是函数式模型
# top_model=Sequential()
# top_model.add(Flatten(input_shape=model.output_shape[1:]))
# top_model.add(Dense(256,activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1,activation='sigmoid'))

x=model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
predictions=Dense(1,activation='sigmoid')(x)

vgg_model=Model(input=model.input,output=predictions)

# vgg_model.load_weights('E:/ML/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# 两个网络的融合,有错
# model.add(top_model)

# 将最后一个卷积块前的卷积参数冻结

for layer in vgg_model.layers[:25]:
    layer.trainable=False

vgg_model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])

datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator=datagen.flow_from_directory(
    'E:/dataset/cat_dog/train_gen',
    target_size=(128,128),
    batch_size=64,
    class_mode='binary')

validation_generator=datagen.flow_from_directory(
    'E:/dataset/cat_dog/validation_less',
    target_size=(128,128),
    batch_size=64,
    class_mode='binary')

vgg_model.fit_generator(
    train_generator,
    steps_per_epoch=2000//32,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=800//32)



