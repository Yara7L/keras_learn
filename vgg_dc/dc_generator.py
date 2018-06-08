import numpy as np 
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
# for real-time data augmentation
# training keras a model using python data generators

'''
2000训练，800验证，经过图像增强，简单CNN网络，30 epochs
运行时间较短，wei精度约85%，未过拟合
'''

# rotation_range,指定随机选择的图片角度；width_shift指定方向上随机移动的程度，是0-1之间的比例
# shear_shift剪切变换的程度；zoom_range进行随机放大
# horizontal_flip随机对图片进行水平翻转；fill_mode指定当需要像素填充时的方式
# datagen=ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# img=load_img('E:/dataset/cat_dog/train_less/cat/cat.0.jpg')
# x=img_to_array(img)
# x=x.reshape((1,)+x.shape)

# i=0
# for batch in datagen.flow(x,batch_size=1,save_to_dir='E:/dataset/cat_dog/train_less/preview',save_prefix='cat',save_format='jpeg'):
#     i+=1
#     if i>10:
#         break

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)
#生成一个batch的图像数据，支持实时数据提升，无限生成数据，直到达到规定的epoch次数。
train_generator=train_datagen.flow_from_directory(
    'E:/dataset/cat_dog/train_gen',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary' #binary_crossentropy
)

validation_generator=test_datagen.flow_from_directory(
    'E:/dataset/cat_dog/validation_less',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary' #binary_crossentropy
)

model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=30,
    validation_data=validation_generator,
    nb_val_samples=800
)

model.save_weights('E:/ML/models/dc_vgg/dc_less_gen.h5')