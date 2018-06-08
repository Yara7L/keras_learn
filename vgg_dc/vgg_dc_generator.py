# -- coding:utf-8 --
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#0,显示所有日志；1，显示info、warning、error日志；2，显示warning喝error日志；3，显示error日志
import numpy as np 
import matplotlib.pyplot as plt 
from keras import callbacks,regularizers,utils
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import Adam,SGD
from keras.preprocessing import image
from sklearn import metrics

'''
图像生成，加载VGG16（include_top=Flase）的权重，predict_generator（一定注意参数）先训练得到bottleneck特征并保存
用bottleneck的特征加上两个全连接层训练
30 epochs，速度超快，就是明显过拟合了
加了L2正则后，泛化能力有提升，最后也就88%，提升空间很大。
'''

def save_bottleneck_features():

    from keras import applications  
    # include_top: whether to include the 3 fully-connected layers at the top of the network. 
    model = applications.VGG16(include_top=False, weights='imagenet') 
  
    datagen=image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator=datagen.flow_from_directory(
        'E:/dataset/cat_dog/train_gen',
        target_size=(128,128),
        batch_size=128,
        class_mode=None, #no labels
        shuffle=False  #be in order
        )

    model.load_weights('E:/ML/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    bottleneck_features_train=model.predict_generator(train_generator,steps=2000/128)
    np.save(open('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_features_train.npy','wb'),bottleneck_features_train)

    validation_generator=datagen.flow_from_directory(
        'E:/dataset/cat_dog/validation_less',
        target_size=(128,128),
        batch_size=128,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation=model.predict_generator(validation_generator,steps=800/128)
    np.save(open('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_features_validation.npy','wb'),bottleneck_features_validation)

def train_top_model():
    
    # 导入bottleneck_features数据
    train_data=np.load(open('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_features_train.npy','rb'))
    train_labels=np.array([0]*1000+[1]*1000)

    validation_data=np.load(open('E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_features_validation.npy','rb'))
    validation_labels=np.array([0]*400+[1]*400)
   
    # train_labels=keras.utils.to_categorical(train_labels,2)
    # validation_labels=keras.utils.to_categorical(validation_labels,2)

    model=Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.02)))

    print(model.summary())

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    best_model=callbacks.ModelCheckpoint("E:/ML/models/dc_vgg/vgg_dc_generator/bottleneck_fc_model.h5",monitor='val_loss',verbose=1,save_best_only=True,mode='min')
    history=callbacks.TensorBoard(log_dir='E:/ML/.vscode/dc_logs/vgg_generator_less1',histogram_freq=1,write_graph=True,write_images=False)
    early_stop=callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=0,mode='min')
    fitted_model=model.fit(train_data,train_labels,epochs=30,batch_size=128,validation_data=(validation_data,validation_labels),callbacks=[history,best_model,early_stop])

    # matplotlib show or TensorBoard的log
    fig=plt.figure(figsize=(15,5))
    plt.plot(fitted_model.history['loss'],'g',label="train_losses")
    plt.plot(fitted_model.history['val_loss'],'r',label="val_losses")
    plt.grid(True)
    plt.title('Training loss vs. Validation loss--VGG16')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    fig=plt.figure(figsize=(15,5))
    plt.plot(fitted_model.history['acc'],'g',label="train_accuracy")
    plt.plot(fitted_model.history['val_acc'],'r',label="val_accuracy")
    plt.grid(True)
    plt.title('Training accuracy vs. Validation accuracy--VGG16')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
# save_bottleneck_features()
train_top_model()

