import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np 
from keras import callbacks,regularizers
from keras.models import Sequential,model_from_yaml,load_model
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D
from keras.optimizers import Adam,SGD
from keras.preprocessing import image
from keras.utils import np_utils,plot_model
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input,decode_predictions
from sklearn import metrics

np.random.seed(7)
img_h, img_w = 150, 150
image_size = (150, 150)
nbatch_size = 128
nepochs = 45
nb_classes = 2

def load_data():
    path='E:/dataset/cat_dog/train/'
    files=os.listdir(path)
    images=[]
    labels=[]
    for f in files:
        img_path=path+f
        img=image.load_img(img_path,target_size=image_size)
        #img_array=np.asarray(img)
        img_array=image.img_to_array(img)
        images.append(img_array)

        if 'cat' in f:
            labels.append(0)
        else:
            labels.append(1)
    
    data=np.array(images)
    labels=np.array(labels)
    print(labels.shape)
    print(labels)
    labels=np_utils.to_categorical(labels,2)
    print(labels.shape)
    print(labels)
    return data,labels


def main():
    model=Sequential()

    model.add(Conv2D(32,kernel_size=(5,5),input_shape=(img_h,img_w,3),activation='relu',padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64,kernel_size=(5,5),activation='relu',padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128,kernel_size=(5,5),activation='relu',padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(256,kernel_size=(5,5),activation='relu',padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.5))

    model.add(Dense(2,activation='softmax',kernel_regularizer=regularizers.l2(0.02)))

    model.summary()
    
    print(model.summary())

    # print("compile=============================")
    # sgd=Adam(lr=0.0003)
    # model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    
    # print("load_data============================")
    # images,labels=load_data()
    # imagesmean=np.mean(images,axis=0)
    # imagesstd=np.std(images,axis=0)
    # images=images-imagesmean
    # images=images/imagesstd
    # x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.2)
    # print(x_train.shape,y_train.shape)

    # print("train================================")
    # best_model=callbacks.ModelCheckpoint("E:/ML/models/keras_dc_2.h5",monitor='val_loss',verbose=1,save_best_only=True,mode='min')
    # tbcallbacks=callbacks.TensorBoard(log_dir='E:/ML/.vscode/dc_logs/keras_dc_2',histogram_freq=1,write_graph=True,write_images=False)
    # early_stop=callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=0,mode='min')
    # model.fit(x_train,y_train,batch_size=nbatch_size,epochs=nepochs,verbose=1,validation_data=(x_test,y_test),callbacks=[best_model,tbcallbacks,early_stop])

    # print("evaluate==============================")
    # score,accuracy=model.evaluate(x_test,y_test,batch_size=nbatch_size)
    # print('score:',score,'accuracy:',accuracy)

    # yaml_string=model.to_yaml()
    # with open('E:/ML/models/keras_dc_2.yaml','w') as outfile:
    #     outfile.write(yaml_string)
    # model.save_weights('E:/ML/models/keras_dc_2.h5')


def pred_data():
    
    with open('E:/ML/models/keras_dc_2.yaml') as yamlfile:
        loaded_model_yaml=yamlfile.read()
    model=model_from_yaml(loaded_model_yaml)
    model.load_weights('E:/ML/models/keras_dc_2.h5')

    sgd=Adam(lr=0.0003)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    images=[]
    labels_test=[]
    result_test=[]
    path='E:/dataset/predict/dog_cat/'
    for f in os.listdir(path):
        img=image.load_img(path+f,target_size=image_size)
        img_array=image.img_to_array(img)

        x=np.expand_dims(img_array,axis=0)
        x=preprocess_input(x)
        result=model.predict_classes(x,verbose=0)

        print(f,result[0])
        result_test.append(result[0])
        if 'cat' in f:
            labels_test.append(0)
        else:
            labels_test.append(1)
    print(metrics.classification_report(labels_test,result_test))

if __name__ =='__main__':
    print('Start!')
    # load_data()
    main()
    # 3
    # pred_data()

