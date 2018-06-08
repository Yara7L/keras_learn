import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#0,显示所有日志；1，显示info、warning、error日志；2，显示warning喝error日志；3，显示error日志
import numpy as np 
import random
import matplotlib.pyplot as plt 
from keras import callbacks,regularizers
from keras.models import Sequential,model_from_yaml,load_model,Model
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,ZeroPadding2D,MaxPooling2D
from keras.optimizers import Adam,SGD
from keras.preprocessing import image
from keras.utils import np_utils,plot_model
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input,decode_predictions
from sklearn import metrics
from keras import Input
from keras import applications

'''
wrong
'''


img_size=224
'''
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

    labels=np_utils.to_categorical(labels,2)
    return data,labels
'''
def VGG_16(weights_path=None):
    model=Sequential()
     # Block 1
    model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')) 
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block1_pool'))

    # Block 2
    model.add( Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add( Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block2_pool'))

    # Block 3
    model.add( Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add( Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add( Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block3_pool'))

    # Block 4
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block4_pool'))

    # Block 5
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2),padding='same', name='block5_pool'))

    # model.add(Flatten())
    # model.add(Dense(4096,activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096,activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000,activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    return model

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):

    model = VGG16(weights='imagenet', include_top=True)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    x=Dense(num_classes, activation='softmax')(model.output)
    model=Model(model.input,x)
    for layer in model.layers[:8]:
        layer.trainable = False
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])  

    return model


train_path='E:/dataset/cat_dog/train'
test_path='test'

def make_label(file_name):
    label=file_name.split('.')[0]

    if label=='cat':
        return [0]
    else:
        return [1]

def make_data(img_path,img_size):
    path_length=len(os.listdir(img_path))
    images=np.zeros((path_length,img_size,img_size,3),dtype=np.uint8)
    labels=np.zeros((path_length,1),dtype=np.float32)
    count=0
    for file_name in os.listdir(img_path):
        labels[count] = make_label(file_name)
        images[count] = cv2.resize(cv2.imread(img_path+'/'+file_name),(img_size,img_size))
        b,g,r = cv2.split(images[count])       # get b,g,r
        images[count] = cv2.merge([r,g,b])  # switch it to rgb
        count+=1
    
    #shuffle
    p=np.random.permutation(path_length)
    images,labels=images[p],labels[p]
    return images,labels

print("begin==============\n")
train_img,train_label=make_data(train_path,224)
np.save('train_img.npy',train_img)
np.save('train_label.npy',train_label)


for i in range(10):
    index=random.randint(0,len(os.listdir(train_path)))
    plt.subplot(2,5,i+1)
    plt.title(train_label[index])
    plt.imshow(train_img[index])
    plt.axis('off')
plt.show()

img_size=224
input =Input(shape=(img_size,img_size,3))
base_model = applications.VGG16(include_top=False, weights='imagenet') 
# base_model=VGG_16('E:/ML/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# base_model = VGG16(weights='imagenet', input_tensor=input,include_top=False)

x=Flatten()(base_model)
x=Dense(2048,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.7)(x)

output1=Dense(1,activation='sigmoid')

model=Model(input=input,output=output1)


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot,plot_model

plot_model(model,to_file='model.png',show_shapes=True)
SVG(model_to_dot(model,1).create(prog='dot',format='svg'))


for layer in model.layers[:19]:
    layer.trainable=False

opt=SGD(lr=0.0001,momentum=0.9)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

# model = vgg16_model(224, 224, 3, 2)

model.fit(train_img,train_label,validation_split=0.2,callbacks=[TensorBoard(log_dir='E:/ML/.vscode/dc_logs/vgg_dc1')])
model.save('E:/ML/models/vgg_model_dc.h5')

print('finished')
