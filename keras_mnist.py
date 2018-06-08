from keras import backend as K  
from keras.models import Sequential  
from keras.layers.convolutional import Conv2D  
from keras.layers.convolutional import MaxPooling2D  
from keras.layers.core import Activation,Flatten,Dense  
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data  
from keras.utils import np_utils  
from keras.optimizers import SGD,RMSprop,Adam  
import numpy as np  
import matplotlib.pyplot as plt  
  
class LeNet:  
    @staticmethod  
    def createLeNet(input_shape,nb_class):  
        feature_layers = [  
        Conv2D(20,kernel_size=5,padding="same",input_shape=input_shape),  
        Activation("relu"),  
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),  
        Conv2D(50,kernel_size=5,border_mode="same"),  
        Activation("relu"),  
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),  
        Flatten()  
        ]  
          
        classification_layers=[  
        Dense(500),  
        Activation("relu"),  
        Dense(nb_class),  
        Activation("softmax")  
        ]  
          
        model = Sequential(feature_layers+classification_layers)  
        return model  
  
#parameters  
NB_EPOCH = 20  
BATCH_SIZE = 128  
VERBOSE = 1  
OPTIMIZER = Adam()  
VALIDATION_SPLIT = 0.2  
IMG_ROWS,IMG_COLS = 28,28  
NB_CLASSES = 10  
INPUT_SHAPE =(1,IMG_ROWS,IMG_COLS)  
  
#load mnist dataset  
#  =input_data.read_data_sets("E:/ML/.vscode/MNIST_data/",one_hot=True)
# (X_train,Y_train),(X_test,Y_test) = mnist.load_data()  
K.set_image_dim_ordering("th") # channel first  
  
#normalize the data  
X_train = X_train.astype("float32")  
X_test = X_test.astype("float32")  
X_train /= 255  
X_test /= 255  
X_train = X_train.reshape(X_train.shape[0],1,IMG_ROWS,IMG_COLS)  
X_test = X_test.reshape(X_test.shape[0],1,IMG_ROWS,IMG_COLS)  
print(X_train.shape[0],"train samples")  
print(Y_test.shape[0],"test samples")  
  
#convert class vectors to binary class matrices  
Y_train = np_utils.to_categorical(Y_train,NB_CLASSES)  
Y_test = np_utils.to_categorical(Y_test,NB_CLASSES)  
  
# init the optimizer and model  
model = LeNet.createLeNet(input_shape=INPUT_SHAPE,nb_class=NB_CLASSES)  
model.compile(loss="categorical_crossentropy",optimizer=OPTIMIZER,metrics=["accuracy"])  
  
history = model.fit(X_train,Y_train,  
                batch_size = BATCH_SIZE,  
                epochs = NB_EPOCH,  
                verbose=VERBOSE,  
                validation_split=VALIDATION_SPLIT)  
                  
score = model.evaluate(X_test,Y_test,verbose = VERBOSE)  
print("Test score:",score[0])  
print("Test accuracy:",score[1])  
model.save("my_model.h5")  
  
#show the data in history  
print(history.history.keys())  
  
#summarize history for accuracy  
plt.plot(history.history["acc"])  
plt.plot(history.history["val_acc"])  
plt.title("Model accuracy")  
plt.ylabel("accuracy")  
plt.xlabel("epoch")  
plt.legend(["train","test"],loc="upper left")  
plt.show()  
  
#summarize history for loss  
plt.plot(history.history["loss"])  
plt.plot(history.history["val_loss"])  
plt.title("Model loss")  
plt.ylabel("loss")  
plt.xlabel("epoch")  
plt.legend(["train","test"],loc="upper left")  
plt.show()  