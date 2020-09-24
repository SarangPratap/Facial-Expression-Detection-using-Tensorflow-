# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:15:30 2020

@author: Sarang
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator 
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)  #Returns if Tensorflow can access gpu
tf.test.gpu_device_name()
filename='fer2013.csv'


Df=pd.read_csv('fer2013.csv')
print(Df['emotion'].value_counts())

#Extracting Data to 2-d Array type For cnn
def Getdata(filename):
    Y=[]
    X=[]
    first=True
    for line in open(filename): #file stream input 
        if first:  #we dont need labels 
            first=False
        else:
            row=line.split(',')                         #splitting the columns 
            Y.append(int(row[0]))                       #Since our Target variable in 1st column
            X.append([int(p) for p in row[1].split()])  # pixels array in second column
    X,Y=np.array(X),np.array(Y)                         # conv to array
    return X,Y     #returning array

X,Y=Getdata(filename)

X.shape                         #shape of array is == shape of dataset (35887,2304)
num_class=len(set(Y))                     #num of targets in the Y Variable

N,D=X.shape
X=X.reshape(N,48,48,1)  #you have to make 35887 arrays which will have 48 arrays of 48 arrays having 1 elements
X.shape


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)


from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from keras import regularizers
from keras.layers import LeakyReLU



def mymodel():
    with tf.device('/GPU:0'):
        cnn=Sequential()
        inputshape=(48,48,1)
        cnn.add(Conv2D(64, (5, 5), input_shape=inputshape,kernel_regularizer=regularizers.l2(0.01), padding='same',strides=(2,2)))
        cnn.add(LeakyReLU(alpha=0.1))
        cnn.add(MaxPooling2D(pool_size=(2,2)))
        #cnn_model.add(LeakyReLU(alpha=0.1))
        cnn.add(Conv2D(64, (5, 5), padding='same'))  #input_shape is not needed on hidden layer
        cnn.add(LeakyReLU(alpha=0.1))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.2))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        
        cnn.add(Conv2D(128, (5, 5),padding='same'))
        cnn.add(LeakyReLU(alpha=0.1))
        cnn.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.2))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dense(128))

        cnn.add(Conv2D(256, (3, 3),padding='same'))
        cnn.add(LeakyReLU(alpha=0.1))
        cnn.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.2))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
    
        cnn.add(Flatten())
        cnn.add(Dense(units=128)) #128 neurons 
        cnn.add(LeakyReLU(alpha=0.1))
        cnn.add(BatchNormalization())
        cnn.add(Dropout(0.2))
        cnn.add(Dense(7))  #7 output so 7 neurons 
        cnn.add(Activation('softmax'))  # multiclass classidication activation function
        
        cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
        return cnn

model=mymodel()
model.summary()

tf.keras.backend.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the cnn
with tf.device('/GPU:0'):
    history=model.fit(x=X_train,     
                y=y_train, 
                batch_size=64, 
                epochs=20, 
                verbose=1, 
                validation_data=(X_test,y_test),
                shuffle=True
                )
    
objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
y_pos = np.arange(len(objects))
print(y_pos)
    
test_image=image.load_img('img.jpg',grayscale=True,target_size=(48,48)) #target size must be equal to input size
show_img=image.load_img('img.jpg', grayscale=False, target_size=(200, 200))
test_image = image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
    #test_image /= 255
result=model.predict(test_image)
import matplotlib.pyplot as plt
plt.gray()
plt.imshow(show_img)
plt.show()


m=float('-inf')
a=result[0]
for i in range(0,len(a)):
    if (a[i]>m):
        m=a[i]
        ind=i  
print('Expression Prediction:',objects[ind])


"""========================For live emotion detection====================="""
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
face_class=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

source= cv2.VideoCapture(0)

while(True):
    _,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_class.detectMultiScale(gray,1.3,5)
    
    for x,y,w,h in faces:
        face_img=gray[y:y+w,x:x+w]  #crop the face only
        resized=cv2.resize(face_img,(48,48))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,48,48,1)) #cnn takes 4d input
        result=model.predict(reshaped)
        m=float('-inf')
        a=result[0]
        for i in range(0,len(a)):
            if a[i]>m:
                m=a[i]
                ind=i
        cv2.putText(img,objects[ind],(10,250), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('emotionDetector', img)
    key=cv2.waitKey(1) #wait for 1 millis
    if(key==27):
        break
cv2.destroyAllWindows()
source.release() #releasing camera
     
    
