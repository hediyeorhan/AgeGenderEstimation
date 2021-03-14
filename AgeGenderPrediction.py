import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
from PIL import Image

import numpy as np

from random import shuffle
from sklearn.model_selection import train_test_split

import keras 
from keras.layers import *
from keras.models import *
from keras import backend as K


path = "UTKFace/"
files = os.listdir(path)
size = len(files)

images = []
ages = []
genders = []
for file in files:
    image = cv2.imread(path+file,0)
    image = cv2.resize(image,dsize=(64,64))
    image = image.reshape((image.shape[0],image.shape[1],1))
    images.append(image)
    split_var = file.split('_')
    ages.append(split_var[0])
    genders.append(int(split_var[1]))

def age_group(age):
    if age >=0 and age < 18:
        return 1
    elif age < 30:
        return 2
    elif age < 80:
        return 3
    else:
        return 4 


idx = 350
sample = images[idx]

target = np.zeros((size,2),dtype='float32')
features = np.zeros((size,sample.shape[0],sample.shape[1],1),dtype = 'float32')
for i in range(size):
    target[i,0] = age_group(int(ages[i])) / 4
    target[i,1] = int(genders[i])
    features[i] = images[i]
features = features / 255
print("Shape of the images:",sample.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.50, shuffle = True)
print("Training images numbers:",x_train.shape[0])
print("Testing images numbers:",x_test.shape[0])


(x_valid , y_valid) = (x_test[:5927], y_test[:5927])
(x_test, y_test) = (x_test[5927:], y_test[5927:])

inputs = Input(shape=(64,64,1))
conv = Conv2D(16, kernel_size=(3, 3),activation='relu')(inputs)
pool = MaxPooling2D(pool_size=(2, 2))(conv)
conv1 = Conv2D(32, kernel_size=(3, 3),activation='relu')(pool)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, kernel_size=(3, 3),activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, kernel_size=(3, 3),activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
x = Dropout(0.25)(pool3)
flat = Flatten()(x)

dropout = Dropout(0.5)
age_model = Dense(128, activation='relu')(flat)
age_model = dropout(age_model)
age_model = Dense(64, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = Dense(32, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = Dense(1, activation='relu')(age_model)

dropout = Dropout(0.5)
gender_model = Dense(128, activation='relu')(flat)
gender_model = dropout(gender_model)
gender_model = Dense(64, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(32, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(16, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(8, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(1, activation='sigmoid')(gender_model)


model = Model(inputs=inputs, outputs=[age_model,gender_model])
model.compile(optimizer = RMSprop(lr=0.001), loss =['mse','binary_crossentropy'],metrics=['accuracy'])

model.summary()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('loss') < 0.05):
            self.model.stop_training = True

callbacks = myCallback()
history = model.fit(x_train,[y_train[:,0],y_train[:,1]],validation_data=(x_valid,[y_valid[:,0],y_valid[:,1]]),
                    epochs = 100, callbacks = [callbacks])


# YUZ TANIMA - KAMERA 
face_clsfr = cv2.CascadeClassifier("C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")


color_dict={0:(0,0,255)}

rect_size = 9
cap = cv2.VideoCapture(0) 

def get_age(distr):
    distr = distr*4
    if distr >= 0.65 and distr <= 1.4:return "0-18"
    if distr >= 1.65 and distr <= 2.4:return "19-30"
    if distr >= 2.65 and distr <= 3.4:return "31-80"
    if distr >= 3.65 and distr <= 4.4:return "80 +"
    return "Unknown"

def get_gender(gender):
    if(gender <0.5):
        return "Female"
    else:
        return "Male"

while True:
    (ret, img) = cap.read()
    
    img = cv2.flip(img, 1, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    resized = cv2.resize(img,(64,64))
    faces = face_clsfr.detectMultiScale(resized)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
     
        face_img = img[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(64, 64))
        normalized=resized/255.0
        reshaped = np.reshape(normalized, (1,64,64,1))
        reshaped = np.vstack([reshaped])

        result = model.predict(reshaped)    
        age = get_age(result[0])
        gender = get_gender(result[1])
        sonuc = """Gender : {} & Age : {}""".format(gender, age)
    
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[0],2)
        cv2.rectangle(img,(x,y),(x+w,y),color_dict[0],-1) 
        cv2.putText(img, sonuc, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2) 

    cv2.imshow('AGE&GENDER',   img)
    key = cv2.waitKey(10)
    
    if key == 27:   # Esc
        break

cap.release()

cv2.destroyAllWindows()
