#----------------------------------------------------------------
# Model Code for Udacity Self Driving Car Engineer Nanodegree 
# Term 1, Project 3, Behavioral Cloning
# Deep CNN for Steering Angle Prediction (regression problem)
# Final network is similar to Nvidia's network
# Utilized Udacity's data and augmented with multiple laps of my data
# Also added left and right camera images
# Data loaded using generators
#----------------------------------------------------------------
# Zikri Bayraktar, Ph.D.
#----------------------------------------------------------------

import os
import csv
import cv2
import random
from random import shuffle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.core import Lambda 
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.core import Dropout
from keras.regularizers import l2, activity_l2

print('Libraries are imported!')

#----------------------------------------------------------------
# Input data is generated with Unity's simulator over training track:
# Important Notes:
# When running Unity simulator, make sure to drive the car with mouse to get better angle records.
# Make sure to use Udacity provided data as well.
# Below read in data using generators:

samples = []
cnt=0
with open('Unity_Data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)
print(len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('File names are red in! There are ' )

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:  #loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      angles = []

      for batch_sample in batch_samples:
        #Center image
        fname = batch_sample[0].split('\\')[-1]
        fname = fname.split('/')[-1]
        name = './Unity_Data/IMG/'+fname.strip()
        center_image = cv2.imread(name.strip())
        #center_image = cv2.GaussianBlur(center_image, (3,3), 0)
        center_angle = float(batch_sample[3])
        
        if center_image is not None:
          images.append(center_image)
          angles.append(center_angle)

          #augmentation - mirror:
          images.append(cv2.flip(center_image,1))
          angles.append(center_angle*-1.0)

        #Left image
        fname = batch_sample[1].split('\\')[-1]
        fname = fname.split('/')[-1]
        name = './Unity_Data/IMG/'+fname.strip()
        left_image = cv2.imread(name.strip())
        #left_image = cv2.GaussianBlur(left_image, (3,3), 0)

        if left_image is not None:
          left_angle = center_angle + 0.07
          images.append(left_image)
          angles.append(left_angle)

        #Right Image
        fname = batch_sample[2].split('\\')[-1]
        fname = fname.split('/')[-1]
        name = './Unity_Data/IMG/'+fname.strip()
        right_image = cv2.imread(name.strip())
        #right_image = cv2.GaussianBlur(right_image, (3,3), 0)

        if right_image is not None:
          right_angle = center_angle - 0.07
          images.append(right_image)
          angles.append(right_angle)

      #trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function:
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#----------------------------------------------------------------
# Generate the Keras model based on NVIDIA's papers:
# Few things learned:
# 1. First couple of Conv Layers should not use dropout. It hinders feature extraction.
# 2. No need for max pooling layers.
model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,20),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

print(model.summary())

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mse', optimizer=sgd)
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(4*train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model.h5')

#---------------------------------------------------------------------