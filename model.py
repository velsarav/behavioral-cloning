import os
import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
# 	for i in range(3):
    source_path = line[0]
    tokens = source_path.split('/')
    filename = tokens[-1]
    local_path = "./data/IMG/" + filename
    image = cv2.imread(local_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
# 	measurements.append(measurement+correction)
# 	measurements.append(measurement-correction)


# augemented_images = []
# augemented_measurements = []
# for image, measurement in zip(images, measurements):
# 	augemented_images.append(image)
# 	augemented_measurements.append(measurement)
# 	flipped_image = cv2.flip(image,3)
# 	flipped_measurement = float(measurement) * -1.0
# 	augemented_images.append(flipped_image)
# 	augemented_measurements.append(flipped_measurement)


X_train = np.array(images)
y_train = np.array(measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(Dropout(0.8))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

