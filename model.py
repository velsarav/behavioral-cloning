import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from keras.models import Model
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file = './data/driving_log.csv'
img_path = './data/IMG/'

#read the csv file
def read_data_csv(file):
    lines = []
    # with open('./data/driving_log.csv') as csvfile:
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines[1:]

#read image data
def read_img_data(line_ref_path, img_path):
    path = img_path
    lines = line_ref_path
    images = []
    angles = []
    for line in lines:
        correction = 0.2
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            local_path = path + filename
            image = cv2.imread(local_path)
            images.append(image)
            angle = float(line[3])
            if i == 0:
                angles.append(angle)
            elif i == 1:
                angles.append(angle + correction)
            else:
                angles.append(angle - correction)
    X_train = np.array(images)
    y_train = np.array(angles)
    return X_train, y_train

#split the names of data
train_data, validation_data = train_test_split(read_data_csv(file), shuffle = True, test_size=0.2)

#classify the data
X_train, y_train = read_img_data(train_data, img_path)
X_valid, y_valid = read_img_data(validation_data, img_path)

#Check the data sample
assert len(X_train) == len(y_train), "X_train {} and y_train {} are not equal".format(len(X_train), len(y_train))
assert len(X_valid) == len(y_valid), "X_valid {} and y_valid {} are not equal".format(len(X_valid), len(y_valid))
print('Total Train samples: {}\n Total valid samples: {}'. format(len(X_train), len(X_valid)))  

#Data augementation
def img_brighten(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bias = 0.25
    img_brightness = bias + np.random.uniform()
    hsv_img[:, :, 2] = hsv_img[:, :, 2] * img_brightness
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def bias(n):
    return 1. / (n+1.)


def generator(lines_path, img_path, batch_size = 32):
    path = img_path
    lines = lines_path
    sum_lines = len(lines)
    batch_number = 1
    while 1:
        shuffle(lines)
        for offset  in range(0, sum_lines, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                img_choice = np.random.randint(3)
                angle =float(batch_sample[3])
                if angle + bias(batch_number) < np.random.uniform():
                    if img_choice == 0:
                        name = path + batch_sample[1].split('/')[-1]
                        if abs(angle)  > 1:
                            angle += 0.25
                        else:
                            angle += 0.18
                    elif img_choice == 1:
                        name = path + batch_sample[0].split('/')[-1]
                    else:
                        name = path + batch_sample[2].split('/')[-1]
                        if abs(angle)  > 1:
                            angle -= 0.25
                        else:
                            angle -= 0.18
                    image = cv2.imread(name)
                    if np.random.randint(10) == 0:
                        images.append(image)
                        angles.append(angle)
                    if angle!=0.18 and angle!=-0.18 and angle!=0:
                        if np.random.randint(3) == 0:
                            image_new = np.fliplr(image)
                            angle_new = -angle
                            images.append(image_new)
                            angles.append(angle_new)
                        if np.random.randint(3) == 1 or 2:
                            image_new = img_brighten(image)
                            images.append(image_new)
                            angles.append(angle)
                            if np.random.randint(3) == 2:
                                image_new = np.fliplr(image)
                                angle_new = -angle
                                images.append(image_new)
                                angles.append(angle_new)
                batch_number += 1
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)   

train_generator = generator(train_data, img_path)
validate_generator = generator(validation_data, img_path)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

file = 'model_gen.h5'
earlystopper = EarlyStopping(patience=5, verbose = 1)
checkpointer = ModelCheckpoint(file, monitor='val_loss', verbose = 1, save_best_only=True)
model.fit_generator(train_generator, steps_per_epoch= len(train_data), validation_data = validate_generator,
                        validation_steps=len(validation_data), epochs = 1)
model.save(file)
