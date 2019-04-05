import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras
from keras.layers import Input
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils import plot_model
from keras.optimizers import Adam

from keras.models import Model
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Dropout, concatenate
from keras.applications.vgg16 import VGG16
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
print('Total Train samples: {}\nTotal valid samples: {}'. format(len(X_train), len(X_valid)))  

#Data augementation
def img_brighteness(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bias = 0.25
    img_brightness = bias + np.random.uniform()
    hsv_img[:, :, 2] = hsv_img[:, :, 2] * img_brightness
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def bias(n):
    return 1. / (n+1.)

def position_based_angle(angle, angle_position):
    if angle_position == "positive":
        if abs(angle)  > 1:
            angle += 0.25
        else:
            angle += 0.18
    else:
        if abs(angle)  > 1:
            angle -= 0.25
        else:
            angle -= 0.18
    return angle
        
  

def adjust_steering(image, angle, images, angles):
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
            image_new = img_brighteness(image)
            images.append(image_new)
            angles.append(angle)
        if np.random.randint(3) == 2:
            image_new = np.fliplr(image)
            angle_new = -angle
            images.append(image_new)
            angles.append(angle_new)
    return images, angles


def generator(lines_path, img_path, batch_size = 32):
    path = img_path
    lines = lines_path
    sum_lines = len(lines)
    batch_number = 1
    angle_position = "positive"
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
                        img_name = path + batch_sample[1].split('/')[-1]
                        angle_position = "positive"
                        augementation_angle = position_based_angle(angle, angle_position)
                     elif img_choice == 1:
                        img_name = path + batch_sample[0].split('/')[-1]
                        augementation_angle = angle
                     else:
                        img_name = path + batch_sample[2].split('/')[-1]
                        angle_position = "negative"
                        augementation_angle = position_based_angle(angle, angle_position)
                        
                     image = cv2.imread(img_name)
                     images, angles = adjust_steering(image, augementation_angle, images, angles)
                batch_number += 1
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)   

train_generator = generator(train_data, img_path)
validate_generator = generator(validation_data, img_path)


def get_vgg16_model(image, velocity):
    # Inputs
    image = Cropping2D(cropping=((60, 20), (0, 0)))(image)

    # Load pretrained VGG16
    base_model = VGG16(input_tensor=image, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # Flattening for fully-connected
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(.25)(x)
    x = Dense(100, activation='relu')(x)

    # Steering tower
    xa = Dense(50, activation='relu')(x)
    xa = Dense(10, activation='relu')(xa)
    xa = Dense(1, name='steering')(xa)

    # Throttle tower
    xt = concatenate([x, xa, velocity])
    xt = Dropout(.5)(xt)
    xt = Dense(10, activation='relu')(xt)
    xt = Dense(1, name='throttle')(xt)

    # Define model with multiple inputs and outputs
    model = Model(inputs=[base_model.input, velocity], outputs=[xa, xt])
    return model

# Keras inputs
image = Input(shape=(160, 320, 3), name='image')
velocity = Input(shape=(1,), name='velocity')

model = get_vgg16_model(image, velocity)
# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer=Adam(lr = 2e-5), loss='mse', loss_weights=[1., .01])

plot_model(model, '{}.png'.format("vgg16"), show_shapes=True)
print(model.summary())

file = 'model_new.h5'
callbacks = [
        # Save best model
        ModelCheckpoint(file, monitor='val_loss', verbose = 1, save_best_only=True),
        # Stop training after 5 epochs without improvement
        EarlyStopping(monitor='val_loss', patience=2),
        # Polynomial decaying learning rate
        LearningRateScheduler(lambda x: 2e-5 * (.8 ** x)),
    ]

history_object = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_data),
        epochs=100,
        callbacks=callbacks,
        validation_data=validate_generator,
        validation_steps=len(validation_data)
    )

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
