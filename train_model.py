import cv2
import numpy as np
import pandas as pd
import pickle
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from os.path import join

from imgaug import augmenters as iaa
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from nvidia import get_nvidia
from pretrained import get_mobilenet, get_vgg16

DATA_PATH = './data'

models = {
    'nvidia': {
        'model': get_nvidia,
        'batch': 8,
        'lr': 1e-3,
        'decay': .5,
        'patience': 5
    },
    'mobilenet': {
        'model': get_mobilenet,
        'batch': 8,
        'lr': 1e-4,
        'decay': .7,  # ~1/2 every 2 epochs
        'patience': 2
    },
    'vgg16': {
        'model': get_vgg16,
        'batch': 4,
        'lr': 2e-5,
        'decay': .8,  # ~1/2 every 3 epochs
        'patience': 2
    }
}


def load_train_test_data(csvpath=glob(join(DATA_PATH, '*.csv'))):
    if getattr(csvpath, 'lower'):
        df = pd.read_csv(csvpath)
    elif len(csvpath) == 1:
        df = pd.read_csv(csvpath[0])
    elif len(csvpath) > 1:
        df = pd.concat(
            [pd.read_csv(f) for f in csvpath],
            ignore_index=True)
    else:
        raise ValueError('No csv files found')

    center = df.center.values
    speed = df.speed.values / df.speed.max()
    throttle = df.throttle.values
    y = df.steering.values

    print(">>>>> load_train_test_data")
    return train_test_split(center, speed, throttle, y, test_size=0.2)


def bucket(angle):
    # Bucket by first decimal
    return np.uint8(np.abs(angle * 10))


def generator(images, speed, throttle, steering, batch_size, training=False):
    x_batch, s_batch, a_batch, t_batch = [], [], [], []
    buckets = np.zeros((11,))  # [0, 10]
    original = set()
    while True:
        if training:
            images, speed, throttle, steering = shuffle(images, speed, throttle, steering)
            aug = iaa.SomeOf((1, None), [
                iaa.Add((-80, 0)),
                iaa.Dropout(p=(0, 0.2)),
                iaa.Affine(translate_percent=(-0.05, 0.05))
            ])
        for x, s, t, a in zip(images, speed, throttle, steering):
            if training:
                b = bucket(a)
                if buckets[b] / np.sum(buckets) > 1 / 3:
                    # Make sure no angle bucket dominate the training dataset
                    continue
                buckets[b] += 1
            path = join('./data/IMG', x)
            print(">>>" + path)
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
#             if training:
#                 if np.random.random() < 0.5:
#                     image = np.fliplr(image)
#                     a = -a
#                 if x in original:
#                     # Only augment after original has been added
#                     image = aug.augment_image(image)
#             original.add(x)
            # Normalize
            # Sample-wise center
            # image = (image - np.min(image)) / (np.max(image) - np.min(image))
            # image = (image - 0.5) * 2
            # Dataset-wise center
            image = (image - 127.5) / 127.5
            x_batch.append(image)
            s_batch.append(s)
            a_batch.append(a)
            t_batch.append(t)
            if len(x_batch) >= batch_size:
                yield [np.array(x_batch), np.array(s_batch)], [np.array(a_batch), np.array(t_batch)]
                x_batch, s_batch, a_batch, t_batch = [], [], [], []


def train(model_name='nvidia'):
    # Load dataset
    ct, cv, st, sv, tt, tv, yt, yv = load_train_test_data('data/driving_log.csv')

    # Keras inputs
    image = Input(shape=(160, 320, 3), name='image')
    velocity = Input(shape=(1,), name='velocity')

    # Keras model
    config = models[model_name]
    model = config['model'](image, velocity)
    model.compile(optimizer=Adam(config['lr']), loss='mse', loss_weights=[1., .01])
    plot_model(model, '{}.png'.format(model_name), show_shapes=True)
    print(model.summary())

    # Training
    learning_rate = config['lr']
    callbacks = [
        # Save best model
        ModelCheckpoint('./{}.h5'.format(model_name), monitor='val_loss', save_best_only=True),
        # Stop training after 5 epochs without improvement
        EarlyStopping(monitor='val_loss', patience=config['patience']),
        # Polynomial decaying learning rate
        LearningRateScheduler(lambda x: learning_rate * (config['decay'] ** x)),
    ]

    history = model.fit_generator(
        generator=generator(ct, st, tt, yt, batch_size=config['batch'], training=True),
        steps_per_epoch=yt.size // config['batch'],
        epochs=100,
        callbacks=callbacks,
        validation_data=generator(cv, sv, tv, yv, config['batch']),
        validation_steps=yv.size // config['batch']
    )

    with open('{}.p'.format(model_name), 'wb') as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print('Choose an architecture to train:\n\n{}'.format('\n'.join(models.keys())))
        sys.exit(1)
    train(sys.argv[1] if len(sys.argv) > 1 else 'nvidia')