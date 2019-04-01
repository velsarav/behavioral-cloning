from keras.models import Model
from keras.layers import Cropping2D, Conv2D, MaxPool2D, \
    Flatten, Dropout, Dense, concatenate


def get_nvidia(image, velocity):
    # Inputs
    x = Cropping2D(cropping=((60, 20), (0, 0)))(image)

    # Convolution tower
    x = Conv2D(filters=24, kernel_size=5, activation='relu')(x)
    x = MaxPool2D(strides=(2, 2))(x)
    x = Conv2D(filters=36, kernel_size=5, activation='relu')(x)
    x = MaxPool2D(strides=(2, 2))(x)
    x = Conv2D(filters=48, kernel_size=5, activation='relu')(x)
    x = MaxPool2D(strides=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu')(x)

    # Flattening for fully-connected
    x = Flatten()(x)
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
    model = Model(inputs=[image, velocity], outputs=[xa, xt])
    return model