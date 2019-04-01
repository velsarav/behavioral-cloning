from keras.models import Model
from keras.layers import Cropping2D, Lambda, Flatten, Dense, Dropout, concatenate
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet


def get_vgg16(image, velocity):
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


def resize(image, size=(160, 160)):
    from keras.backend import tf
    return tf.image.resize_images(image, size)


def get_mobilenet(image, velocity):
    # Inputs
    image = Cropping2D(cropping=((60, 20), (0, 0)))(image)
    image = Lambda(resize)(image)

    # Load pretrained MobileNet
    base_model = MobileNet(
        input_shape=(160, 160, 3),
        input_tensor=image, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # Flattening for fully-connected
    x = base_model.output
    x = Flatten()(x)

    # Steering tower
    xa = Dense(100, activation='elu')(x)
    xa = Dense(50, activation='elu')(xa)
    xa = Dense(10, activation='elu')(xa)
    xa = Dense(1, name='steering')(xa)

    # Throttle tower
    xt = concatenate([x, velocity])
    xt = Dropout(.5)(xt)
    xt = Dense(10, activation='relu')(xt)
    xt = Dense(1, name='throttle')(xt)

    # Define model with multiple inputs and outputs
    model = Model(inputs=[base_model.input, velocity], outputs=[xa, xt])
    return model