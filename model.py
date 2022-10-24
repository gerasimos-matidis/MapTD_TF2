import tensorflow as tf
import keras
from keras.applications.resnet import ResNet50
from keras.utils import plot_model

UNPOOL_LAYERS_FILTERS = [128, 64, 32]
SKIP_LAYERS = ['conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out']
CONV_REGULARIZER = keras.regularizers.l2(1e-5)
BN_PARAMS = {
    'momentum' : 0.997,
    'epsilon': 1e-5,
    'scale': True,
}

def resize_bln(inputs, resize_factor=2):
    return tf.image.resize(inputs, size=[tf.shape(inputs)[1] * resize_factor, tf.shape(inputs)[2] * resize_factor])

def unpool_block(model, x, conv_filters = None, skip_layer_name=None, index=None, bn_parameters=None, conv_regularizer=None):
    x = keras.layers.Lambda(resize_bln, name=f'unpool_{index}')(x)
    x = keras.layers.concatenate([x, model.get_layer(skip_layer_name).output], axis=-1)
    x = keras.layers.Conv2D(conv_filters, (1, 1), padding='same', kernel_regularizer=conv_regularizer)(x)
    x = keras.layers.BatchNormalization(**bn_parameters)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(conv_filters, (3, 3), padding='same', kernel_regularizer=conv_regularizer)(x)
    x = keras.layers.BatchNormalization(**bn_parameters)(x)
    x = keras.layers.Activation('relu')(x)

    return x

def maptd_model(input_shape=(512, 512, 3)):
    inputs = keras.layers.Input(shape=input_shape)
    resnet_backbone = ResNet50(input_tensor=inputs, include_top=False)
    x = resnet_backbone.get_layer(index=-1).output
    for i in range(3):
        x = unpool_block(resnet_backbone, x, conv_filters=UNPOOL_LAYERS_FILTERS[i], skip_layer_name=SKIP_LAYERS[i], bn_parameters=BN_PARAMS, conv_regularizer=CONV_REGULARIZER)
    x = keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=CONV_REGULARIZER)(x)
    x = keras.layers.BatchNormalization(**BN_PARAMS)(x)
    x = keras.layers.Activation('relu')(x)






