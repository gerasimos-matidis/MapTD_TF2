import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disables INFO & WARNING logs 
import tensorflow as tf
import keras
from keras.applications.resnet import ResNet50

UNPOOL_LAYERS_FILTERS = [128, 64, 32]
SKIP_LAYERS = ['conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out']
CONV_REGULARIZER = keras.regularizers.l2(1e-5)
BN_PARAMS = {
    'momentum' : 0.997,
    'epsilon': 1e-5,
    'scale': True,
}

# Resize the inputs by using bilinear interpolation
def resize_bln(inputs, resize_factor=2):
    return tf.image.resize(inputs, size=[tf.shape(inputs)[1] * resize_factor, 
        tf.shape(inputs)[2] * resize_factor])

def unpool_block(model, x, conv_filters = None, skip_layer_name=None, 
        index=None, bn_parameters=None, conv_regularizer=None):
    x = keras.layers.Lambda(resize_bln, name=f'unpool_{index + 1}')(x)
    x = keras.layers.concatenate([x, model.get_layer(skip_layer_name).output], 
        axis=-1, name=f'concatenate_{index + 1}')
    x = keras.layers.Conv2D(conv_filters, 1, padding='same', 
        kernel_regularizer=conv_regularizer)(x)
    x = keras.layers.BatchNormalization(**bn_parameters)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(conv_filters, 3, padding='same', 
        kernel_regularizer=conv_regularizer)(x)
    x = keras.layers.BatchNormalization(**bn_parameters)(x)
    x = keras.layers.Activation('relu')(x)

    return x

def maptd_model(input_shape=(None, None, 3), scoremap_acf='sigmoid'):
    inputs = keras.layers.Input(shape=input_shape)
    resnet_backbone = ResNet50(input_tensor=inputs, include_top=False)
    x = resnet_backbone.get_layer(index=-1).output
    for i in range(3):
        x = unpool_block(resnet_backbone, x, 
            conv_filters=UNPOOL_LAYERS_FILTERS[i], 
            skip_layer_name=SKIP_LAYERS[i], bn_parameters=BN_PARAMS, 
            index=i,
            conv_regularizer=CONV_REGULARIZER)
    x = keras.layers.Conv2D(32, 3, padding='same', 
        kernel_regularizer=CONV_REGULARIZER)(x)
    x = keras.layers.BatchNormalization(**BN_PARAMS)(x)
    x = keras.layers.Activation('relu')(x)
    predictions = keras.layers.Conv2D(1, 1, activation=scoremap_acf, 
        name='predictions_vector')(x)
    # NOTE: Not sure about the value of the rboxes_scale. Jerod, in his code, has put
    # the number 512. I here use the size of the rows of the patch. It is ok when we deal
    # with square patches. But what happens when we deal with non-square patches 
    # (e.g. when we predict from overlapping non-square patches).
    rboxes_scale = 512 # NOTE: Must change for training with different Input Shape
    rboxes = rboxes_scale * keras.layers.Conv2D(4, 1, activation='sigmoid', 
        name='region_boxes_vector')(x)
    angles_factor = 2 # NOTE: Factor to multiply by the angles vector, because 
    # the codomain of tf.atan belongs to (-pi, pi) (see next line)
    angles = angles_factor * keras.layers.Conv2D(1, 1, activation=tf.atan, # NOTE: I changed tf.math.atan to 'sigmoid' just to check
        name='angles_vector')(x)
    geometry = keras.layers.Concatenate(name='geometry_vector')([rboxes, angles])
    
    return keras.Model(inputs=inputs, outputs=[predictions, geometry])
    