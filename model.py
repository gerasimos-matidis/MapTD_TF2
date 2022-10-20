from pyexpat import model
import tensorflow as tf
from keras.applications.resnet import ResNet50

resnet_backbone = ResNet50(include_top=False, input_shape=(512, 512, 3))

# TODO: Check if the default parameters for Batch Normalization in TensorFlow are the same 
# with the ones which Jerod defines within function "outputs" in "model.py"


def unpool(inputs,new_size=None):
    #NOTE: Conversion to TF2 of Jerod's function
    """
    Double the feature map patial size via bilinear upsampling

    Parameters
     inputs: feature map of rank 4 to be upsampled (changes in spatial size)
     new_size: Two-tuple (width, height) of new size for outputs. If not given, 
               default behavior doubles the size.
    Returns
     outputs: upsized rank 4 tensor. If inputs shape is [a, b, c, d], outputs
               shape is (a, new_size[0], new_size[1], d)
    """
    if not new_size:
        new_size = [tf.shape(inputs)[1]*2, tf.shape(inputs)[2]*2]
        
    return tf.image.resize( inputs, size=new_size)

