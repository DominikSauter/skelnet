"""Model builder

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Conv2DTranspose, Dropout
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.layers import LeakyReLU, Reshape
from keras.models import Model
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.applications.densenet import DenseNet121
from keras.regularizers import l2
from keras.layers.merge import add

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import numpy as np
import argparse


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 normalization=True,
                 conv_first=True,
                 transpose=False):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    if transpose:
        conv = Conv2DTranspose(num_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if normalization:
            x = InstanceNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if normalization:
            x = InstanceNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def encoder_layer(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    # bottleneck residual unit
    x = inputs
    y = resnet_layer(inputs=x,
                     num_filters=filters,
                     kernel_size=1,
                     strides=1,
                     activation=activation,
                     normalization=False,
                     conv_first=False)
    y = resnet_layer(inputs=y,
                     num_filters=filters,
                     kernel_size=kernel_size,
                     strides=1,
                     conv_first=False)
    y = resnet_layer(inputs=y,
                     num_filters=filters,
                     strides=strides,
                     kernel_size=1,
                     normalization=False,
                     conv_first=False)
    x = resnet_layer(inputs=x,
                     num_filters=filters,
                     kernel_size=1,
                     strides=strides,
                     activation=None,
                     normalization=False)

    x = add([x, y])
    return x


def build_model(input_shape, output_shape=None):
    base_model = DenseNet121(weights='imagenet', include_top=False)
    inputs = Input(shape=input_shape)
    x = concatenate([inputs, inputs, inputs], axis=-1)
    tail = Model(inputs, x)
    y = base_model.output
    y = encoder_layer(y,
                      256,
                      strides=2,
                      kernel_size=3)
    y = encoder_layer(y,
                      128,
                      strides=2,
                      kernel_size=3)
    y = encoder_layer(y,
                      64,
                      strides=2,
                      kernel_size=3)
    y = encoder_layer(y,
                      32,
                      strides=2,
                      kernel_size=3)
    y = GlobalAveragePooling2D()(y)
    y = Dense(630*4, activation='tanh')(y)
    y = Reshape((630, 4))(y)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=y)
    model.summary()
    g = Model(inputs, model(tail(inputs)))

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
            layer.trainable = False

    return g


def encoder_layer1(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation(activation)(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    # x = Dropout(0.2)(x)
    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):

    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation(activation)(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs])
    # x = Dropout(0.2)(x)
    return x


def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3,
                    name=None):

    channels = int(output_shape[-1])

    inputs1 = Input(shape=input_shape)
    e11 = encoder_layer(inputs1,
                        32,
                        strides=1,
                        kernel_size=kernel_size)
    # 256x256 x 32
    e12 = encoder_layer(e11,
                        64,
                        kernel_size=kernel_size)
    # 128x128 x 64
    e13 = encoder_layer(e12,
                        128,
                        kernel_size=kernel_size)
    # 64x64 x 128
    e14 = encoder_layer(e13,
                        256,
                        kernel_size=kernel_size)
    # 32x32 x 256
    e15 = encoder_layer(e14,
                        512,
                        kernel_size=kernel_size)
    # 16x16 x 512
    e16 = encoder_layer(e15,
                        1024,
                        kernel_size=kernel_size)
    # 8x8 x 1024
    e17 = encoder_layer(e16,
                        2048,
                        kernel_size=kernel_size)
    # 4x4 x 2048


    d11 = decoder_layer(e17,
                        e16,
                        1024,
                        kernel_size=kernel_size)
    # 8x8 x 1024+1024 
    d12 = decoder_layer(d11,
                        e15,
                        512,
                        kernel_size=kernel_size)
    # 16x16 x 512+512
    d13 = decoder_layer(d12,
                        e14,
                        256,
                        kernel_size=kernel_size)
    # 32x32 x 256+256
    d14 = decoder_layer(d13,
                        e13,
                        128,
                        kernel_size=kernel_size)
    # 64x64 x 128+128
    d15 = decoder_layer(d14,
                        e12,
                        64,
                        kernel_size=kernel_size)
    # 128x128 x 64+64
    d16 = decoder_layer(d15,
                        e11,
                        32,
                        kernel_size=kernel_size)
    # 256x256 x 32+32

    #o1 = InstanceNormalization()(d15)
    #o1 = Activation('relu')(o1)
    o1 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same')(d16)
    # 256x256 x channels


    inputs2 = Input(shape=input_shape)
    e21 = encoder_layer(inputs2,
                        32,
                        strides=1,
                        kernel_size=kernel_size)
    # 256x256 x 32
    e22 = encoder_layer(e21,
                        64,
                        strides=4,
                        kernel_size=kernel_size)
    # 64x64 x 64
    e23 = encoder_layer(e22,
                        128,
                        strides=4,
                        kernel_size=kernel_size)
    # 16x16 x 128
    e24 = encoder_layer(e23,
                        256,
                        strides=4,
                        kernel_size=kernel_size)
    # 4x4 x 256

    d21 = decoder_layer(e24,
                        e23,
                        128,
                        strides=4,
                        kernel_size=kernel_size)
    # 16x16 x 128+128
    d22 = decoder_layer(d21,
                        e22,
                        64,
                        strides=4,
                        kernel_size=kernel_size)
    # 64x64 x 64+64
    d23 = decoder_layer(d22,
                        e21,
                        32,
                        strides=4,
                        kernel_size=kernel_size)
    # 256x256 x 32+328
    # o2 = InstanceNormalization()(d22)
    # o2 = Activation('relu')(o2)
    o2 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same')(d23)
    # 256x256 x 1

    inputs3 = Input(shape=input_shape)
    e31 = encoder_layer(inputs3,
                        32,
                        strides=1,
                        kernel_size=kernel_size)
    # 256x256 x 32
    e32 = encoder_layer(e31,
                        64,
                        strides=8,
                        kernel_size=kernel_size)
    # 32x32 x 64
    e33 = encoder_layer(e32,
                        128,
                        strides=8,
                        kernel_size=kernel_size)
    # 4x4 x 128 

    d31 = decoder_layer(e33,
                        e32,
                        64,
                        strides=8,
                        kernel_size=kernel_size)
    # 32x32 x 64+64 
    d32 = decoder_layer(d31,
                        e31,
                        128,
                        strides=8,
                        kernel_size=kernel_size)
    # 256x256 x 128+128
    # o3 = InstanceNormalization()(d31)
    # o3 = Activation('relu')(o3)
    o3 = Conv2DTranspose(channels,
                         kernel_size=1,
                         strides=1,
                         padding='same')(d32)

    y = concatenate([o1, o2, o3])
    y = Activation('relu')(y)
    y = Conv2DTranspose(32,
                        kernel_size=1,
                        strides=1,
                        padding='same')(y)
    y = Activation('relu')(y)
    y = Conv2DTranspose(channels,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same')(y)
    y = Activation('relu')(y)
    y = Flatten()(y)
    y = Dense(64, activation='relu')(y)
    outputs1 = Dense(630, activation='tanh', name='x')(y)
    outputs2 = Dense(630, activation='tanh', name='y')(y)
    outputs3 = Dense(630, activation='tanh', name='r')(y)
    outputs4 = Dense(630, activation='sigmoid', name='p')(y)

    inputs = [inputs1, inputs2, inputs3]
    outputs = [outputs1, outputs2, outputs3, outputs4]
    generator = Model(inputs, outputs, name=name)

    return generator


def build_discriminator(input_shape,
                        output_shape,
                        kernel_size=3,
                        name=None):

    inputs = Input(shape=input_shape)
    outputs = Input(shape=output_shape)
    x = concatenate([inputs, outputs])
    x = encoder_layer(x,
                      32,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 128x128x32
    x = encoder_layer(x,
                      64,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 64x64x64
    x = encoder_layer(x,
                      128,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 32x32x128
    x = encoder_layer(x,
                      256,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 16x16x256
    x = encoder_layer(x,
                      512,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    # 8x8x512
    x = encoder_layer(x,
                      1024,
                      kernel_size=kernel_size,
                      strides=1,
                      activation='leaky_relu',
                      instance_norm=False)
    # 8x8x1024

    x = Flatten()(x)
    x = Dense(1)(x)
    preal = Activation('linear')(x)
    discriminator = Model([inputs, outputs], preal, name=name)

    return discriminator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Train cifar10 colorization"
    parser.add_argument("-c",
                        "--cifar10",
                        action='store_true',
                        help=help_)
    args = parser.parse_args()
    input_shape = (256, 256, 1)
    output_shape = (256, 256, 1)
    unet = build_unet(input_shape, output_shape)
    unet.summary()
    plot_model(unet, to_file='skelpix.png', show_shapes=True)



