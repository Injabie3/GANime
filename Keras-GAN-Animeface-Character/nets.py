#!/usr/bin/env python3
import os
import sys
import numpy as np
import random
from keras import models
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers import Input
from keras.optimizers import Adam, Adagrad, Adadelta, Adamax, SGD
from keras.callbacks import CSVLogger
# GAN doesn't like spare gradients (says ganhack). LeakyReLU better.
from keras.layers.advanced_activations import LeakyReLU
import scipy
import h5py
from args import Args
from data import denormalize4gan
from layers import bilinear2x
from discrimination import MinibatchDiscrimination

#import tensorflow as tf
#import keras
#keras.backend.get_session().run(tf.initialize_all_variables())



def build_enc( shape ) :
    return build_discriminator(shape, build_disc=False)



def build_discriminator( shape, build_disc=True ) :
    '''
    Build discriminator.
    Set build_disc=False to build an encoder network to test
    the encoding/discrimination capability with autoencoder...
    '''
    def conv2d( x, filters, shape=(4, 4), **kwargs ) :
        '''
        I don't want to write lengthy parameters so I made a short hand function.
        '''
        x = Conv2D( filters, shape, strides=(2, 2),
            padding='same',
            kernel_initializer=Args.kernel_initializer,
            **kwargs )( x )
        #x = MaxPooling2D()( x )
        x = BatchNormalization(momentum=Args.bn_momentum)( x )
        x = LeakyReLU(alpha=Args.alpha_D)( x )
        return x

    # https://github.com/tdrussell/IllustrationGAN
    # As proposed by them, unlike GAN hacks, MaxPooling works better for anime dataset it seems.
    # However, animeGAN doesn't use it so I'll keep it more similar to DCGAN.

    face = Input( shape=shape )
    x = face

    # Warning: Don't batchnorm the first set of Conv2D.
    x = Conv2D( 64, (4, 4), strides=(2, 2),
        padding='same',
        kernel_initializer=Args.kernel_initializer )( x )
    x = LeakyReLU(alpha=Args.alpha_D)( x )
    # 32x32

    x = conv2d( x, 128 )
    # 16x16

    x = conv2d( x, 256 )
    # 8x8

    x = conv2d( x, 512 )
    # 4x4

    if build_disc:
        x = Flatten()(x)
        # add 16 features. Run 1D conv of size 3.
        #x = MinibatchDiscrimination(16, 3)( x )

        #x = Dense(1024, kernel_initializer=Args.kernel_initializer)( x )
        #x = LeakyReLU(alpha=Args.alpha_D)( x )

        # 1 when "real", 0 when "fake".
        x = Dense(1, activation='sigmoid',
            kernel_initializer=Args.kernel_initializer)( x )
        return models.Model( inputs=face, outputs=x )
    else:
        # build encoder.
        x = Conv2D(Args.noise_shape[2], (4, 4), activation='tanh')(x)
        return models.Model( inputs=face, outputs=x )



def build_gen( shape ) :
    def deconv2d( x, filters, shape=(4, 4) ) :
        '''
        Conv2DTransposed gives me checkerboard artifact...
        Select one of the 3.
        '''
        # Simpe Conv2DTranspose
        # Not good, compared to upsample + conv2d below.

        # Run the input through transposed 2D Convolution (deconvolution), with:
        # - filters number of output filters
        # - Use a shape[0] x shape[1] kernel
        # and use the kernel_initializer from args.py, which is currently
        # using "glorot_uniform".
        x= Conv2DTranspose( filters, shape, padding='same',
            strides=(2, 2), kernel_initializer=Args.kernel_initializer )(x)

        # simple and works
        # UpSampling2D repeats the rows and columns of the data by size[0] and
        # size[1], respectively.
        #x = UpSampling2D( (2, 2) )( x )

        # Conv2D runs the inputs through 2D convolution with:
        # - filters number of output filters
        # - Use a shape[0] by shape[1] kernel
        #x = Conv2D( filters, shape, padding='same' )( x )

        # Bilinear2x... Not sure if it is without bug, not tested yet.
        # Tend to make output blurry though
        #x = bilinear2x( x, filters )
        #x = Conv2D( filters, shape, padding='same' )( x )

        x = BatchNormalization(momentum=Args.bn_momentum)( x )
        x = LeakyReLU(alpha=Args.alpha_G)( x )
        return x

    # https://github.com/tdrussell/IllustrationGAN  z predictor...?
    # might help. Not sure.

    # 01. Build input layer.
    noise = Input( shape=Args.noise_shape )
    x = noise
    # 1x1x256
    # noise is not useful for generating images.

    # 02. Run the input through transposed 2D Convolution (deconvolution), with:
    # - 512 output filters
    # - Use a 4 x 4 kernel
    # and use the kernel_initializer from args.py, which is currently
    # using "glorot_uniform".
    x = Conv2DTranspose( 512, (4, 4),
        kernel_initializer=Args.kernel_initializer )(x)

    # 03. Add another layer, this time we normalize, keep the activation
    # mean close to 0 and standard deviation close to 1.
    x = BatchNormalization(momentum=Args.bn_momentum)( x )

    # 04. Add another layer for Leaky ReLU, and is recommended by GANHacks.
    # LeakyReLU is similar to standard ReLU, except we allow some of it to
    # go through by multiplying it by alpha.
    # f(x) = alpha * x    , x < 0
    # f(x) = x            , x >= 0
    x = LeakyReLU(alpha=Args.alpha_G)( x )

    # 4x4
    x = deconv2d( x, 256 )
    # 8x8
    x = deconv2d( x, 128 )
    # 16x16
    x = deconv2d( x, 64 )
    # 32x32

    # Extra layer
    x = Conv2D( 64, (3, 3), padding='same',
        kernel_initializer=Args.kernel_initializer )( x )
    x = BatchNormalization(momentum=Args.bn_momentum)( x )
    x = LeakyReLU(alpha=Args.alpha_G)( x )
    # 32x32

    x= Conv2DTranspose( 3, (4, 4), padding='same', activation='tanh',
        strides=(2, 2), kernel_initializer=Args.kernel_initializer )(x)
    # 64x64

    return models.Model( inputs=noise, outputs=x )
