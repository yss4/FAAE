import matplotlib as mpl

mpl.use('Agg')

from keras.layers import Reshape, Flatten, Lambda
from keras.layers import Input
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import pandas as pd
import numpy as np
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial.legacy import AveragePooling2D, BatchNormalization, l1l2, Dense, fit, Convolution2D
from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from cifar10_utils import cifar10_data
from keras.layers import LeakyReLU, Activation
from image_utils import dim_ordering_unfix, dim_ordering_shape
import os
import random
from keras.utils import plot_model

def model_generator(latent_dim, units=512, dropout=0.5, reg=lambda: l1l2(l1=1e-7, l2=1e-7)):
    model = Sequential(name="decoder")
    h = 5
    model.add(Dense(units * 4 * 4, input_dim=latent_dim, W_regularizer=reg()))
    model.add(BatchNormalization(mode=0))
    model.add(Reshape(dim_ordering_shape((units, 4, 4))))
    model.add(Convolution2D(units / 2, h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(units / 2, h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(units / 4, h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, h, h, border_mode='same', W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


def model_encoder(latent_dim, input_shape, units=512, reg=lambda: l1l2(l1=1e-7, l2=1e-7), dropout=0.5):
    k = 5
    x = Input(input_shape)
    h = Convolution2D(units / 4, k, k, border_mode='same', W_regularizer=reg())(x)
    # h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units / 2, k, k, border_mode='same', W_regularizer=reg())(h)
    # h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units / 2, k, k, border_mode='same', W_regularizer=reg())(h)
    # h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units, k, k, border_mode='same', W_regularizer=reg())(h)
    # h = SpatialDropout2D(dropout)(h)
    h = LeakyReLU(0.2)(h)
    h = Flatten()(h)
    mu = Dense(latent_dim, name="encoder_mu", W_regularizer=reg())(h)
    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", W_regularizer=reg())(h)
    z = Lambda(lambda (_mu, _lss): _mu + K.random_normal(K.shape(_mu)) * K.exp(_lss / 2),
               output_shape=lambda (_mu, _lss): _mu)([mu, log_sigma_sq])
    return Model(x, z, name="encoder")


def model_discriminator():
    nch = 512
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)

    c1 = Convolution2D(int(nch / 4), h, h, border_mode='same', W_regularizer=reg(),
                       input_shape=dim_ordering_shape((3, 32, 32)))
    c2 = Convolution2D(int(nch / 2), h, h, border_mode='same', W_regularizer=reg())
    c3 = Convolution2D(nch, h, h, border_mode='same', W_regularizer=reg())
    c4 = Convolution2D(1, h, h, border_mode='same', W_regularizer=reg())

    model = Sequential()
    model.add(c1)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(c2)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(c3)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(LeakyReLU(0.2))
    model.add(c4)
    model.add(AveragePooling2D(pool_size=(4, 4), border_mode='valid'))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    return model


def example_aae(path, adversarial_optimizer):
    # z \in R^100
    latent_dim = 256
    units = 512
    # x \in R^{28x28}
    input_shape = dim_ordering_shape((3, 32, 32))

    # generator (z -> x)
    generator = model_generator(latent_dim, units=units)
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape, units=units)
    # autoencoder (x -> x')
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
    # discriminator (z -> y)
    discriminator = model_discriminator()

    # build AAE
    zreal = discriminator.inputs[0]
    x = generator.inputs[0]
    z = generator(x)
    xpred = encoder(z)
    yreal = discriminator(zreal)
    yfake = discriminator(z)
    aae = Model([zreal, x], fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))

    # print summary of models
    generator.summary()
    encoder.summary()
    discriminator.summary()

    
    #plot_model(generator, to_file='generator.png', show_shapes=True, show_layer_names=False)
    #plot_model(encoder, to_file = 'encoder.png',show_shapes=True, show_layer_names=False)
    #plot_model(discriminator, to_file = 'discriminator.png', show_shapes=True, show_layer_names=False)

    #import pdb;pdb.set_trace()

    encoder.load_weights(os.path.join(path, "encoder.h5"))
    generator.load_weights(os.path.join(path, "generator.h5"))
    discriminator.load_weights(os.path.join(path, "discriminator.h5"))
    
    # build adversarial model
    generative_params = generator.trainable_weights + encoder.trainable_weights
    model = AdversarialModel(base_model=aae,
                             player_params=[generative_params, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[Adam(3e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                              loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
                                    "xpred": "mean_squared_error"},
                              player_compile_kwargs=[{"loss_weights": {"yfake": 1, "yreal": 1,
                                                                       "xpred": 8}}] * 2)

    xtrain, xtest = cifar10_data()
    # callback for image grid of generated samples
    def generator_sampler():
        zsamples = np.random.randn(10 * 10, latent_dim)
    #    zsamples = 1/(1+np.exp(-zsamples))
        return dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1)).reshape((10, 10, 32, 32, 3))

    generator_cb = ImageGridCallback(os.path.join(path, "generated-epoch-{:03d}.png"), generator_sampler)

    # callback for image grid of autoencoded samples
    def autoencoder_sampler():
        xsamples = n_choice(xtest, 10)
        xrep = np.repeat(xsamples, 9, axis=0)
        xgen = dim_ordering_unfix(autoencoder.predict(xrep)).reshape((10, 9, 3, 32, 32))
        xsamples = dim_ordering_unfix(xsamples).reshape((10, 1, 3, 32, 32))
        samples = np.concatenate((xsamples, xgen), axis=1)
        samples = samples.transpose((0, 1, 3, 4, 2))
        return samples

    autoencoder_cb = ImageGridCallback(os.path.join(path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler,
                                       cmap=None)
    
    # train network
    # generator, discriminator; pred, yfake, yreal
    #n = xtrain.shape[0]
    #y = [xtrain, np.ones((n, 1)), np.zeros((n, 1)), xtrain, np.zeros((n, 1)), np.ones((n, 1))]
    #ntest = xtest.shape[0]
    #ytest = [xtest, np.ones((ntest, 1)), np.zeros((ntest, 1)), xtest, np.zeros((ntest, 1)), np.ones((ntest, 1))]
    #history = fit(model, x=xtrain, y=y, validation_data=(xtest, ytest),
    #              callbacks=[generator_cb, autoencoder_cb],
    #             nb_epoch=100, batch_size=32)

    
    train_datagen = gen_sample(128, 256, False)
    test_datagen = gen_sample(32, 256, True)
    history = model.fit_generator(train_datagen, epochs = 200, steps_per_epoch= 1000, validation_data=test_datagen, validation_steps = 100, callbacks=[generator_cb, autoencoder_cb])
    # save history
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "history.csv"))

    # save model
    encoder.save(os.path.join(path, "encoder.h5"))
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))

def gen_sample(batch_size, latent_dim, test):
    xtrain, xtest = cifar10_data()
    if test:
        data = xtest
        limit = xtest.shape[0]
    else:
        data = xtrain
        limit = xtrain.shape[0]

    while True:
        noise = np.random.randn(batch_size, latent_dim)
    #    noise = 1/(1+np.exp(-noise))
        yield [data[random.sample(range(limit), batch_size)], noise], [noise, np.ones((batch_size, 1)), np.zeros((batch_size, 1)), noise, np.zeros((batch_size, 1)), np.ones((batch_size, 1))] 

def gen_sample_new(batch_size, latent_dim, test):
    directory = 'new'
    files = os.listdir(directory)    
    limit = len(files)

    while True:
        noise = np.array(list(map(lambda x: np.random.normal(size=latent_dim), range(batch_size))))
        batch_indicies = random.sample(range(limit), batch_size)
        batch_images = []
        for i in batch_indices:
	    filename = os.path.join(directory, files[i])
	    image = scipy.misc.imread(filename).astype(np.float)
            image = image.astype(np.float32) / 255.0
	    batch_images.append(image)
        yield [batch_images, noise], [noise, np.ones((batch_size, 1)), np.zeros((batch_size, 1)), noise, np.zeros((batch_size, 1)), np.ones((batch_size, 1))]

def main():
    example_aae("output/aae-cifar10", AdversarialOptimizerSimultaneous())


if __name__ == "__main__":
    main()
