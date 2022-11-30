from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model


"""Define discriminator model"""


def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(
        # 5 nodes
        5,
        # then, the rest
        activation='relu',
        kernel_initializer='he_uniform',
        input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


"""
Define generator model
"""


def define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(
        15, activation='relu',
        kernel_initializer='he_uniform',
        input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model


"""
Define the combined generator and 
    discriminator model, for updating
    the generator
"""


def define_gan(generator, discriminator):
    # Make weights in the discriminator not trainable
    discriminator.trainable = False

    # Connect them
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam')

    return model


"""Size of the latent space"""
latent_dim = 5

"""Create generator and discriminator"""
generator = define_generator(latent_dim)
discriminator = define_discriminator()

"""Create the GAN"""
gan_model = define_gan(generator, discriminator)

"""Summarize the GAN model"""
gan_model.summary()

"""Plot the GAN model"""
plot_model(gan_model,
           to_file='GAN_plot.jpg',
           show_shapes=True,
           show_layer_names=True)
