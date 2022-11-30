import numpy as np
from numpy.random import randn
from numpy import zeros
from numpy import ones
from numpy import hstack
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt


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


"""
Generate n (20) real samples with class labels
"""


def generate_real_samples(n=20):

    """Load the data source"""
    df = pd.read_csv('Table_1.csv')

    """Extract values"""
    # Lterr samples only
    adenosine = df['Ade_ug/g'][:20]
    # Lterr samples only
    zeatin = df['Zeatin_ug/g'][:20]

    """Stack arrays"""
    adeno = np.array(adenosine).reshape(n, 1)
    zea = np.array(zeatin).reshape(n, 1)
    X = hstack((adeno, zea))

    """Generate class labels"""
    y = ones((n, 1))

    return X, y


"""
Generate points in latent space as 
        inputs for the generator
"""


def generate_latent_points(latent_dim, n):

    # Generate points in latent space
    x_input = randn(latent_dim * n)

    # Reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)

    return x_input


"""
Use the generator to generate n fake examples
        (20, or any number) and plot the results
"""


def generate_fake_samples(
        generator,
        latent_dim,
        # n=20, or any number
        n):

    # Generate points in latent space
    x_input = generate_latent_points(latent_dim, n)

    # Predict outputs
    X = generator.predict(x_input)

    # Create class labels
    y = zeros((n, 1))

    return X, y


"""
Evaluate the discriminator and plot 
real and fake points
"""


def summarize_performance(epoch, generator,
                          discriminator,
                          latent_dim, n=20):

    # Prepare real and fake samples
    x_real, y_real = generate_real_samples(n)

    x_fake, y_fake = generate_fake_samples(generator,
                                           latent_dim,
                                           n)

    # Evaluate discriminator on real and fake samples
    _, acc_real = discriminator.evaluate(x_real,
                                         y_real,
                                         verbose=0)

    _, acc_fake = discriminator.evaluate(x_fake,
                                         y_fake,
                                         verbose=0)

    # Summarize discriminator performance
    print(epoch, acc_real, acc_fake)

    # Scatter plot real and fake data points
    plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')

    # Save plot to file
    filename = 'generated_plot_e%03d.jpg' % (epoch + 1)
    plt.savefig(filename)
    plt.close()


"""Train the generator and the discriminator"""


def train(g_model, d_model, gan_model, latent_dim,

          # Training for 10,000 epochs!
          n_epochs=10000,

          n_batch=40,
          n_eval=2000):

    half_batch = int(n_batch/2)

    """Manually enumerate epochs"""
    for i in range(n_epochs):

        # Prepare real and fake samples
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

        # Update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)

        # Prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)

        # Create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))

        # Update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)

        # Evaluate the model every n_eval epochs
        if (i+1) % n_eval == 0:
            summarize_performance(
                i, g_model, d_model, latent_dim)


"""Size of the latent space"""
latent_dim = 5

"""Create the generator and the discriminator"""
generator = define_generator(latent_dim)
discriminator = define_discriminator()

"""Create the GAN"""
gan_model = define_gan(generator, discriminator)

"""Train model"""
train(generator, discriminator, gan_model, latent_dim)
