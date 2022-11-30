from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt


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

    # Plot the results
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


"""Size of the latent space"""
latent_dim = 5

"""Define the generator model"""
model = define_generator(latent_dim)

"""Generate and plot generated samples"""
generate_fake_samples(
    model,
    latent_dim,
    # n
    100)
