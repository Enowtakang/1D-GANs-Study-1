from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model


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


"""Define the generator model"""
model = define_generator(5)

"""Summarize model"""
model.summary()

"""plot the model"""
plot_model(model, to_file='generator.png',
           show_shapes=True,
           show_layer_names=True)
