import numpy as np
from numpy import random
from numpy import zeros
from numpy import ones
from numpy import array
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
Generate 20 fake samples with class labels
"""


def generate_fake_samples(n=20):
    # Generate X1 adenosine values in [0.00, 0.08]
    X1 = array([round(random.uniform(0.00, 0.09), 9) for i in range(n)])
    # Generate X2 zeatin values in [0.000, 0.006]
    X2 = array([round(random.uniform(0.000, 0.006), 9) for i in range(n)])

    # Stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))

    # Generate class labels
    y = zeros((n, 1))

    return X, y


"""
Train the discriminator model
- Plot the relationship between acc_real and acc_fake
"""


def train_discriminator(model,
                        n_epochs=1000,

                        # 20 fake and 20 real examples
                        n_batch=40):

    """
    Create empty lists to hold accuracy values
    """
    real = []
    fake = []

    half_batch = int(n_batch/2)

    """Run epochs manually"""
    for i in range(n_epochs):

        # Generate real examples
        X_real, y_real = generate_real_samples(half_batch)

        # Update model
        model.train_on_batch(X_real, y_real)

        # Generate fake examples
        X_fake, y_fake = generate_fake_samples(half_batch)

        # Update model
        model.train_on_batch(X_fake, y_fake)

        # Evaluate the model
        _, acc_real = model.evaluate(X_real, y_real, verbose=0)
        _, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)

        print(i,
              ' Acc_Real= ', acc_real,
              ' Acc_fake= ', acc_fake)

        real.append(acc_real)
        fake.append(acc_fake)

    """
    Plot the relationship between acc_real and acc_fake
    """
    plt.plot(real[970:999], fake[970:999])
    plt.xlabel('Acc_real')
    plt.ylabel('Acc_fake')
    # plt.show()


"""
Define the discriminator model
"""
model = define_discriminator()

"""
Fit the model
"""
train_discriminator(model)
