#sparse Autoencoder

from tensorflow import keras
from tensorflow.keras import layers,regularizers

import matplotlib.pyplot as plt
import numpy as np

(x_train,_),(x_test,_) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1,784).astype("float32") /255
x_test = x_test.reshape(-1,784).astype("float32")/255

input_ = keras.Input(shape=(784,))
encoded = layers.Dense(128,activation='relu',activity_regularizer=regularizers.l1(1e-5))(input_)
decoded = layers.Dense(784,activation='sigmoid')(encoded)

sparse_autoencoder = keras.Model(input_,decoded)
sparse_autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
sparse_autoencoder.summary()

sparse_autoencoder.fit(x_train,x_train,epochs=20,batch_size=256,shuffle=True,validation_data=(x_test,x_test))


def plot_sparse_reconstructions(model,data):
    decoded_imgs=model.predict(data[:10])
    plt.figure(figsize=(20,4))
    for i in range(10):
        ax=plt.subplot(2,10,i+1)
        plt.imshow(data[i].reshape(28,28),cmap="gray")
        plt.axis("off")

        ax = plt.subplot(2,10,i+11)
        plt.imshow(data[i].reshape(28,28),cmap="gray")
        plt.axis("off")
    plt.show()

plot_sparse_reconstructions(sparse_autoencoder,x_test)

# This network is trained to reconstruct its input just like a normal autoencoder.
# The key difference is the L1 penalty on the activation layer, encouraging sparsity.
