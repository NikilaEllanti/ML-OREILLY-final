# 1.Dense Stacked Autoencoder

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
#preprocessing
#reshapes the 28x28 Fashion MNIST images into flat 784-dimensional vectors
#and normalizes their pixel values from [0, 255] to [0, 1] as float32 type

(x_train,_),(x_test, _) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1,784).astype("float32") /255
x_test = x_test.reshape(-1,784).astype("float32") / 255

input_ = keras.Input(shape=(784,))
h1=layers.Dense(256,activation="relu")(input_)
h2=layers.Dense(64,activation="relu")(h1)
code=layers.Dense(32,activation="relu")(h2)
h3=layers.Dense(64,activation="relu")(code)
h4=layers.Dense(256,activation="relu")(h3)
output_=layers.Dense(784,activation="sigmoid")(h4)

autoencoder = keras.Model(input_,output_)
autoencoder.compile(optimizer="adam",loss="binary_crossentropy")
autoencoder.fit(x_train,x_train,epochs=20,batch_size=256,validation_data=(x_test,x_test))

def plot_reconstructions(model,data):
    decoded_imgs = model.predict(data[:10])
    plt.figure(figsize=(20,4))
    for i in range(10):
        ax = plt.subplot(2,10,i+1)
        plt.imshow(decoded_imgs[i].reshape(28,28),cmap="gray")
        plt.axis("off")
        ax = plt.subplot(2,10,i+11)
        plt.imshow(decoded_imgs[i].reshape(28,28),cmap="gray")
    plt.show()
plot_reconstructions(autoencoder,x_test)
