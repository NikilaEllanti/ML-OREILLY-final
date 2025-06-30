from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

(x_train,_),(x_test, _)= keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32")/255.
x_test=x_test.astype("float32")/255.
x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)

def add_noise(data,noise_factor=0.5):
    noisy = data + noise_factor * np.random.normal(loc=0.0,scale =1.0,size=data.shape)
    return np.clip(noisy,0.,1.)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

input_ = keras.Input(shape=(784,))
encoded = layers.Dense(128,activation='relu')(input_)
decoded = layers.Dense(784,activation="sigmoid")(encoded)

denoise_autoencoder = keras.Model(input_,decoded)
denoise_autoencoder.compile(optimizer="adam",loss="binary_crossentropy")
denoise_autoencoder.summary()

denoise_autoencoder.fit(x_train_noisy,x_train,
                        epochs=20,batch_size=256,shuffle=True,
                        validation_data=(x_test_noisy,x_test))


def plot_denoising_reconstructions(model,noisy_data,clean_data):
    decoded_imgs = model.predict(noisy_data[:10])
    plt.figure(figsize=(30,6))
    for i in range(10):
        ax= plt.subplot(3,10,i+1)
        plt.imshow(clean_data[i].reshape(28,28),cmap="gray")
        plt.axis("off")

        ax=plt.subplot(3,10,i+11)
        plt.imshow(noisy_data[i].reshape(28,28),cmap="gray")
        plt.axis("off")

        ax=plt.subplot(3,10,i+21)
        plt.imshow(decoded_imgs[i].reshape(28,28),cmap="gray")
        plt.axis("off")
    plt.show()


plot_denoising_reconstructions(denoise_autoencoder,x_test_noisy,x_test)


# - This teaches the model to learn robust features and ignore irrelevant noise.
# - Great for tasks where input data may be corrupted (e.g., real-world sensors, OCR, etc.)
