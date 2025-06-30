from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

data = keras.datasets.fashion_mnist
(x_train,_),(x_test,_) = data.load_data()

x_train = x_train.astype("float32") /255.
x_test = x_test.astype("float32") /255.

x_train = np.reshape(x_train,(-1,28,28,1))
x_test = np.reshape(x_test,(-1,28,28,1))

input_img = keras.Input(shape=(28,28,1))

x = layers.Conv2D(32,(3,3),activation = 'relu',padding ='same')(input_img)
x = layers.MaxPooling2D((2,2),padding ='same')(x)
x = layers.Conv2D(16,(2,2),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2,2),padding='same')(x)

x = layers.Conv2D(16,(3,3),activation='relu',padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)
x = layers.Conv2D(32,(3,3),activation='relu',padding ='same')(x)
x = layers.UpSampling2D((2,2))(x)
decoded = layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)


conv_autoencoder = keras.Model(input_img,decoded)
conv_autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
conv_autoencoder.summary()
conv_autoencoder.fit(x_train,x_train,epochs=20,batch_size =128,shuffle=True,validation_data=(x_test,x_test))

def plot_conv_reconstructions(model,data):
    decoded_imgs= model.predict(data[:10])
    plt.figure(figsize=(20,4))
    for i in range(10):
        ax=plt.subplot(2,10,i+1)
        plt.imshow(data[i].reshape(28,28),cmap="gray")
        plt.axis("off")
      
        ax = plt.subplot(2,10,i+11)
        plt.imshow(decoded_imgs[i].reshape(28,28),cmap="gray")
        plt.axis("off")
    plt.show()
plot_conv_reconstructions(conv_autoencoder,x_test)


# - Convolutional Autoencoders preserve spatial features (edges, textures)
# - Use Conv2D + Pooling to compress and Conv2D + UpSampling to reconstruct
# - Perfect for image reconstruction and dimensionality reduction