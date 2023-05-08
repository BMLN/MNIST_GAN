import tensorflow.keras as keras
import numpy as np
from matplotlib import pyplot



# def l_v(shape=(1,)):

#     lv_shape = tuple( x for x in generator.layers[0].input_shape if x is not None )
#     lv_arr = [ np.random.normal(0, 1, lv_shape) for x in range(np.product(shape)) ]
#     lv_arr = np.reshape(lv_arr, shape + lv_shape)

#     return lv_arr



class Gan():
    l_v = None

    def __init__(self, discriminator, generator, gan_loss, gan_optimizer):
        self.discriminator = discriminator
        self.discriminator.trainable = False
        self.generator = generator
        
        self.model = keras.Sequential()
        self.model.add(generator)
        self.model.add(discriminator)
        self.model.compile(loss=gan_loss, optimizer=gan_optimizer)

    def __call__(self, inputs=None, n=1):
        if inputs is None:
            X = Gan.l_v(shape= (n,) + tuple( x for x in self.generator.layers[0].input_shape if x is not None ))
        else:
            X = inputs

        return self.generator(X)

    def update(self, X_disc, y_disc, X_gan):
        d_loss, _ = self.discriminator.train_on_batch(X_disc, y_disc)
        g_loss = self.model.train_on_batch(X_gan, np.ones((len(X_gan), 1)))

        return d_loss, g_loss

    def snapshot(self, name, n=100, save=True, plot=True):
        out_model, out_plot = None, None
        samples = self(n=n)

        if plot:
            for i, sample in enumerate(samples):
                pyplot.subplot(int(np.sqrt(n)), int(np.sqrt(n)), 1 + i)
                pyplot.axis('off')
                pyplot.imshow(sample, cmap='gray_r')
            
            out_plot = name + "_plot.png"
            pyplot.savefig(out_plot)
            pyplot.close()
             
        if save:
            out_model = name + "_model.h5" 
            self.model.save(out_model) 

        return out_model, out_plot







#depr
if __name__ == "__main__":
    discriminator = keras.Sequential()
    discriminator.add(keras.layers.Conv2D(64, (3,3), strides=(2,2), padding="same", input_shape=(28,28,3)))
    discriminator.add(keras.layers.BatchNormalization()) 
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(keras.layers.Conv2D(64, (3,3), strides=(2,2), padding="same"))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(1, activation="sigmoid"))

    discriminator.compile(loss="binary_crossentropy")


    generator = keras.Sequential()
    generator.add(keras.layers.Dense(64 * 7 * 7, input_dim=100))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    generator.add(keras.layers.Reshape((7, 7, 64)))

    generator.add(keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding="same"))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU(alpha=0.2))

    generator.add(keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding="same"))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU(alpha=0.2))
    generator.add(keras.layers.Conv2D(3, (3,3), activation="tanh", padding="same"))


    


    c = Gan(discriminator, generator, "binary_crossentropy", keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    Gan.l_v = l_v