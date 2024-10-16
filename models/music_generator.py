import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D, Dropout

class MusicGeneratorGAN:
    def __init__(self):
        # Input shape
        self.noise_dim = 100
        self.piano_roll_shape = (128, 500, 1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), 
                                   loss='binary_crossentropy', metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build and compile the GAN
        self.gan = self.build_gan()

    def build_generator(self):
        model = tf.keras.Sequential()
        model.add(Dense(16 * 62 * 128, input_dim=self.noise_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((16, 62, 128)))
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(1, (7, 7), activation='tanh', padding='same'))
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.piano_roll_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    def build_gan(self):
        # Freeze the discriminator during generator training
        self.discriminator.trainable = False

        model = tf.keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
        return model

    def train(self, data, epochs=10000, batch_size=32, save_interval=1000):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_piano_rolls = data[idx]
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            gen_piano_rolls = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_piano_rolls, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_piano_rolls, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            g_loss = self.gan.train_on_batch(noise, valid)

            # Output training status
            if epoch % save_interval == 0:
                print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

# Example usage to initialize and train the GAN
if __name__ == "__main__":
    data = np.load('data/midi/processed_piano_rolls.npy')
    data = data.reshape(data.shape[0], 128, 500, 1)

    gan = MusicGeneratorGAN()
    gan.train(data, epochs=5000, batch_size=32, save_interval=500)
