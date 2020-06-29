import tensorflow as tf
import os


class TrainBuilder():

    def __init__(self, databuilder, modelbuilder):
        self.data = databuilder
        self.model = modelbuilder

        self.built = False

        # constructions
        self.optimizer_generator : tf.keras.optimizers.Optimizer = None
        self.optimizer_discriminator : tf.keras.optimizers.Optimizer = None
        self.writer = None

        # default parameters
        self.learning_rate_generator = 0.0001
        self.learning_rate_discriminator = 0.0001

    def build(self):
        self.optimizer_generator = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_generator,
                beta_1=0.5,
                beta_2=0.9,)

        self.optimizer_discriminator = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate_discriminator,
                beta_1=0.5,
                beta_2=0.9,)

        self.built = True
        return self

    def summary_writer(self, path):
        logpath = os.path.join(path, "tensorboard")

        if self.writer is not None:
            self.writer.flush()

        self.writer = tf.summary.create_file_writer(logpath)

        return self

    def learning_rate(self, generator, discriminator):
        if generator is not None:
            self.learning_rate_generator = generator

        if discriminator is not None:
            self.learning_rate_discriminator = discriminator

        return self

    def execute(self, train_type, epochs):
        if not self.built:
            raise RuntimeError("TrainBuilder has not been built yet")
        pass

