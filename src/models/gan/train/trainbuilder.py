import tensorflow as tf
import os
import timeit
from contextlib import nullcontext

import src.utils
log = src.utils.getLogger(__name__)


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
        summarypath = os.path.join(path, "tensorboard")

        if self.writer is not None:
            self.writer.flush()

        self.writer = tf.summary.create_file_writer(summarypath)

        return self

    def learning_rate(self, generator, discriminator):
        if generator is not None:
            self.learning_rate_generator = generator

        if discriminator is not None:
            self.learning_rate_discriminator = discriminator

        return self

    def execute(self, strategy, epochs):
        if not self.built:
            raise RuntimeError("TrainBuilder has not been built yet")

        # choose strategy
        if strategy == "single":
            current_strategy = self.strategy_single
        else:
            raise NotImplementedError(f"unknown training strategy: {strategy}")

        # prepare summary writer, if any
        if self.writer is None:
            writer_context = nullcontext()
        else:
            writer_context = self.writer.as_default()

        # run strategy
        with writer_context:
            log.info(f"training started with strategy: {strategy}")
            starttime = timeit.default_timer()

            try:
                current_strategy(epochs)

            except KeyboardInterrupt:
                pass

            finally:
                if self.writer is not None:
                    self.writer.flush()

            runtime = timeit.default_timer() - starttime
            log.info(f"training done in {runtime/60/60:.2f} hours")

        return self

    def strategy_single(self, epochs):
        dataset = self.data.dataset

        generator = self.model.generator
        discriminator = self.model.discriminator
        wasserstein_distance = self.model.wasserstein_distance
        gradient_penalty = self.model.gradient_penalty

        for epoch in range(epochs):
            for ii, (label, real, noise) in dataset.enumerate():
                pass
