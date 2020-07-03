import tensorflow as tf
import numpy as np
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
        self.execute_index = 0

        # constructions
        self.optimizer_generator : tf.keras.optimizers.Optimizer = None
        self.optimizer_discriminator : tf.keras.optimizers.Optimizer = None
        self.writer = None

        # default parameters
        self.learning_rate_generator = 0.0001
        self.learning_rate_discriminator = 0.0001

    def build(self):
        self.optimizer_generator = tf.keras.optimizers.Adam(
                learning_rate=self._get_learning_rate_generator,
                beta_1=0.5,
                beta_2=0.9,)

        self.optimizer_discriminator = tf.keras.optimizers.Adam(
                learning_rate=self._get_learning_rate_discriminator,
                beta_1=0.5,
                beta_2=0.9,)

        self.built = True
        return self

    def add_summary_writer(self, path="."):
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

    def _get_learning_rate_generator(self):
        return self.learning_rate_generator

    def _get_learning_rate_discriminator(self):
        return self.learning_rate_discriminator

    def execute(self, strategy, update_steps, *args, **kwargs):
        if not self.built:
            raise RuntimeError("TrainBuilder has not been built yet")

        # choose strategy
        if strategy == "single":
            current_strategy = self._strategy_single
        elif strategy == "all":
            current_strategy = self._strategy_all
        elif strategy == "random":
            current_strategy = self._strategy_random
        elif strategy == "ensemble":
            current_strategy = self._strategy_ensemble
        elif strategy == "random ensemble":
            current_strategy = self._strategy_random_ensemble
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
                current_strategy(update_steps, *args, **kwargs)

            except KeyboardInterrupt:
                pass

            finally:
                self.execute_index += 1
                if self.writer is not None:
                    self.writer.flush()

            runtime = timeit.default_timer() - starttime
            log.info(f"training done in {runtime/60/60:.2f} hours")

        return self


    # training helper functions

    def train_generator(self, label, real, noise, variables):
        generator : tf.keras.Model = self.model.generator
        wasserstein_distance : tf.keras.Model = self.model.wasserstein_distance

        optimizer : tf.keras.optimizers.Optimizer = self.optimizer_generator

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)

            fake = generator([label, *noise])
            distance = wasserstein_distance([label, *real, *fake])

            loss = distance

        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradient, variables))

    def train_discriminator(self, label, real, noise, variables):
        generator : tf.keras.Model = self.model.generator
        wasserstein_distance : tf.keras.Model = self.model.wasserstein_distance
        gradient_penalty : tf.keras.Model = self.model.gradient_penalty

        optimizer : tf.keras.optimizers.Optimizer = self.optimizer_discriminator

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)

            fake = generator([label, *noise])
            distance = wasserstein_distance([label, *real, *fake])
            penalty = gradient_penalty([label, *real, *fake])

            loss = - distance + penalty

        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradient, variables))

    def make_summary(self, name, step):
        dataset = self.data.dataset

        generator : tf.keras.Model = self.model.generator
        wasserstein_distance : tf.keras.Model = self.model.wasserstein_distance
        gradient_penalty : tf.keras.Model = self.model.gradient_penalty

        distance = 0
        penalty = 0
        for label, real, noise in dataset.take(100):
            fake = generator([label, *noise])
            distance += wasserstein_distance([label, *real, *fake])
            penalty += gradient_penalty([label, *real, *fake])
        distance /= 10
        penalty /= 10

        tf.summary.scalar(
                "Wasserstein Distance - " + name,
                distance,
                step=step)

        tf.summary.scalar(
                "Gradient Penalty - " + name,
                penalty,
                step=step)


    # strategy implementations

    def _strategy_single(self, update_steps):
        dataset = self.data.dataset

        generator : tf.keras.Model = self.model.generator
        discriminator : tf.keras.Model = self.model.discriminator

        generator.ensemble = 0
        discriminator.ensemble = 0

        generator_variables = generator.trainable_weights
        discriminator_variables = discriminator.trainable_weights

        for step, (label, real, noise) in dataset.repeat().enumerate():
            if step >= update_steps:
                break

            if step % 100 == 0:
                # write summary each 100 steps
                self.make_summary(f"{self.execute_index} - single", step)

            if step % 6 == 5:
                # train generators
                discriminator.ensemble = 0
                for ensemble in range(generator.num_model):
                    generator.ensemble = ensemble+1
                    self.train_generator(
                            label, real, noise, generator_variables)

            else:
                # train discriminators
                generator.ensemble = 0
                for ensemble in range(discriminator.num_model):
                    discriminator.ensemble = ensemble+1
                    self.train_discriminator(
                            label, real, noise, discriminator_variables)

        # write summary after each strategy
        self.make_summary(f"{self.execute_index} - single", step)

    def _strategy_all(self, update_steps):
        dataset = self.data.dataset

        generator : tf.keras.Model = self.model.generator
        discriminator : tf.keras.Model = self.model.discriminator

        generator.ensemble = 0
        discriminator.ensemble = 0

        generator_variables = generator.trainable_weights
        discriminator_variables = discriminator.trainable_weights

        for step, (label, real, noise) in dataset.repeat().enumerate():
            if step >= update_steps:
                break

            if step % 100 == 0:
                # write summary each 100 steps
                self.make_summary(f"{self.execute_index} - all", step)
                generator.ensemble = 0
                discriminator.ensemble = 0

            if step % 6 == 5:
                # train generators
                self.train_generator(
                        label, real, noise, generator_variables)

            else:
                # train discriminators
                self.train_discriminator(
                        label, real, noise, discriminator_variables)

        # write summary after each strategy
        self.make_summary(f"{self.execute_index} - all", step)

    def _strategy_random(self, update_steps):
        dataset = self.data.dataset

        generator : tf.keras.Model = self.model.generator
        discriminator : tf.keras.Model = self.model.discriminator

        generator.ensemble = 0
        discriminator.ensemble = 0

        generator_variables = generator.trainable_weights
        discriminator_variables = discriminator.trainable_weights

        rng = np.random.default_rng()
        generator_ensemble = rng.choice(
                a=np.array([True, False]),
                size=(update_steps, generator.num_model))
        discriminator_ensemble = rng.choice(
                a=np.array([True, False]),
                size=(update_steps, discriminator.num_model))

        for step, (label, real, noise) in dataset.repeat().enumerate():
            if step >= update_steps:
                break

            if step % 100 == 0:
                # write summary each 100 steps
                self.make_summary(f"{self.execute_index} - random", step)

            if step % 6 == 5:
                # train generators
                generator.ensemble = generator_ensemble[step]
                discriminator.ensemble = discriminator_ensemble[step]
                self.train_generator(
                        label, real, noise, generator_variables)

            else:
                # train discriminators
                generator.ensemble = generator_ensemble[step]
                discriminator.ensemble = discriminator_ensemble[step]
                self.train_discriminator(
                        label, real, noise, discriminator_variables)

        # write summary after each strategy
        self.make_summary(f"{self.execute_index} - random", step)

    def _strategy_ensemble(self, update_steps):
        dataset = self.data.dataset

        generator : tf.keras.Model = self.model.generator
        discriminator : tf.keras.Model = self.model.discriminator

        generator.ensemble = 0
        discriminator.ensemble = 0

        generator_variables = generator.trainable_weights + [generator.ensemble_weights]
        discriminator_variables = discriminator.trainable_weights + [discriminator.ensemble_weights]

        for step, (label, real, noise) in dataset.repeat().enumerate():
            if step >= update_steps:
                break

            if step % 100 == 0:
                # write summary each 100 steps
                self.make_summary(f"{self.execute_index} - ensemble", step)
                generator.ensemble = 0
                discriminator.ensemble = 0

            if step % 6 == 5:
                # train generators
                self.train_generator(
                        label, real, noise, generator_variables)

            else:
                # train discriminators
                self.train_discriminator(
                        label, real, noise, discriminator_variables)

        # write summary after each strategy
        self.make_summary(f"{self.execute_index} - ensemble", step)

    def _strategy_random_ensemble(self, update_steps):
        dataset = self.data.dataset

        generator : tf.keras.Model = self.model.generator
        discriminator : tf.keras.Model = self.model.discriminator

        generator.ensemble = 0
        discriminator.ensemble = 0

        generator_variables = generator.trainable_weights + [generator.ensemble_weights]
        discriminator_variables = discriminator.trainable_weights + [discriminator.ensemble_weights]

        rng = np.random.default_rng()
        generator_ensemble = rng.choice(
                a=np.array([True, False]),
                size=(update_steps, generator.num_model))
        discriminator_ensemble = rng.choice(
                a=np.array([True, False]),
                size=(update_steps, discriminator.num_model))

        for step, (label, real, noise) in dataset.repeat().enumerate():
            if step >= update_steps:
                break

            if step % 100 == 0:
                # write summary each 100 steps
                self.make_summary(f"{self.execute_index} - random ensemble", step)

            if step % 6 == 5:
                # train generators
                generator.ensemble = generator_ensemble[step]
                discriminator.ensemble = discriminator_ensemble[step]
                self.train_generator(
                        label, real, noise, generator_variables)

            else:
                # train discriminators
                generator.ensemble = generator_ensemble[step]
                discriminator.ensemble = discriminator_ensemble[step]
                self.train_discriminator(
                        label, real, noise, discriminator_variables)

        # write summary after each strategy
        self.make_summary(f"{self.execute_index} - random ensemble", step)

