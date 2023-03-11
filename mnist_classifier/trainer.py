import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf

class Trainer:
    def __init__(self, model, input, model_save_path, loss_fn, optimizer, train_loss, metric, epoches):
        self.model = model
        self.input = input

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.metric = metric

        self.epoches = epoches

        # self.model_save_path = 'saved_models/'
        self.model_save_path = model_save_path

    @tf.function
    def train_step(self, batch):
        images, labels = batch

        with tf.GradientTape() as tape:
            predictions = self.model(images, training = True)
            step_loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(step_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(step_loss)
        self.metric.update_state(labels, predictions)

        return step_loss, predictions

    def train(self):
        print('\nModel training...', file = sys.stderr)

        for epoch in range(self.epoches):
            step_loss = 0.0
            for step, training_batch in enumerate(self.input):
                step_loss, predictions = self.train_step(training_batch)

            train_acc = self.metric.result()
            train_loss = self.train_loss.result()

            self.metric.reset_states()
            self.train_loss.reset_states()

            print(f'Epoch: {epoch} train loss: {train_loss} train accuracy: {train_acc}', file = sys.stderr)

            # save_path = os.path.join(self.model_save_path)
            self.model.save(os.path.join(self.model_save_path, 'model.h5'))
