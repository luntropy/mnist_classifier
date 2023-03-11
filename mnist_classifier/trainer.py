import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf

class Trainer:
    def __init__(self, model, input, validation_input, model_save_path, loss_fn, optimizer, train_loss, metric, val_loss, val_metric, epoches):
        self.model = model
        self.input = input
        self.validation_input = validation_input

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.metric = metric

        self.val_loss = val_loss
        self.val_metric = val_metric

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

    @tf.function
    def validation_step(self, batch):
        images, labels = batch

        predictions = self.model(images, training = False)
        loss = self.loss_fn(labels, predictions)

        self.val_loss.update_state(loss)
        self.val_metric.update_state(labels, predictions)

    def train(self):
        print('\nModel training...', file = sys.stderr)

        for epoch in range(self.epoches):
            step_loss = 0.0
            for step, training_batch in enumerate(self.input):
                step_loss, predictions = self.train_step(training_batch)

            for step, validation_batch in enumerate(self.validation_input):
                self.validation_step(validation_batch)

            train_acc = self.metric.result()
            train_loss = self.train_loss.result()

            val_loss = self.val_loss.result()
            val_metric = self.val_metric.result()

            self.metric.reset_states()
            self.train_loss.reset_states()

            self.val_loss.reset_states()
            self.val_metric.reset_states()

            print(f'Epoch: {epoch} train loss: {train_loss} train accuracy: {train_acc} validation loss: {val_loss} validation accuracy {val_metric}', file = sys.stderr)

            # save_path = os.path.join(self.model_save_path)
            self.model.save(os.path.join(self.model_save_path, 'model.h5'))
