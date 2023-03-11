import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf

class Tester:
    def __init__(self, model, input, loss_fn, test_loss, metric):
        self.model = model
        self.input = input

        self.loss_fn = loss_fn
        self.test_loss = test_loss
        self.metric = metric

    @tf.function
    def test_step(self, batch):
        images, labels = batch

        predictions = self.model(images, training = False)
        loss = self.loss_fn(labels, predictions)

        self.test_loss.update_state(loss)
        self.metric.update_state(labels, predictions)

        return loss, predictions

    def evaluate(self):
        print('\nModel testing...', file = sys.stderr)

        step_loss = 0.0
        for step, testing_batch in enumerate(self.input):
            step_loss, predictions = self.test_step(testing_batch)

        test_loss = self.test_loss.result()
        metric = self.metric.result()

        print(f'Test loss: {test_loss} test accuracy: {metric}\n', file = sys.stderr)
