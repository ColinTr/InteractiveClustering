"""
Orange Labs
Authors : Colin Troisemaine
Maintainer : colin.troisemaine@gmail.com
"""

from models.ThreadedTrainingTask import KilledException, ThreadedTrainingTask
from torch import nn
import numpy as np
import threading
import torch
import math


class ProjectionInClassifierThreadedTrainingTask(ThreadedTrainingTask):
    def __init__(self, dataset_name, target_name, known_classes, unknown_classes, selected_features, random_state, color_by, model_config, corresponding_tsne_config_name, ProjectionInClassifierModelToTrain, x_train, y_train, batch_size, num_epochs):
        super().__init__(ProjectionInClassifierModelToTrain.app, dataset_name, target_name, known_classes, unknown_classes, selected_features, random_state, color_by, model_config, corresponding_tsne_config_name, ProjectionInClassifierModelToTrain.model_name)
        self.model_to_train = ProjectionInClassifierModelToTrain
        self.x_train = torch.tensor(x_train, device=ProjectionInClassifierModelToTrain.device, dtype=torch.float)
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = ProjectionInClassifierModelToTrain.device
        self.app = ProjectionInClassifierModelToTrain.app

        # Event that will be set when .stop() is called on this thread
        self._stop_event = threading.Event()

    def run(self):
        optimizer = torch.optim.AdamW(self.model_to_train.parameters(), lr=self.model_to_train.learning_rate)

        losses_dict = {
            'epoch_mean_train_losses': [],
            'epoch_mean_train_acc': [],
        }

        n_batchs = math.ceil((self.x_train.shape[0]) / self.batch_size)
        n_current_training_step = 0
        n_total_training_step = n_batchs * self.num_epochs

        try:
            for epoch in range(self.num_epochs):
                self.app.logger.debug("Training progress at {:.1f}%...".format(self.progress_percentage))

                train_losses = []

                batch_start_index, batch_end_index = 0, min(self.batch_size, len(self.x_train))
                for batch_index in range(n_batchs):
                    batch_x_train = self.x_train[batch_start_index:batch_end_index]
                    batch_y_train = self.y_train[batch_start_index:batch_end_index]
                    batch_y_train = torch.tensor(batch_y_train, dtype=torch.int64, device=self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    encoded_batch_x = self.model_to_train.encoder_forward(batch_x_train)
                    y_pred = self.model_to_train.classifier_forward(encoded_batch_x)

                    # compute loss
                    supervised_loss = nn.CrossEntropyLoss()(y_pred, batch_y_train)

                    # backward
                    supervised_loss.backward()

                    # update the weights using gradient descent
                    optimizer.step()

                    # Save loss for plotting purposes
                    train_losses.append(supervised_loss.item())

                    n_current_training_step += 1
                    self.progress_percentage = (n_current_training_step / n_total_training_step) * 100

                    batch_start_index += self.batch_size
                    batch_end_index = min((batch_end_index + self.batch_size), self.x_train.shape[0])

                    if self.stopped() is True:
                        raise KilledException

                losses_dict['epoch_mean_train_losses'].append(np.mean(train_losses))
                # losses_dict['epoch_mean_train_acc'].append(evaluate_supervised_model_accuracy(self.x_train, self.y_train, self.model_to_train))

            self.app.logger.info("Training complete")

        except KilledException:
            print("Training thread stopping...")
            # ToDo optional cleanup
