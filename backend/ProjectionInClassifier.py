"""
Orange Labs
Authors : Colin Troisemaine
Maintainer : colin.troisemaine@gmail.com
"""
from time import sleep

from TabularNCD import get_simple_layer
from sklearn.cluster import KMeans
from torch import nn
import numpy as np
import threading
import utils
import torch
import math


class NewThreadedTrainingTask(threading.Thread):
    def __init__(self, ProjectionInClassifierModelToTrain, x_train, y_train, batch_size, num_epochs):
        super(NewThreadedTrainingTask, self).__init__()
        self.progress_percentage = 0
        self.model_to_train = ProjectionInClassifierModelToTrain
        self.x_train = torch.tensor(x_train, device=ProjectionInClassifierModelToTrain.device, dtype=torch.float)
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = ProjectionInClassifierModelToTrain.device
        self.app = ProjectionInClassifierModelToTrain.app

    def run(self):
        optimizer = torch.optim.AdamW(self.model_to_train.parameters(), lr=self.model_to_train.learning_rate)

        losses_dict = {
            'epoch_mean_train_losses': [],
            'epoch_mean_train_acc': [],
        }

        n_batchs = math.ceil((self.x_train.shape[0]) / self.batch_size)

        n_current_training_step = 0
        n_total_training_step = n_batchs * self.num_epochs

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

                sleep(1)

            losses_dict['epoch_mean_train_losses'].append(np.mean(train_losses))
            # losses_dict['epoch_mean_train_acc'].append(evaluate_supervised_model_accuracy(self.x_train, self.y_train, self.model_to_train))

        self.app.logger.info("Training complete")


class ProjectionInClassifier(nn.Module):
    def __init__(self, app, layers_sizes, n_clusters, p_dropout, activation_fct, learning_rate):
        super(ProjectionInClassifier, self).__init__()

        if len(layers_sizes) > 2:
            encoder_layers = []
            for i in range(1, len(layers_sizes) - 1):
                simple_layer = get_simple_layer(layers_sizes[i - 1], layers_sizes[i],
                                                add_dropout=True, p_dropout=p_dropout,
                                                activation_fct=activation_fct)
                [encoder_layers.append(layer) for layer in simple_layer]

            self.encoder = nn.Sequential(*encoder_layers)

        self.classifier = nn.Linear(layers_sizes[-2], layers_sizes[-1])

        self.learning_rate = learning_rate
        self.n_clusters = n_clusters
        self.app = app
        self.device = utils.setup_device(app, use_cuda=True)
        self.to(self.device)

    def encoder_forward(self, x):
        return self.encoder(x)

    def classifier_forward(self, encoded_x):
        return self.classifier(encoded_x)

    def predict_new_data(self, new_data):
        projected_new_data = self.encoder(new_data)

        kmeans_model = KMeans(n_clusters=self.n_clusters, n_init="auto")

        clustering_prediction = kmeans_model.fit_predict(projected_new_data)

        return clustering_prediction
