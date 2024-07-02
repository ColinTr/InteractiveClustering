# Software Name : InteractiveClustering
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

from sklearn.cluster import KMeans
from torch import nn
import numpy as np
import utils
import torch


class ProjectionInClassifierModel(nn.Module):
    def __init__(self, app, layers_sizes, n_clusters, p_dropout, activation_fct, learning_rate, USE_CUDA):
        super(ProjectionInClassifierModel, self).__init__()

        # Architecture
        if len(layers_sizes) > 2:
            encoder_layers = []
            for i in range(1, len(layers_sizes) - 1):
                simple_layer = utils.get_simple_layer(layers_sizes[i - 1], layers_sizes[i],
                                                      add_dropout=True, p_dropout=p_dropout,
                                                      activation_fct=activation_fct)
                [encoder_layers.append(layer) for layer in simple_layer]

            self.encoder = nn.Sequential(*encoder_layers)

        self.classifier = nn.Linear(layers_sizes[-2], layers_sizes[-1])

        # Variables for training
        self.app = app
        self.learning_rate = learning_rate
        self.n_clusters = n_clusters
        self.device = utils.setup_device(app, use_cuda=USE_CUDA)
        self.to(self.device)
        self.model_name = "projection_in_classifier"

    def encoder_forward(self, x):
        return self.encoder(x)

    def classifier_forward(self, encoded_x):
        return self.classifier(encoded_x)

    def predict_new_data(self, new_data):
        self.eval()

        with torch.no_grad():

            new_data = torch.tensor(new_data, device=self.device, dtype=torch.float)

            projected_new_data = self.encoder(new_data)
            projected_new_data = np.array(projected_new_data.cpu())

            kmeans_model = KMeans(n_clusters=self.n_clusters, n_init="auto")

            clustering_prediction = kmeans_model.fit_predict(projected_new_data)

        self.train()

        return clustering_prediction
