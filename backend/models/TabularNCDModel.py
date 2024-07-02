# Software Name : InteractiveClustering
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

from utils import *


class TabularNCDModel(nn.Module):
    def __init__(self, input_size, hidden_layers_sizes,  n_known_classes, n_unknown_classes,
                 activation_fct, p_dropout, use_batchnorm, app, USE_CUDA):
        """
        The TabularNCD model object. It is composed of 5 main networks : The *encoder*, for SSL the *mask_vector_estimator*
        and *feature_vector_estimator* and for the joint learning the *classification_head* and *clustering_head*.
        :param hidden_layers_sizes: Sizes of the hidden layers.
        :param n_known_classes: The number of known classes, or the number of output neurons of the classification head.
        :param n_unknown_classes:  The number of unknown classes, or the number of output neurons of the clustering head.
        :param activation_fct: The activation function that is *between* the first and last layers of each network. Choices : ['relu', 'sigmoid', None].
        :param p_dropout: The probability of dropout. Use p_dropout=0 for no dropout.
        """
        super(TabularNCDModel, self).__init__()

        self.app = app
        self.device = setup_device(app, use_cuda=USE_CUDA)

        self.model_name = "tabularncd"

        # ==================== Encoder ====================
        self.encoder_layers = []

        # First layer:
        self.encoder_layers.append(nn.Linear(input_size, hidden_layers_sizes[0]))

        # Hidden layers:
        if len(hidden_layers_sizes) > 1:
            for i in range(1, len(hidden_layers_sizes)):
                if activation_fct is not None:
                    self.encoder_layers.append(get_activation_function(activation_fct))
                if use_batchnorm is True:
                    self.encoder_layers.append(nn.BatchNorm1d(num_features=hidden_layers_sizes[i - 1]))
                if p_dropout > 0:
                    self.encoder_layers.append(nn.Dropout(p=p_dropout))
                self.encoder_layers.append(nn.Linear(hidden_layers_sizes[i - 1], hidden_layers_sizes[i]))

        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        # =================================================

        # ================= Joint learning ================
        self.classification_head = nn.Linear(hidden_layers_sizes[-1], n_known_classes)
        self.clustering_head = nn.Linear(hidden_layers_sizes[-1], n_unknown_classes)
        # =================================================

        self.to(self.device)

    def encoder_forward(self, x):
        return self.encoder_layers(x)

    def classification_head_forward(self, encoded_x):
        return self.classification_head(encoded_x)

    def clustering_head_forward(self, encoded_x):
        return self.clustering_head(encoded_x)

    def predict_new_data(self, new_data):
        self.eval()
        with torch.no_grad():
            new_data = torch.tensor(new_data, device=self.device, dtype=torch.float)
            clustering_prediction = F.softmax(self.clustering_head_forward(self.encoder_forward(new_data)), -1).argmax(dim=1).cpu().numpy()
        self.train()
        return clustering_prediction
