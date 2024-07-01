"""
Orange Labs
Authors : Colin Troisemaine
Maintainer : colin.troisemaine@gmail.com
"""
from utils import *

from models.fast_gpu_kmeans import fast_gpu_kmeans


class PBNModel(nn.Module):
    def __init__(self, input_size, pbn_hidden_layers, n_known_classes, n_clusters,
                 use_norm, use_batchnorm, activation_fct, p_dropout, app, USE_CUDA):
        """
        The PBN model object. It is composed of 3 main networks : The *encoder*, the *decoder* and the *classifier*.
        ToDo: Complete the documentation.
        """
        super(PBNModel, self).__init__()

        self.app = app
        self.device = setup_device(app, use_cuda=USE_CUDA)

        self.model_name = "pbn"

        self.use_norm = use_norm
        self.n_clusters = n_clusters

        # (1) Encoder
        self.encoder_layers = []

        # (1.1) First layer:
        self.encoder_layers.append(nn.Linear(input_size, pbn_hidden_layers[0]))

        # (1.2) Hidden layers:
        if len(pbn_hidden_layers) > 1:
            for i in range(1, len(pbn_hidden_layers)):
                if activation_fct is not None:
                    self.encoder_layers.append(get_activation_function(activation_fct))
                if use_batchnorm is True:
                    self.encoder_layers.append(nn.BatchNorm1d(num_features=pbn_hidden_layers[i - 1]))
                if p_dropout > 0:
                    self.encoder_layers.append(nn.Dropout(p=p_dropout))
                self.encoder_layers.append(nn.Linear(pbn_hidden_layers[i - 1], pbn_hidden_layers[i]))

        self.encoder_layers = nn.Sequential(*self.encoder_layers)

        # (2) Decoder
        self.decoder_layers = []
        decoder_layers_dims = list(reversed(pbn_hidden_layers))

        # (2.1) Hidden layers:
        if len(decoder_layers_dims) > 1:
            for i in range(1, len(decoder_layers_dims)):
                self.decoder_layers.append(nn.Linear(decoder_layers_dims[i - 1], decoder_layers_dims[i]))
                if activation_fct is not None:
                    self.decoder_layers.append(get_activation_function(activation_fct))
                if use_batchnorm is True:
                    self.decoder_layers.append(nn.BatchNorm1d(num_features=decoder_layers_dims[i]))
                if p_dropout > 0:
                    self.decoder_layers.append(nn.Dropout(p=p_dropout))

        # (2.2) Last layer:
        self.decoder_layers.append(nn.Linear(decoder_layers_dims[-1], input_size))

        self.decoder_layers = nn.Sequential(*self.decoder_layers)

        # (3) Classication layer
        self.classifier = nn.Linear(pbn_hidden_layers[-1], n_known_classes)

        self.to(self.device)

    def apply_norm(self, x):
        if self.use_norm is None:
            return x
        elif self.use_norm == "l1":
            return x / torch.linalg.norm(x, dim=1, ord=1).unsqueeze(-1)
        elif self.use_norm == "l2":
            return x / torch.linalg.norm(x, dim=1, ord=2).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown norm: {self.use_norm}")

    def encoder_forward(self, x):
        z = self.encoder_layers(x)
        z = self.apply_norm(z)
        return z

    def decoder_forward(self, encoded_x):
        return self.decoder_layers(encoded_x)

    def classifier_forward(self, encoded_x):
        return self.classifier(encoded_x)

    def predict_new_data(self, x_unknown):
        self.eval()
        with torch.no_grad():
            x_unknown = torch.tensor(x_unknown, device=self.device, dtype=torch.float)

            projected_x_unknown = self.encoder_layers(x_unknown)

            km = fast_gpu_kmeans(k_clusters=self.n_clusters)
            clustering_prediction = km.fit_predict(projected_x_unknown, n_init=1).cpu().numpy()

        self.train()

        return clustering_prediction

    def evaluate_classif_accuracy(self, x_known, y_known):
        self.eval()
        with torch.no_grad():
            y_pred = self.classifier_forward(self.encoder_forward(x_known))
            y_pred = F.softmax(y_pred, -1).argmax(dim=1)
        self.train()
        return accuracy_score(y_known, y_pred.cpu())
