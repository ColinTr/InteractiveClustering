# Software Name : InteractiveClustering
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

from models.ThreadedTrainingTask import ThreadedTrainingTask, KilledException
from utils import *
import threading
import gc


class PBNThreadedTrainingTask(ThreadedTrainingTask):
    def __init__(self, dataset_name, target_name, known_classes, unknown_classes,
                 selected_features, random_state, color_by, model_config, PBNModelToTrain,
                 lr, epochs, w, batch_size, x_full, y_train_classifier, unknown_class_value):
        super().__init__(PBNModelToTrain.app, dataset_name, target_name, known_classes, unknown_classes,
                         selected_features, random_state, color_by, model_config, PBNModelToTrain.model_name)
        self.app = PBNModelToTrain.app
        self.model_to_train = PBNModelToTrain
        self.x_full = x_full
        self.y_train_classifier = y_train_classifier
        self.unknown_class_value = unknown_class_value
        self.device = PBNModelToTrain.device

        # Joint training parameters
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.w = w

        # Event that will be set when .stop() is called on this thread
        self._stop_event = threading.Event()

    def run(self):
        losses_dict = {
            'train_losses': [],
            'ce_losses': [],
            'mse_losses': [],
        }

        self.app.logger.debug(self.model_to_train)

        try:
            # ========================== Joint learning ========================
            optimizer = torch.optim.AdamW(self.model_to_train.parameters(), lr=self.lr)

            ce_loss_func = nn.CrossEntropyLoss()
            mse_loss_func = nn.MSELoss(reduction="sum")

            n_batchs = math.ceil((self.x_full.shape[0]) / self.batch_size)

            n_current_training_step = 0
            n_total_training_step = n_batchs * self.epochs

            for epoch in range(self.epochs):
                self.app.logger.debug("Training progress at {:.1f}%...".format(self.progress_percentage))

                train_losses, train_ce_losses, train_mse_losses = [], [], []

                batch_start_index, batch_end_index = 0, min(self.batch_size, len(self.x_full))
                for batch_index in range(n_batchs):
                    batch_x_train = self.x_full[batch_start_index:batch_end_index]
                    batch_y_train = self.y_train_classifier[batch_start_index:batch_end_index]
                    batch_y_train = torch.tensor(batch_y_train, dtype=torch.int64, device=self.x_full.device)
                    mask_known = batch_y_train != self.unknown_class_value

                    if len(batch_x_train) < 2:
                        print("Skipping batch of size 1...")
                        continue

                    optimizer.zero_grad()

                    # ========== forward ==========
                    # (1) Encode all the data
                    encoded_batch_x = self.model_to_train.encoder_forward(batch_x_train)

                    # (2) Reconstruct all the data
                    reconstructed_batch_x = self.model_to_train.decoder_forward(encoded_batch_x)

                    # (3) Learn to classify the known data only
                    y_known_pred = self.model_to_train.classifier_forward(encoded_batch_x[mask_known])
                    # =============================

                    ce_loss = ce_loss_func(y_known_pred, batch_y_train[mask_known])
                    mse_loss = mse_loss_func(reconstructed_batch_x, batch_x_train) / len(batch_x_train)

                    full_loss = self.w * ce_loss + (1 - self.w) * mse_loss

                    full_loss.backward()

                    optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(full_loss.item())
                    train_ce_losses.append(ce_loss.item())
                    train_mse_losses.append(mse_loss.item())

                    n_current_training_step += 1
                    self.progress_percentage = (n_current_training_step / n_total_training_step) * 100

                    batch_start_index += self.batch_size
                    batch_end_index = min((batch_end_index + self.batch_size), self.x_full.shape[0])

                    if self.stopped() is True:
                        raise KilledException

            # ==================================================================

        except KilledException:
            self.app.logger.debug("Thread received KilledException, stopping training...")
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as err:
            self.app.logger.debug(err)
            self.error_message = err.args[0]
            torch.cuda.empty_cache()
            gc.collect()

        torch.cuda.empty_cache()  # Free some memory up

        return losses_dict, self.model_to_train
