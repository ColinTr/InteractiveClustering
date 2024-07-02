# Software Name : InteractiveClustering
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.

from abc import abstractmethod
import threading


class KilledException(Exception):
    pass


class ThreadedTrainingTask(threading.Thread):
    def __init__(self, app, dataset_name, target_name, known_classes, unknown_classes, selected_features, random_state, color_by, model_config, model_name):
        super().__init__()

        # Variables for image generation and server requests
        self.progress_percentage = 0
        self.app = app
        self.dataset_name = dataset_name
        self.target_name = target_name
        self.known_classes = known_classes
        self.unknown_classes = unknown_classes
        self.selected_features = selected_features
        self.random_state = random_state
        self.color_by = color_by
        self.model_config = model_config
        self.model_name = model_name

        self.model_to_train = None

        self.error_message = None

        # Event that will be set when .stop() is called on this thread
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    @abstractmethod
    def run(self):
        # This is where the training of the model should occur
        pass
