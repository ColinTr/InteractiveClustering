"""
Orange Labs
Authors : Colin Troisemaine
Maintainer : colin.troisemaine@gmail.com
"""
import gc

from models.ThreadedTrainingTask import ThreadedTrainingTask, KilledException
from ncl_memory_module import NCLMemoryModule
from utils import *
import threading


class TabularNCDThreadedTrainingTask(ThreadedTrainingTask):
    def __init__(self, dataset_name, target_name, known_classes, unknown_classes, selected_features, random_state, color_by, model_config, TabularNCDModelToTrain,
                 M, lr, epochs, k_neighbors, w1, w2, topk, batch_size, x_full, y_train_classifier, unknown_class_value):
        super().__init__(TabularNCDModelToTrain.app, dataset_name, target_name, known_classes, unknown_classes, selected_features, random_state, color_by, model_config, TabularNCDModelToTrain.model_name)
        self.app = TabularNCDModelToTrain.app
        self.model_to_train = TabularNCDModelToTrain
        self.x_full = x_full
        self.y_train_classifier = y_train_classifier
        self.unknown_class_value = unknown_class_value

        # Joint training parameters
        self.M = M
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.k_neighbors = k_neighbors
        self.w1 = w1
        self.w2 = w2
        self.topk = topk

        # Event that will be set when .stop() is called on this thread
        self._stop_event = threading.Event()

    def run(self):
        losses_dict = {
            # Losses
            'full loss': [],
            'classification loss': [], 'ce loss': [], 'cs classification loss': [],
            'clustering loss': [], 'bce loss': [], 'cs clustering loss': [],
        }

        self.app.logger.debug(self.model_to_train)

        try:
            device = self.model_to_train.device

            unlab_memory_module = NCLMemoryModule(device, M=self.M, labeled_memory=False)
            lab_memory_module = NCLMemoryModule(device, M=self.M, labeled_memory=True)

            optimizer = torch.optim.AdamW(self.model_to_train.parameters(), lr=self.lr)

            cross_entropy_loss = nn.CrossEntropyLoss()
            mse_loss = nn.MSELoss()

            n_batchs = math.ceil((self.x_full.shape[0]) / self.batch_size)
            n_current_training_step = 0
            n_total_training_step = n_batchs * self.epochs

            for epoch in range(self.epochs):
                self.app.logger.debug("Training progress at {:.1f}%...".format(self.progress_percentage))

                train_full_losses = []
                train_classification_losses = []
                train_clustering_losses = []
                train_bce_losses = []
                train_ce_losses = []
                train_cs_classification_losses = []
                train_cs_clustering_losses = []

                batch_start_index, batch_end_index = 0, self.batch_size
                for batch_index in range(n_batchs):
                    optimizer.zero_grad()

                    # (1) ===== Get the data =====
                    batch_x_train = self.x_full[batch_start_index:batch_end_index]
                    batch_y_train = self.y_train_classifier[batch_start_index:batch_end_index]

                    mask_unlab = batch_y_train == self.unknown_class_value
                    mask_lab = ~mask_unlab

                    if sum(mask_unlab) < 2 or sum(mask_lab) < 2:
                        print("Skipping batch of size 1...")
                        continue

                    assert mask_unlab.sum() > 0, "No unlabeled data in batch"

                    # Augment/Transform the data
                    with torch.no_grad():
                        augmented_x_unlab = smotenc_transform_batch_2(batch_x_train[mask_unlab], None,
                                                                      unlab_memory_module.original_data_memory, device,
                                                                      k_neighbors=self.k_neighbors)
                        augmented_x_lab = smotenc_transform_batch_2(batch_x_train[mask_lab], None,
                                                                    lab_memory_module.original_data_memory, device,
                                                                    k_neighbors=self.k_neighbors)

                    encoded_x = self.model_to_train.encoder_forward(batch_x_train)
                    encoded_x_unlab = encoded_x[mask_unlab]

                    # (2) ===== Forward the classification data and compute the losses =====
                    y_pred_lab = self.model_to_train.classification_head_forward(encoded_x)

                    augmented_y_pred = torch.zeros(y_pred_lab.shape, device=device)
                    encoded_augmented_x_unlab = self.model_to_train.encoder_forward(augmented_x_unlab)
                    augmented_y_pred[mask_unlab] = self.model_to_train.classification_head_forward(encoded_augmented_x_unlab)
                    encoded_augmented_x_lab = self.model_to_train.encoder_forward(augmented_x_lab)
                    augmented_y_pred[mask_lab] = self.model_to_train.classification_head_forward(encoded_augmented_x_lab)

                    ce_loss = cross_entropy_loss(y_pred_lab, torch.tensor(batch_y_train, device=device))

                    cs_loss_classifier = mse_loss(y_pred_lab, augmented_y_pred)

                    classifier_loss = self.w1 * ce_loss + (1 - self.w1) * cs_loss_classifier

                    # (3) ===== Forward the clustering data and compute the losses =====
                    y_pred_unlab = self.model_to_train.clustering_head_forward(encoded_x_unlab)

                    encoded_augmented_x_unlab = self.model_to_train.encoder_forward(augmented_x_unlab)
                    augmented_y_pred_unlab = self.model_to_train.clustering_head_forward(encoded_augmented_x_unlab)

                    # ========== Define the pseudo labels ==========
                    computed_top_k = int((self.topk / 100) * len(encoded_x_unlab))

                    # Because it is symmetric, we compute the upper corner and copy it to the lower corner
                    upper_list_1, upper_list_2 = np.triu_indices(len(encoded_x_unlab), k=1)
                    unlab_unlab_similarities = nn.CosineSimilarity()(encoded_x_unlab[upper_list_1],
                                                                     encoded_x_unlab[upper_list_2])
                    similarity_matrix = torch.zeros((len(encoded_x_unlab), len(encoded_x_unlab)), device=device)
                    similarity_matrix[upper_list_1, upper_list_2] = unlab_unlab_similarities
                    similarity_matrix += similarity_matrix.T.clone()

                    top_k_most_similar_instances_per_instance = similarity_matrix.argsort(descending=True)[:,
                                                                :computed_top_k]

                    pseudo_labels_matrix = torch.zeros((len(encoded_x_unlab), len(encoded_x_unlab)), device=device)
                    pseudo_labels_matrix = pseudo_labels_matrix.scatter_(index=top_k_most_similar_instances_per_instance, dim=1, value=1)

                    # The matrix isn't symmetric, because the graph is directed
                    # So if there is one link between two points, regardless of the direction, we consider this pair to be positive
                    pseudo_labels_matrix += pseudo_labels_matrix.T.clone()
                    pseudo_labels_matrix[pseudo_labels_matrix > 1] = 1  # Some links will overlap
                    pseudo_labels = pseudo_labels_matrix[upper_list_1, upper_list_2]
                    # ==============================================

                    bce_loss = unsupervised_classification_loss(y_pred_unlab[upper_list_1], y_pred_unlab[upper_list_2], pseudo_labels)

                    cs_loss_clustering = mse_loss(y_pred_unlab, augmented_y_pred_unlab)

                    clustering_loss = self.w2 * bce_loss + (1 - self.w2) * cs_loss_clustering

                    full_loss = classifier_loss + clustering_loss

                    # Backward
                    full_loss.backward()
                    optimizer.step()

                    # Save losses for plotting purposes
                    train_full_losses.append(full_loss.item())
                    train_classification_losses.append(classifier_loss.item())
                    train_clustering_losses.append(clustering_loss.item())
                    train_bce_losses.append(bce_loss.item())
                    train_ce_losses.append(ce_loss.item())
                    train_cs_classification_losses.append(cs_loss_classifier.item())
                    train_cs_clustering_losses.append(cs_loss_clustering.item())

                    # Update the memory modules
                    unlab_memory_module.memory_step(encoded_x_unlab.detach().clone(),
                                                    batch_x_train[mask_unlab].detach().clone())
                    lab_memory_module.memory_step(encoded_x[mask_lab].detach().clone(),
                                                  batch_x_train[mask_lab].detach().clone(),
                                                  input_labels=torch.tensor(batch_y_train[mask_lab], device=device))

                    n_current_training_step += 1
                    self.progress_percentage = (n_current_training_step / n_total_training_step) * 100

                    batch_start_index += self.batch_size
                    batch_end_index = min((batch_end_index + self.batch_size), self.x_full.shape[0])

                    # Losses
                losses_dict['full loss'].append(np.mean(train_full_losses))
                losses_dict['classification loss'].append(np.mean(train_classification_losses))
                losses_dict['clustering loss'].append(np.mean(train_clustering_losses))
                losses_dict['bce loss'].append(np.mean(train_bce_losses))
                losses_dict['ce loss'].append(np.mean(train_ce_losses))
                losses_dict['cs classification loss'].append(np.mean(train_cs_classification_losses))
                losses_dict['cs clustering loss'].append(np.mean(train_cs_clustering_losses))
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
