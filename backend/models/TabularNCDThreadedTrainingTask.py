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
    def __init__(self, dataset_name, target_name, known_classes, unknown_classes, selected_features, random_state, color_by, model_config, corresponding_tsne_config_name, TabularNCDModelToTrain,
                 use_unlab, use_ssl, M, lr_classif, lr_cluster, epochs, k_neighbors, w1, w2, p_m, alpha, batch_size, x_full, y_train_classifier, grouped_unknown_class_val, cat_columns_indexes):
        super().__init__(TabularNCDModelToTrain.app, dataset_name, target_name, known_classes, unknown_classes, selected_features, random_state, color_by, model_config, corresponding_tsne_config_name, TabularNCDModelToTrain.model_name)
        self.app = TabularNCDModelToTrain.app
        self.model_to_train = TabularNCDModelToTrain
        self.use_unlab = use_unlab  # Boolean
        self.use_ssl = use_ssl
        self.device = TabularNCDModelToTrain.device
        self.x_full = x_full
        self.y_train_classifier = y_train_classifier
        self.grouped_unknown_class_val = grouped_unknown_class_val
        self.cat_columns_indexes = cat_columns_indexes

        # Pre-training parameters
        self.M = M
        self.p_m = p_m
        self.alpha = alpha

        # Joint training parameters
        self.batch_size = batch_size
        self.lr_classif = lr_classif
        self.lr_cluster = lr_cluster
        self.epochs = epochs
        self.k_neighbors = k_neighbors
        self.w1 = w1
        self.w2 = w2

        # self.batch_size = TabularNCDModelToTrain.batch_size
        # self.num_epochs = TabularNCDModelToTrain.num_epochs
        self.device = TabularNCDModelToTrain.device

        # Event that will be set when .stop() is called on this thread
        self._stop_event = threading.Event()

    def run(self):
        losses_dict = {
            # Losses
            'train_classification_losses': [],
            'train_clustering_losses': [],
            'bce_losses': [],
            'ce_losses': [],
            'train_cs_classification_losses': [],
            'train_cs_clustering_losses': [],

            # Performance metrics
            'train_ari': [],
            # 'test_ari': [],
            'train_nmi': [],
            # 'test_nmi': [],
            'train_classification_accuracy': [],
            # 'test_classification_accuracy': [],
            'train_clustering_accuracy': [],
            # 'test_clustering_accuracy': [],
            'balanced_train_clustering_accuracy': [],
            # 'balanced_test_clustering_accuracy': [],

            'estimated_top_k_lists': []
        }

        self.app.logger.debug(self.model_to_train)

        try:
            # ==================== Self-Supervised Learning ====================
            if self.use_ssl is True:
                self.app.logger.info('Starting self-supervised learning...')
                vime_losses_dict = vime_training(self.x_full, self.model_to_train, self.device, p_m=self.p_m, alpha=self.alpha, lr=0.001, num_epochs=30, batch_size=128)
            # ==================================================================

            torch.cuda.empty_cache()

            # ========================== Joint learning ========================
            self.app.logger.info('Starting joint learning...')
            unlab_memory_module = NCLMemoryModule(self.device, M=self.M, labeled_memory=False)
            lab_memory_module = NCLMemoryModule(self.device, M=self.M, labeled_memory=True)

            optimizer_classification = torch.optim.AdamW(self.model_to_train.parameters(), lr=self.lr_classif)
            optimizer_clustering = torch.optim.AdamW(self.model_to_train.parameters(), lr=self.lr_cluster)

            n_batchs = math.ceil((self.x_full.shape[0]) / self.batch_size)
            n_current_training_step = 0
            n_total_training_step = n_batchs * self.epochs

            # with tqdm(range(n_batchs * config['epochs'])) as t:
            for epoch in range(self.epochs):
                self.app.logger.debug("Training progress at {:.1f}%...".format(self.progress_percentage))

                cross_entropy_loss = nn.CrossEntropyLoss()
                mse_loss = nn.MSELoss()

                train_classification_losses = []
                train_clustering_losses = []
                train_bce_losses = []
                train_ce_losses = []
                train_cs_classification_losses = []
                train_cs_clustering_losses = []

                estimated_topk_list = []

                # t.set_description(str(epoch + 1) + "/" + str(config['epochs']))

                batch_start_index, batch_end_index = 0, self.batch_size
                for batch_index in range(n_batchs):
                    # (1) ===== Get the data =====
                    batch_x_train = self.x_full[batch_start_index:batch_end_index]
                    batch_y_train = self.y_train_classifier[batch_start_index:batch_end_index]

                    mask_unlab = batch_y_train == self.grouped_unknown_class_val
                    mask_lab = ~mask_unlab
                    assert mask_unlab.sum() > 0, "No unlabeled data in batch"

                    # augment/transform the data
                    with torch.no_grad():  # <= /!\
                        augmented_x_unlab = smotenc_transform_batch_2(batch_x_train[mask_unlab], self.cat_columns_indexes, unlab_memory_module.original_data_memory, self.device, k_neighbors=self.k_neighbors)
                        augmented_x_lab = smotenc_transform_batch_2(batch_x_train[mask_lab], self.cat_columns_indexes, lab_memory_module.original_data_memory, self.device, k_neighbors=self.k_neighbors)

                    # (2) ===== Forward the classification data and compute the losses =====
                    encoded_x = self.model_to_train.encoder_forward(batch_x_train)
                    encoded_augmented_x_unlab = self.model_to_train.encoder_forward(augmented_x_unlab)
                    encoded_augmented_x_lab = self.model_to_train.encoder_forward(augmented_x_lab)
                    y_pred_lab = self.model_to_train.classification_head_forward(encoded_x)
                    augmented_y_pred = torch.zeros(y_pred_lab.shape, device=self.device)
                    augmented_y_pred[mask_lab] = self.model_to_train.classification_head_forward(encoded_augmented_x_lab)
                    augmented_y_pred[mask_unlab] = self.model_to_train.classification_head_forward(encoded_augmented_x_unlab)

                    ce_loss = cross_entropy_loss(y_pred_lab, torch.tensor(batch_y_train, device=self.device))

                    cs_loss_classifier = mse_loss(y_pred_lab, augmented_y_pred)

                    classifier_loss = self.w1 * ce_loss + (1 - self.w1) * cs_loss_classifier

                    # backward
                    optimizer_classification.zero_grad()
                    classifier_loss.backward()
                    optimizer_classification.step()

                    # (2.5) ===== Estimate the cosine_top_k using the labeled data only =====
                    with torch.no_grad():
                        if self.use_unlab is False or self.use_unlab:
                            # Compute the cosine similarity matrix of the labeled data:
                            encoded_x_lab = encoded_x[mask_lab]
                            labeled_similarities = F.cosine_similarity(encoded_x_lab.unsqueeze(1), encoded_x_lab, dim=-1)
                            labeled_similarities -= torch.eye(len(labeled_similarities), device=self.device)

                            # Compute the ground truth pairwise pseudo label matrix of the labeled data
                            ground_truth_matrix = np.equal.outer(batch_y_train[mask_lab], batch_y_train[mask_lab])
                            ground_truth_matrix = torch.tensor(ground_truth_matrix, device=self.device, dtype=torch.long, requires_grad=False)

                            # Estimate the lower bound of the error curve with the golden section search:
                            estimated_topk = custom_golden_section_search(labeled_similarities, ground_truth_matrix, pseudo_labeling_method="top_k_cosine_per_instance", device=self.device, a=1, b=100, iterations=10)
                        else:
                            lab_unlab_similarities = F.cosine_similarity(encoded_x.unsqueeze(1), encoded_x, dim=-1)
                            lab_unlab_similarities -= torch.eye(len(lab_unlab_similarities), device=self.device)

                            full_ground_truth_matrix = np.equal.outer(batch_y_train, batch_y_train)
                            full_ground_truth_matrix = torch.tensor(full_ground_truth_matrix, device=self.device, dtype=torch.long, requires_grad=False)

                            unlab_indexes = np.arange(len(mask_unlab))[mask_unlab]
                            estimated_topk = custom_golden_section_search(lab_unlab_similarities, full_ground_truth_matrix, pseudo_labeling_method="top_k_cosine_per_instance", device=self.device, a=1, b=100, unlab_indexes=unlab_indexes, iterations=10)
                        torch.cuda.empty_cache()
                    estimated_topk_list.append(estimated_topk)
                    top_k = estimated_topk
                    # (2.5) =====================================

                    # (3) ===== Forward the clustering data and compute the losses =====
                    encoded_x = self.model_to_train.encoder_forward(batch_x_train)
                    encoded_x_unlab = encoded_x[mask_unlab]
                    y_pred_unlab = self.model_to_train.clustering_head_forward(encoded_x_unlab)

                    encoded_augmented_x_unlab = self.model_to_train.encoder_forward(augmented_x_unlab)
                    augmented_y_pred_unlab = self.model_to_train.clustering_head_forward(encoded_augmented_x_unlab)

                    # ========== Define the pseudo labels ==========
                    computed_top_k = int((top_k / 100) * len(encoded_x_unlab))

                    # Because it is symmetric, we compute the upper corner and copy it to the lower corner
                    upper_list_1, upper_list_2 = np.triu_indices(len(encoded_x_unlab), k=1)
                    unlab_unlab_similarities = nn.CosineSimilarity()(encoded_x_unlab[upper_list_1], encoded_x_unlab[upper_list_2])
                    similarity_matrix = torch.zeros((len(encoded_x_unlab), len(encoded_x_unlab)), device=self.device)
                    similarity_matrix[upper_list_1, upper_list_2] = unlab_unlab_similarities
                    similarity_matrix += similarity_matrix.T.clone()

                    top_k_most_similar_instances_per_instance = similarity_matrix.argsort(descending=True)[:, :computed_top_k]

                    pseudo_labels_matrix = torch.zeros((len(encoded_x_unlab), len(encoded_x_unlab)), device=self.device)
                    pseudo_labels_matrix = pseudo_labels_matrix.scatter_(index=top_k_most_similar_instances_per_instance, dim=1, value=1)

                    # The matrix isn't symmetric, because the graph is directed
                    # So if there is one link between two points, regardless of the direction, we consider this pair to be positive
                    pseudo_labels_matrix += pseudo_labels_matrix.T.clone()
                    pseudo_labels_matrix[pseudo_labels_matrix > 1] = 1  # Some links will overlap
                    # ==============================================

                    pseudo_labels = pseudo_labels_matrix[upper_list_1, upper_list_2]
                    bce_loss = unsupervised_classification_loss(y_pred_unlab[upper_list_1], y_pred_unlab[upper_list_2], pseudo_labels)

                    cs_loss_clustering = mse_loss(y_pred_unlab, augmented_y_pred_unlab)

                    clustering_loss = self.w2 * bce_loss + (1 - self.w2) * cs_loss_clustering

                    # backward
                    optimizer_clustering.zero_grad()
                    clustering_loss.backward()
                    optimizer_clustering.step()

                    # Save losses for plotting purposes
                    train_classification_losses.append(classifier_loss.item())
                    train_clustering_losses.append(clustering_loss.item())
                    train_bce_losses.append(bce_loss.item())
                    train_ce_losses.append(ce_loss.item())
                    train_cs_classification_losses.append(cs_loss_classifier.item())
                    train_cs_clustering_losses.append(cs_loss_clustering.item())

                    # t.set_postfix_str("classif={:05.3f}".format(pretty_mean(train_classification_losses))
                    #                   + " clust={:05.3f}".format(pretty_mean(train_clustering_losses))
                    #                   + " ce={:05.3f}".format(pretty_mean(train_ce_losses))
                    #                   + " bce={:05.3f}".format(pretty_mean(train_bce_losses))
                    #                   + " cs1={:05.3f}".format(pretty_mean(train_cs_classification_losses))
                    #                   + " cs2={:05.3f}".format(pretty_mean(train_cs_clustering_losses)))
                    # t.update()

                    # update the memory modules
                    unlab_memory_module.memory_step(encoded_x_unlab.detach().clone(),
                                                    batch_x_train[mask_unlab].detach().clone())
                    lab_memory_module.memory_step(encoded_x[mask_lab].detach().clone(),
                                                  batch_x_train[mask_lab].detach().clone(),
                                                  input_labels=torch.tensor(batch_y_train[mask_lab], device=self.device))

                    n_current_training_step += 1
                    self.progress_percentage = (n_current_training_step / n_total_training_step) * 100

                    batch_start_index += self.batch_size
                    batch_end_index = min((batch_end_index + self.batch_size), self.x_full.shape[0])

                    if self.stopped() is True:
                        raise KilledException

                # train_classification_accuracy = compute_classification_accuracy(self.x_full, y_train_classifier,
                #                                                                 y_train_classifier, self.model_to_train)
                # test_classification_accuracy = compute_classification_accuracy(x_test, y_test_classifier,
                #                                                                y_train_classifier, self.model_to_train)

                # train_clustering_accuracy = compute_clustering_accuracy(x_unlab, y_unlab, y_unlab, self.model_to_train)
                # test_clustering_accuracy = compute_clustering_accuracy(x_test, y_test, y_unlab, self.model_to_train)

                # balanced_train_clustering_accuracy = compute_balanced_clustering_accuracy(x_unlab, y_unlab, y_unlab,
                #                                                                           self.model_to_train)
                # balanced_test_clustering_accuracy = compute_balanced_clustering_accuracy(x_test, y_test, y_unlab, self.model_to_train)

                # train_ari, train_nmi = compute_ari_and_nmi(x_unlab, y_unlab, y_unlab, self.model_to_train)
                # test_ari, test_nmi = compute_ari_and_nmi(x_test, y_test, y_unlab, self.model_to_train)

                # losses_dict['train_ari'].append(train_ari)
                # losses_dict['test_ari'].append(test_ari)
                # losses_dict['train_nmi'].append(train_nmi)
                # losses_dict['test_nmi'].append(test_nmi)
                # losses_dict['train_classification_losses'].append(np.mean(train_classification_losses))
                # losses_dict['train_clustering_losses'].append(np.mean(train_clustering_losses))
                # losses_dict['bce_losses'].append(np.mean(train_bce_losses))
                # losses_dict['ce_losses'].append(np.mean(train_ce_losses))
                # losses_dict['train_cs_classification_losses'].append(np.mean(train_cs_classification_losses))
                # losses_dict['train_cs_clustering_losses'].append(np.mean(train_cs_clustering_losses))
                # losses_dict['train_classification_accuracy'].append(train_classification_accuracy)
                # losses_dict['test_classification_accuracy'].append(test_classification_accuracy)
                # losses_dict['test_clustering_accuracy'].append(test_clustering_accuracy)
                # losses_dict['train_clustering_accuracy'].append(train_clustering_accuracy)
                # losses_dict['balanced_test_clustering_accuracy'].append(balanced_test_clustering_accuracy)
                # losses_dict['balanced_train_clustering_accuracy'].append(balanced_train_clustering_accuracy)
                losses_dict['estimated_top_k_lists'].append(np.mean(estimated_topk_list))

                # self.app.logger.debug("Train / Test clustering accuracy = {:05.3f} / {:05.3f}".format(train_clustering_accuracy, test_clustering_accuracy))
                # self.app.logger.debug("Train / Test balanced clustering accuracy = {:05.3f} / {:05.3f}".format(balanced_train_clustering_accuracy, balanced_test_clustering_accuracy))

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
