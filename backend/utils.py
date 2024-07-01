"""
Orange Labs
Authors : Colin Troisemaine
Maintainer : colin.troisemaine@gmail.com
"""

from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score, balanced_accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import torch.nn.functional as F
from torch import nn
import numpy as np
import torch
import math
import os


def setup_device(app, use_cuda=True):
    """
    Initialize the torch device where the code will be executed on.

    :param app: the app, to format logging
    :param use_cuda: Set to True if you want the code to be run on your GPU. If set to False, code will run on CPU.
    :return: torch.device : The initialized device, torch.device.
    """
    if use_cuda is False or not torch.cuda.is_available():
        device_name = "cpu"
        if use_cuda is True:
            app.logger.critical("unable to initialize CUDA, check torch installation (https://pytorch.org/)")
        if use_cuda is False:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device_name = "cuda:0"
        app.logger.info("CUDA successfully initialized on device : " + torch.cuda.get_device_name())

    device = torch.device(device_name)

    app.logger.info("Using device : " + device.type)

    return device


def get_activation_function(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        return None


def get_simple_layer(size_in, size_out, add_dropout=True, p_dropout=0.3, activation_fct='sigmoid'):
    """
    General function to define a layer with a single dense layer, followed *optionally* by a dropout and activation layer.
    :param size_in: The input size of the dense layer.
    :param size_out:  The output size of the dense layer.
    :param add_dropout: Add a dropout layer of not.
    :param p_dropout: The probability of the dropout layer.
    :param activation_fct: The activation function. Choices : ['relu', 'sigmoid', 'tanh', None].
    :return: List : The layers.
    """
    simple_layer = [nn.Linear(size_in, size_out)]

    if add_dropout is True:
        simple_layer.append(nn.Dropout(p=p_dropout))

    if activation_fct == 'relu':
        simple_layer.append(nn.ReLU())
    elif activation_fct == 'sigmoid':
        simple_layer.append(nn.Sigmoid())
    elif activation_fct == 'tanh':
        simple_layer.append(nn.Tanh())

    return simple_layer


def pretext_generator(m, x):
    """
    Generation of corrupted samples.
    This is a sped up version of the original pretext_generator of VIME's code.
    It is about 5 times faster and should be equivalent.

    :param m: The corruption mask, np.array with shape (n_samples, n_features).
    :param x: The set to corrupt, np.array with shape (n_samples, n_features).
    :return:
        m_new: The new corruption mask, np.array with shape (n_samples, n_features).
        x_tilde: The corrupted samples, np.array with shape (n_samples, n_features).
    """
    # Randomly (and column-wise) shuffle data
    x_bar = x.copy()
    np.random.shuffle(x_bar)

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m

    # Define new mask matrix (as it is possible that the corrupted samples are the same as the original ones)
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def pretext_generator_fixed(m, x):
    """
    Generation of corrupted samples.
    This is a sped up version of the original pretext_generator of VIME's code.
    It is about 2 times faster and it is equivalent.

    :param m: The corruption mask, np.array with shape (n_samples, n_features).
    :param x: The set to corrupt, np.array with shape (n_samples, n_features).
    :return:
        m_new: The new corruption mask, np.array with shape (n_samples, n_features).
        x_tilde: The corrupted samples, np.array with shape (n_samples, n_features).
    """
    # Randomly (and column-wise) shuffle data
    no, dim = x.shape
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        tmp = x[:, i].copy()
        np.random.shuffle(tmp)
        x_bar[:, i] = tmp

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m

    # Define new mask matrix (as it is possible that the corrupted samples are the same as the original ones)
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def vime_loss(mask_pred, mask_true, feature_pred, batch_x_train):
    """
    Note that all the inputs should have values between 0 and 1.

    :param mask_pred: The predicted corruption mask, torch.Tensor of shape (n_samples, n_features).
    :param mask_true: The true corruption mask, torch.Tensor of shape (n_samples, n_features).
    :param feature_pred: The reconstructed values of x, torch.Tensor of shape (n_samples, n_features).
    :param batch_x_train: The original (uncorrupted) values of x, torch.Tensor of shape (n_samples, n_features).
    :return:
        mask_loss: The mean binary cross-entropy loss of the mask estimation.
        feature_loss: The mean binary cross-entropy loss of the feature estimation.
    """
    mask_loss = nn.BCELoss()(mask_pred, mask_true)
    feature_loss = nn.MSELoss()(feature_pred, batch_x_train)
    return mask_loss, feature_loss


def evaluate_vime_model_on_set(x_input, model, device, batch_size=100, p_m=0.3):
    """
    Method that evaluates the feature and mask estimation of corrupted samples of the given model on x_input.

    :param x_input: The input dataset, torch.Tensor of shape (n_samples, n_features).
    :param model: The model to evaluate, torch.nn.Module.
    :param device: The device to send the dataset to, torch.device.
    :param p_m: The corruption probability, int.
    :param batch_size: The batch size (default=100, reduce if GPU is low on memory), int.
    :return:
        mean_mask_loss: The mean mask estimation loss, float.
        mean_feature_loss: The mean feature estimation loss, float.
    """
    mask_losses, feature_losses = [], []
    test_batch_start_index, test_batch_end_index = 0, batch_size
    for batch_index in range(math.ceil((x_input.shape[0]) / batch_size)):
        batch_x_input = x_input[test_batch_start_index:test_batch_end_index]

        m_unlab = np.random.binomial(1, p_m, batch_x_input.shape)
        m_label, x_tilde = pretext_generator(m_unlab, batch_x_input.to('cpu').numpy())
        x_tilde = torch.Tensor(x_tilde).to(device)
        m_label = torch.Tensor(m_label).to(device)

        model.eval()
        with torch.no_grad():
            mask_pred, feature_pred = model.vime_forward(x_tilde)
        model.train()

        mask_loss, feature_loss = vime_loss(mask_pred, m_label, feature_pred, batch_x_input)
        mask_losses.append(mask_loss.item())
        feature_losses.append(feature_loss.item())

        test_batch_start_index += batch_size
        test_batch_end_index = test_batch_end_index + batch_size if test_batch_end_index + batch_size < x_input.shape[0] else x_input.shape[0]

    return np.mean(mask_losses), np.mean(feature_losses)


def vime_training(x_vime, model, device, p_m=0.3, alpha=2.0, lr=0.001, num_epochs=30, batch_size=128, fixed_corruption=False):
    # x_test, compute_lr_accuracy=False, x_train_known=None, y_train_known=None, x_test_known=None, y_test_known=None,
    """
    :param p_m: Loss_tot = Corruption probability
    :param alpha: Loss_tot = mask_estim_loss + alpha * feature_estim_loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses_dict = {
        'epoch_mean_train_losses': [],
        'epoch_mean_test_losses': [],
        'epoch_mean_lr_projection_score': [],
        'lr_base_projection_score': None
    }

    n_batchs = math.ceil((x_vime.shape[0]) / batch_size)

    for epoch in range(num_epochs):
        # t.set_description("Epoch " + str(epoch + 1) + " / " + str(num_epochs))
        train_losses = []
        mask_losses = []
        feature_losses = []

        batch_start_index, batch_end_index = 0, min(batch_size, len(x_vime))
        for batch_index in range(n_batchs):
            batch_X_train = x_vime[batch_start_index:batch_end_index]

            m_unlab = np.random.binomial(1, p_m, batch_X_train.shape)
            if fixed_corruption is False:
                m_label, x_tilde = pretext_generator(m_unlab, batch_X_train.to('cpu').numpy())
            else:
                m_label, x_tilde = pretext_generator_fixed(m_unlab, batch_X_train.to('cpu').numpy())

            x_tilde = torch.Tensor(x_tilde).to(device)
            m_label = torch.Tensor(m_label).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            mask_pred, feature_pred = model.vime_forward(x_tilde)

            # compute losses
            mask_loss, feature_loss = vime_loss(mask_pred, m_label, feature_pred, batch_X_train)
            loss = mask_loss + alpha * feature_loss

            # backward
            loss.backward()

            # update the weights using gradient descent
            optimizer.step()

            # Save loss for plotting purposes
            train_losses.append(loss.item())
            mask_losses.append(mask_loss.item())
            feature_losses.append(feature_loss.item())

            # print statistics
            # t.set_postfix_str("loss={:05.3f}".format(np.mean(train_losses)) +
            #                   " - mask_loss={:05.3f}".format(np.mean(mask_losses)) +
            #                   " - feature_loss={:05.3f}".format(np.mean(feature_losses)))
            # t.update()
            batch_start_index += batch_size
            batch_end_index = min((batch_end_index + batch_size), x_vime.shape[0])

        # Evaluate on the test set
        # test_mask_loss, test_feature_loss = evaluate_vime_model_on_set(x_test, model, device)
        # test_loss = test_mask_loss + alpha * test_feature_loss

        losses_dict['epoch_mean_train_losses'].append(np.mean(train_losses))
        # losses_dict['epoch_mean_test_losses'].append(test_loss.item())

    return losses_dict


def fine_tuning_training(x_sup, y_sup, x_test, y_test, model, device, batch_size=128, num_epochs=10, lr=0.001):
    """
    The fine tuning (step 2) of the representation on the known classes.
    :param x_sup: The training instances from known classes (and the meta class composed of the unknown classes).
    :param y_sup: The labels of x_sup.
    :param x_test: The test data (known classes + the meta class composed of the unknown classes).
    :param y_test: The labels of x_test.
    :param model: torch.nn.Module : The model to train.
    :param device: torch.device : The device.
    :param batch_size: int : The batch size.
    :param num_epochs: int : The number of training steps.
    :param lr: float : The learning rate.
    :return: todo
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses_dict = {
        'epoch_mean_train_losses': [],
        'epoch_mean_test_losses': [],
        'epoch_mean_train_acc': [],
        'epoch_mean_test_acc': []
    }

    for epoch in range(num_epochs):
        n_batchs = math.ceil((x_sup.shape[0]) / batch_size)
        # t.set_description("Epoch " + str(epoch + 1) + " / " + str(num_epochs))
        train_losses = []

        batch_start_index, batch_end_index = 0, min(batch_size, len(x_sup))
        for batch_index in range(n_batchs):
            batch_x_train = x_sup[batch_start_index:batch_end_index]
            batch_y_train = y_sup[batch_start_index:batch_end_index]
            batch_y_train = torch.tensor(batch_y_train, dtype=torch.int64, device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            encoded_batch_x = model.encoder_forward(batch_x_train)
            y_pred = model.classification_head_forward(encoded_batch_x)

            # compute loss
            supervised_loss = nn.CrossEntropyLoss()(y_pred, batch_y_train)

            # backward
            supervised_loss.backward()

            # update the weights using gradient descent
            optimizer.step()

            # Save loss for plotting purposes
            train_losses.append(supervised_loss.item())

            # print statistics
            # t.set_postfix_str("loss={:05.3f}".format(np.mean(train_losses)))
            # t.update()
            batch_start_index += batch_size
            batch_end_index = min((batch_end_index + batch_size), x_sup.shape[0])

        # Evaluate on the test set
        tmp_y_test = torch.tensor(y_test, dtype=torch.int64, device=device)
        test_loss = evaluate_supervised_model_on_set(x_test, tmp_y_test, model)

        train_classification_accuracy = compute_classification_accuracy(x_sup, y_sup, y_sup, model)
        losses_dict['epoch_mean_train_acc'].append(train_classification_accuracy)
        test_classification_accuracy = compute_classification_accuracy(x_test, tmp_y_test.cpu().numpy(), y_sup, model)
        losses_dict['epoch_mean_test_acc'].append(test_classification_accuracy)

        losses_dict['epoch_mean_train_losses'].append(np.mean(train_losses))
        losses_dict['epoch_mean_test_losses'].append(test_loss.item())

    return losses_dict


def compute_classification_accuracy(x_test, y_test, y_train, model):
    """
    ToDo : Documentation
    """
    # Define a mapping of the train classes, as they may not range from 0 to C
    mapper, ind = np.unique(y_train, return_inverse=True)

    x_test_known_mask = np.in1d(y_test, np.unique(y_train))
    x_test_known = x_test[x_test_known_mask]
    y_test_known = y_test[x_test_known_mask]

    model.eval()
    with torch.no_grad():
        # Forward the classification head only for known classes
        x_test_known_projection = model.encoder_forward(x_test_known)
        model_y_test_known_pred = model.classification_head_forward(x_test_known_projection)
    model.train()

    model_y_test_known_pred = F.softmax(model_y_test_known_pred, -1)  # Apply softmax
    model_y_test_known_pred = torch.argmax(model_y_test_known_pred, dim=1)  # Get the prediction from the probabilities
    model_y_test_known_pred = mapper[model_y_test_known_pred.cpu().numpy()]  # Map the prediction back to the true labels

    return accuracy_score(y_test_known, model_y_test_known_pred)


def evaluate_supervised_model_on_set(x_input, y_input, model, batch_size=100):
    """
    Method that evaluates the classification loss of the given model on x_input.

    :param x_input: The input dataset, np.array with shape (n_samples, n_features).
    :param y_input: The labels of the dataset, np.array with shape (n_samples,).
    :param model: The model to evaluate, torch.nn.Module.
    :param batch_size: The batch size (default=100, reduce if GPU is low on memory), int.
    :return:
        The mean classification loss, float.
    """
    losses = []
    test_batch_start_index, test_batch_end_index = 0, batch_size
    for batch_index in range(math.ceil((x_input.shape[0]) / batch_size)):
        batch_x = x_input[test_batch_start_index:test_batch_end_index]
        batch_y = y_input[test_batch_start_index:test_batch_end_index]

        model.eval()
        with torch.no_grad():
            encoded_batch_x = model.encoder_forward(batch_x)
            batch_y_pred = model.classification_head_forward(encoded_batch_x)
        model.train()

        supervised_loss = nn.CrossEntropyLoss()(batch_y_pred, batch_y)
        losses.append(supervised_loss.item())

        test_batch_start_index += batch_size
        test_batch_end_index = test_batch_end_index + batch_size if test_batch_end_index + batch_size < x_input.shape[0] else x_input.shape[0]

    return np.mean(losses)


def compute_clustering_accuracy(x_test, y_test, y_unlab, model):
    """
    Compute the clustering accuracy.
    The computation is based on the assignment of the most probable clusters using scipy's linear_sum_assignment.

    :param x_test: ToDo : Documentation
    :param y_test: ToDo : Documentation
    :param y_unlab: ToDo : Documentation
    :param model: ToDo : Documentation
    :return: Accuracy between 0 and 1.
    """
    # (1) Get the prediction of the model
    x_test_unknown_mask = np.in1d(y_test, np.unique(y_unlab))
    x_test_unknown = x_test[x_test_unknown_mask]
    y_test_unknown = y_test[x_test_unknown_mask]

    model.eval()
    with torch.no_grad():
        x_test_unknown_projection = model.encoder_forward(x_test_unknown)
        model_y_test_unknown_pred = model.clustering_head_forward(x_test_unknown_projection)
    model.train()

    model_y_test_unknown_pred = F.softmax(model_y_test_unknown_pred, -1)
    model_y_test_unknown_pred = torch.argmax(model_y_test_unknown_pred, dim=1)
    model_y_test_unknown_pred = model_y_test_unknown_pred.cpu().numpy()

    # (2) Compute the clustering accuracy using the hungarian algorithm
    y_test_unknown = y_test_unknown.astype(np.int64)
    assert model_y_test_unknown_pred.size == y_test_unknown.size
    D = max(model_y_test_unknown_pred.max(), y_test_unknown.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(model_y_test_unknown_pred.size):
        w[model_y_test_unknown_pred[i], y_test_unknown[i]] += 1
    ind = linear_assignment(w.max() - w)  # The hungarian algorithm

    acc = sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / model_y_test_unknown_pred.size

    return acc


def compute_balanced_clustering_accuracy(x_test, y_test, y_unlab, model):
    """
    Compute the clustering accuracy.
    The computation is based on the assignment of the most probable clusters using scipy's linear_sum_assignment.

    :param x_test: ToDo : Documentation
    :param y_test: ToDo : Documentation
    :param y_unlab: ToDo : Documentation
    :param model: ToDo : Documentation
    :return: Accuracy between 0 and 1.
    """
    # (1) Get the prediction of the model
    x_test_unknown_mask = np.in1d(y_test, np.unique(y_unlab))
    x_test_unknown = x_test[x_test_unknown_mask]
    y_test_unknown = y_test[x_test_unknown_mask]

    model.eval()
    with torch.no_grad():
        x_test_unknown_projection = model.encoder_forward(x_test_unknown)
        model_y_test_unknown_pred = model.clustering_head_forward(x_test_unknown_projection)
    model.train()

    model_y_test_unknown_pred = F.softmax(model_y_test_unknown_pred, -1)
    model_y_test_unknown_pred = torch.argmax(model_y_test_unknown_pred, dim=1)
    model_y_test_unknown_pred = model_y_test_unknown_pred.cpu().numpy()

    # (2) Compute the clustering accuracy using the hungarian algorithm
    y_test_unknown = y_test_unknown.astype(np.int64)
    assert model_y_test_unknown_pred.size == y_test_unknown.size
    D = max(model_y_test_unknown_pred.max(), y_test_unknown.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(model_y_test_unknown_pred.size):
        w[model_y_test_unknown_pred[i], y_test_unknown[i]] += 1
    ind = linear_assignment(w.max() - w)  # The hungarian algorithm

    # Balanced accuracy
    permutations_dict = dict(zip(ind[0], ind[1]))
    return balanced_accuracy_score(y_test_unknown, list(map(permutations_dict.get, model_y_test_unknown_pred)))


def compute_ari_and_nmi(x_test, y_test, y_unlab, model):
    """
    ToDo Documentation
    """
    x_test_known_mask = np.in1d(y_test, np.unique(y_unlab))
    x_test_known = x_test[x_test_known_mask]
    y_test_known = y_test[x_test_known_mask]

    model.eval()
    with torch.no_grad():
        model_y_test_known_pred = model.clustering_head_forward(model.encoder_forward(x_test_known))
    model.train()
    model_y_test_known_pred = F.softmax(model_y_test_known_pred, -1)
    model_y_test_known_pred = torch.argmax(model_y_test_known_pred, dim=1)

    # Both metrics are independent of the absolute values of the labels
    # So a permutation of the class or cluster label values won’t change the score value in any way.
    ari = adjusted_rand_score(y_test_known, model_y_test_known_pred.cpu().numpy())
    nmi = normalized_mutual_info_score(y_test_known, model_y_test_known_pred.cpu().numpy())
    return ari, nmi


def unsupervised_classification_loss(y_pred_1, y_pred_2, labels, eps=1e-7):
    """
    The intuition is that for each pair of samples, if their label is 1, we want the predicted
    probability of the same class to be high, and low if their label is 0.

    :param y_pred_1: The raw class predictions (before softmax) of the first list of samples,
                     torch.tensor of shape (n_samples, n_classes).
    :param y_pred_2: The raw class predictions (before softmax) of the second list of samples,
                     torch.tensor of shape (n_samples, n_classes).
    :param labels: The true labels of the samples (generated by cosine similarity),
                   the value are 0 when the class is different and 1 when it is the same.
                   torch.tensor of shape (n_samples,).
    :param eps: This float is used to 'clip' y_pred_proba's values that are 0, as computing log(0) is impossible.
    :return:
        Mean loss, float.
    """
    prob_1, prob_2 = F.softmax(y_pred_1, -1), F.softmax(y_pred_2, -1)  # Simple softmax of the output
    x = prob_1.mul(prob_2)  # We multiply the prediction of each vector between each other (so same shape is outputted)
    x = x.sum(1)  # We sum the results of each row. If the predictions of the same class were high, the result is close to 1
    return - torch.mean(labels.mul(x.add(eps).log()) + (1 - labels).mul((1 - x).add(eps).log()))  # BCE


def ranking_stats_pseudo_labels(encoded_x_unlab, device, topk=5):
    rank_idx = torch.argsort(encoded_x_unlab, dim=1, descending=True)
    rank_idx1, rank_idx2 = PairEnum(rank_idx)
    rank_idx1, rank_idx2 = rank_idx1[:, :topk], rank_idx2[:, :topk]
    rank_idx1, _ = torch.sort(rank_idx1, dim=1)
    rank_idx2, _ = torch.sort(rank_idx2, dim=1)
    rank_diff = rank_idx1 - rank_idx2
    rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
    target_ulb = torch.ones_like(rank_diff).float().to(device)
    target_ulb[rank_diff > 0] = 0
    return target_ulb


def PairEnum(x):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return x1, x2


def get_error_count_for_topk(pairwise_similarity_matrix, ground_truth_matrix, pseudo_labeling_method, top_k, device, unlab_indexes=None):
    n = pairwise_similarity_matrix.shape[0]

    if pseudo_labeling_method == 'top_k_cosine_per_instance_agreeing':
        top_k = int((top_k / 100) * n)

        top_k_most_similar_instances_per_instance = pairwise_similarity_matrix.argsort(descending=True)[:, :top_k]

        pseudo_labels_matrix = torch.zeros(n, n, device=device, dtype=torch.long, requires_grad=False)
        pseudo_labels_matrix = pseudo_labels_matrix.scatter_(index=top_k_most_similar_instances_per_instance, dim=1, value=1)

        pseudo_labels_matrix += pseudo_labels_matrix.T.clone()  # The matrix isnt symmetric, because the graph is directed
        pseudo_labels_matrix[pseudo_labels_matrix == 1] = 0  # Some will overlap
        pseudo_labels_matrix[pseudo_labels_matrix == 2] = 1  # Some will overlap
    elif pseudo_labeling_method == 'top_k_cosine_per_instance' or pseudo_labeling_method == 'top_k_cosine_per_instance_new':
        top_k = int((top_k / 100) * n)

        top_k_most_similar_instances_per_instance = pairwise_similarity_matrix.argsort(descending=True)[:, :top_k]

        pseudo_labels_matrix = torch.zeros(n, n, device=device, dtype=torch.long, requires_grad=False)
        pseudo_labels_matrix = pseudo_labels_matrix.scatter_(index=top_k_most_similar_instances_per_instance, dim=1, value=1)

        pseudo_labels_matrix += pseudo_labels_matrix.T.clone()  # The matrix isnt symmetric, because the graph is directed
        pseudo_labels_matrix[pseudo_labels_matrix > 1] = 1  # Some will overlap
    elif pseudo_labeling_method == 'top_k_cosine_faster':
        top_k = int( ((n * (n - 1)) / 2) * (top_k / 100))

        tmp_pairwise_similarity_matrix = pairwise_similarity_matrix.triu(diagonal=1).clone().detach()

        _, i = torch.topk(tmp_pairwise_similarity_matrix.flatten(), top_k)

        most_similar_indices = np.array(np.unravel_index(i.cpu().numpy(), tmp_pairwise_similarity_matrix.shape))

        pseudo_labels_matrix = torch.zeros(n, n, device=device, dtype=torch.long, requires_grad=False)

        pseudo_labels_matrix[most_similar_indices] = 1
    elif pseudo_labeling_method == 'cosine':
        top_k = top_k / 100
        tmp_pairwise_similarity_matrix = pairwise_similarity_matrix.detach().clone()
        tmp_pairwise_similarity_matrix[np.tril_indices(n, k=0)] = -math.inf

        pseudo_labels_matrix = (tmp_pairwise_similarity_matrix > top_k).to(torch.int32)
    else:
        raise ValueError('Undefined value for parameter pseudo_labeling_method.')

    diff = (ground_truth_matrix - pseudo_labels_matrix)

    if unlab_indexes is not None:
        # Ignore the errors in the unlab-unlab pairs:
        unlab_unlab_indexes = np.array(np.meshgrid(unlab_indexes, unlab_indexes)).T.reshape(-1, 2)
        abs_triu = torch.triu(diff, diagonal=1).abs()
        return (abs_triu.sum() - abs_triu[unlab_unlab_indexes.T].sum()).item()
    else:
        return torch.triu(diff, diagonal=1).abs().sum().item()


def custom_golden_section_search(pairwise_similarity_matrix, ground_truth_matrix, pseudo_labeling_method, device,
                                 a=0, b=100, unlab_indexes=None, iterations=10, tol=0.1):
    """
    Estimate the top_k value for which the pairwise pseudo-label
    definition method has the lowest error in limited iterations.

    Taken from https://en.wikipedia.org/wiki/Golden-section_search
    """
    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return a, b

    # Required steps to achieve tolerance
    # n = int(math.ceil(math.log(tol / h) / math.log(invphi)))
    # print(f"{n} steps required")

    c = a + invphi2 * h
    d = a + invphi * h
    yc = get_error_count_for_topk(pairwise_similarity_matrix, ground_truth_matrix, pseudo_labeling_method, c, device, unlab_indexes)
    yd = get_error_count_for_topk(pairwise_similarity_matrix, ground_truth_matrix, pseudo_labeling_method, d, device, unlab_indexes)

    for k in range(iterations - 1):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = get_error_count_for_topk(pairwise_similarity_matrix, ground_truth_matrix, pseudo_labeling_method, c, device, unlab_indexes)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = get_error_count_for_topk(pairwise_similarity_matrix, ground_truth_matrix, pseudo_labeling_method, d, device, unlab_indexes)

    if yc < yd:
        return (a + d) / 2  # interval = (a, d)
    else:
        return (c + b) / 2  # (c, b)


def smotenc_transform_batch_2(batch, cat_columns_indexes, data_queue, device, k_neighbors=5, dist='cosine',
                              batch_size=100):
    """
    Faster but harder to understand vectorized transformation for mixed numerical and categorical data.
    See 'smotenc_transform_batch' to really understand the logic.
    Inspired from SMOTE-NC.
    :param batch: torch.Tensor : The batch data to transform.
    :param cat_columns_indexes: Array-like object of the indexes of the categorical columns.
    Only useful when transform_method='new_2'.
    :param data_queue: The unlabelled data stored in the unlab_memory_module object.
    :param device: torch.device : The device.
    :param k_neighbors: int : The number of neighbors to consider during the transformation.
    :param dist: The distance metric to use. Choices : ['cosine', 'euclidean'].
    :param batch_size: int : During computation, the batch is cut in blocs of size batch_size. If you have memory errors, reduce it.
    :return: torch.Tensor : The transformed data.
    """
    full_data = torch.cat([batch, data_queue])

    full_similarities_matrix = torch.tensor([], device=device, dtype=torch.float32)

    n_batchs = math.ceil((full_data.shape[0]) / batch_size)
    batch_start_index, batch_end_index = 0, min(batch_size, len(full_data))
    for batch_index in range(n_batchs):
        if dist == 'cosine':
            similarities = F.cosine_similarity(batch.unsqueeze(1), full_data[batch_start_index:batch_end_index], dim=-1)
        elif dist == 'euclidean':
            # ToDo (below is the non-vectorized code)
            # similarities = torch.cdist(batch[i].view(1, -1), full_data)
            # similarities[i] += torch.inf  # This way, itself wont be in the most similar instances
            # topk_similar_indexes = similarities.topk(k=k_neighbors, largest=False).indices
            pass

        full_similarities_matrix = torch.cat([full_similarities_matrix, similarities], dim=1)

        batch_start_index += batch_size
        batch_end_index = min((batch_end_index + batch_size), full_data.shape[0])

    full_similarities_matrix -= torch.eye(len(batch), len(full_data), device=device)  # This way, itself wont be in the most similar instances

    batch_topk_similar_indexes = full_similarities_matrix.topk(k=k_neighbors, dim=1).indices

    # Select a random point between the k closest
    batch_closest_point_index = torch.gather(batch_topk_similar_indexes, 1, torch.randint(low=0, high=k_neighbors, size=(len(batch),), device=device).view(-1, 1))
    batch_closest_point_index = batch_closest_point_index.flatten()

    batch_closest_point = full_data[batch_closest_point_index]

    batch_diff_vect = (batch_closest_point - batch) * torch.rand(len(batch), device=device).view(-1, 1)

    augmented_batch = batch + batch_diff_vect  # At this point, the categorical values are wrong, next line fixes that

    if len(cat_columns_indexes) > 0:
        augmented_batch[:, cat_columns_indexes] = full_data[:, cat_columns_indexes.flatten()][batch_topk_similar_indexes].mode(1).values

    return augmented_batch
