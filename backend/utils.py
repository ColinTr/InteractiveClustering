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


def pairwise_cosine_similarity(input_a, input_b):
    normalized_input_a = torch.nn.functional.normalize(input_a)
    normalized_input_b = torch.nn.functional.normalize(input_b)
    res = torch.mm(normalized_input_a, normalized_input_b.T)
    return res


def smotenc_transform_batch_2(batch, cat_columns_indices, data_queue, device, k_neighbors=5, dist='cosine',
                              batch_size=100):
    """
    Faster but harder to understand vectorized transformation for mixed numerical and categorical data.
    See 'smotenc_transform_batch' to really understand the logic.
    Inspired from SMOTE-NC.
    :param batch: torch.Tensor : The batch data to transform.
    :param cat_columns_indices: Array-like object of the indexes of the categorical columns.
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
            similarities = pairwise_cosine_similarity(batch, full_data[batch_start_index:batch_end_index])
        elif dist == 'euclidean':
            # ToDo (below is the non-vectorized code)
            # similarities = torch.cdist(batch[i].view(1, -1), full_data)
            # similarities[i] += torch.inf  # This way, itself wont be in the most similar instances
            # topk_similar_indexes = similarities.topk(k=k_neighbors, largest=False).indices
            pass

        full_similarities_matrix = torch.cat([full_similarities_matrix, similarities], dim=1)

        batch_start_index += batch_size
        batch_end_index = min((batch_end_index + batch_size), full_data.shape[0])

    full_similarities_matrix -= torch.eye(len(batch), len(full_data), device=device)  # This way, itself won't be in the most similar instances

    if k_neighbors > full_similarities_matrix.shape[1]:
        print(f"Clipping k_neighbors={k_neighbors} to new max value {full_similarities_matrix.shape[1]}")
        k_neighbors = full_similarities_matrix.shape[1]

    batch_topk_similar_indexes = full_similarities_matrix.topk(k=k_neighbors, dim=1).indices

    # Select a random point between the k closest
    selected_points_indices = torch.gather(batch_topk_similar_indexes, 1, torch.randint(low=0, high=k_neighbors, size=(len(batch),), device=device).view(-1, 1))
    selected_points_indices = selected_points_indices.flatten()

    selected_points = full_data[selected_points_indices]

    batch_diff_vect = (selected_points - batch) * torch.rand(len(batch), device=device).view(-1, 1)

    augmented_batch = batch + batch_diff_vect  # At this point, the categorical values are wrong, next line fixes that

    if cat_columns_indices is not None and len(cat_columns_indices) > 0:
        # The categorical features become the most represented value among the k neighbors
        augmented_batch[:, cat_columns_indices] = full_data[:, cat_columns_indices][batch_topk_similar_indexes].mode(1).values

    return augmented_batch
