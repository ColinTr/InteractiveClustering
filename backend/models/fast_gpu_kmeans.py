# Software Name : InteractiveClustering
# Version: 1.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT License,
# the text of which is available at https://spdx.org/licenses/MIT.html
# or see the "license.txt" file for more details.


class fast_gpu_kmeans:
    def __init__(self, k_clusters):
        super(fast_gpu_kmeans, self).__init__()

        self.centroids = None
        self.k_clusters = k_clusters
        self.inertia = math.inf

    def init_centroids(self, x, n_local_trials=None):
        """
        Using k-means++ to initialize the centroids.
        Each new centroid is chosen from the remaining data points with a probability.
        proportional to its squared distance from the points closest cluster center.
        """
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(self.k_clusters))

        # (1) Choose one initial cluster center randomly (uniform random)
        self.centroids = x[random.randint(0, len(x) - 1)].unsqueeze(0)

        # (2) Initialize the list of pairwise distances
        pairwise_distances = euclidean_distance_2d_1d(x, self.centroids[-1]).reshape(-1, 1)

        while len(self.centroids) < self.k_clusters:
            d2, _ = torch.min(pairwise_distances, dim=1)  # Keep the distance to the closest cluster center only
            d2_sum = d2.sum()
            if d2_sum != 0:
                prob = d2 / d2.sum()  # Define the probability of being chosen
                cumul_prob = torch.cumsum(prob, dim=0)
                rand_indices = []
                for r in torch.rand(n_local_trials):
                    ind = (cumul_prob >= r).nonzero()[0].item()
                    rand_indices.append(ind)
                distances_to_rand_centroids = torch.cdist(x, x[rand_indices])
                inertias_of_rand_centroids = (distances_to_rand_centroids ** 2).sum(dim=0)
                best_candidate_index = inertias_of_rand_centroids.argmin()
                pairwise_distances = torch.cat((pairwise_distances, distances_to_rand_centroids[:, best_candidate_index].reshape(-1, 1)), dim=1)  # Add the distance to the latest new centroid to the list
                self.centroids = torch.cat((self.centroids, x[rand_indices[best_candidate_index]].unsqueeze(0)), dim=0)  # Add the new centroid to the list of centroids
            else:  # Marginal case where all the points are at a distance of 0 to their closest centroid
                best_candidate_index = torch.randint(low=0, high=len(d2), size=(1,))
                distances_to_rand_centroid = torch.cdist(x, x[best_candidate_index])
                pairwise_distances = torch.cat((pairwise_distances, distances_to_rand_centroid), dim=1)  # Add the distance to the latest new centroid to the list
                self.centroids = torch.cat((self.centroids, x[best_candidate_index]), dim=0)  # Add the new centroid to the list of centroids

    def make_centroids_converge(self, x, tolerance=1e-10, n_iterations=1000):
        """
        Make the centroids converge using the base k-means algorithm.
        Convergence will stop if we either reach n_iterations or if the shift is smaller than the tolerance.
        """
        inertia = math.inf
        labels = None

        for it in range(n_iterations):
            centroids_previous_position = self.centroids.clone()

            # For each unlabeled point, get the dist to the closest cluster and the cluster index
            dist = torch.cdist(x, self.centroids, p=2)
            min_dist, labels = torch.min(dist, dim=1)
            inertia = (min_dist ** 2).sum()

            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=len(self.centroids)).type(x.dtype)
            sum_points = torch.matmul(one_hot_labels.t(), x)
            label_counts = one_hot_labels.sum(dim=0)
            mean_points = sum_points / label_counts.reshape(-1, 1)

            nan_mean_points_mask = torch.any(torch.isnan(mean_points), dim=1)
            self.centroids[~nan_mean_points_mask] = mean_points[~nan_mean_points_mask]

            center_shift = torch.sum(torch.sqrt(torch.sum((self.centroids - centroids_previous_position) ** 2, dim=1)))
            if center_shift ** 2 < tolerance:
                # Stop the converge if the centroids don't move much
                break

        # print(f"Stopped after {it} iterations (center_shift={center_shift})")
        return inertia, labels

    def fit_predict(self, x, n_local_trials=None, tolerance=1e-10, n_iterations=1000, n_init=10):
        """
        For n_init executions, initialize and converge the centroids.
        We keep the centroids that achieved the smallest inertia.
        """
        best_inertia = math.inf
        best_centroids = None
        best_labels = None

        for init in range(n_init):
            self.init_centroids(x, n_local_trials=n_local_trials)

            inertia, labels = self.make_centroids_converge(x, tolerance=tolerance, n_iterations=n_iterations)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids.clone()
                best_labels = labels

        # Take the centroids of the iteration that achieved the best inertia
        self.inertia = best_inertia
        self.centroids = best_centroids

        return best_labels

    def predict(self, x):
        u_dist = torch.cdist(x, self.centroids, p=2)
        u_mindist, u_labels = torch.min(u_dist, dim=1)
        return u_labels
    

def euclidean_distance_2d_1d(x_2d, y_1d):
    # Expand dimensions of y to match the shape of x along the second dimension
    y_expanded = y_1d.unsqueeze(0).expand(x_2d.size(0), -1)
    # Calculate Euclidean distance
    return torch.norm(x_2d - y_expanded, dim=1)