import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):  # y=None for sklearn Pipeline compatibility
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initializing the centroids randomly
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

        for _ in range(self.max_iters):

            # Assignment step where each point is assigned the nearest centriod based on their euclidean distances
            #np.newaxis is used for adding an extra dimension of 1 [for eg (a,b) -> (a,1,b)]
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
                axis=2
            )
            labels = np.argmin(distances, axis=1)

            # Update step where the centroids are updated to the mean of its cluster (minimizes the squared distance)
            new_centroids = np.zeros_like(self.cluster_centers_)

            for k in range(self.n_clusters):
                cluster_points = X[labels == k]

                if len(cluster_points) == 0:
                    new_centroids[k] = self.cluster_centers_[k]
                else:
                    new_centroids[k] = np.mean(cluster_points, axis=0)

            # 4. Convergence check
            centroid_shift = np.linalg.norm(
                new_centroids - self.cluster_centers_
            )

            self.cluster_centers_ = new_centroids

            if centroid_shift < self.tol:
                break

        self.labels = labels
        self.inertia_ = self._compute_inertia(X)

        return self

    def _compute_inertia(self, X):
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            inertia += np.sum(
                (cluster_points - self.cluster_centers_[k]) ** 2
            )
        return inertia

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels

            

        