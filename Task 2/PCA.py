import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):  # <-- y added
        X = np.asarray(X)

        # Centering the data is the first step of PCA
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        n_samples = X_centered.shape[0]

        # Computing the covariance matrix as this measures how features vary together
        covariance_matrix = (1 / (n_samples - 1)) * (X_centered.T @ X_centered)

        # We will do eigen decomposition where; Eigenvectors -> principal directions, Eigenvalues -> variance captured
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sorting in descending order as we want maximum variance i.e. most important components should be first
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = (
            self.explained_variance / np.sum(eigenvalues)
        )

        return self

    def transform(self, X):
        X = np.asarray(X)
        X_centered = X - self.mean_

        # Reducing the dataset and returning it
        return X_centered @ self.components

    def fit_transform(self, X, y=None):  # <-- y added
        self.fit(X, y)
        return self.transform(X)
