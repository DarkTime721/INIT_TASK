# Creating a KNN Regressor from scratch so that we can use it in exam score predictions
# KNN does not learn a model during training, what it does is it stores the entire training dataset and make decisions only at prediction time

import numpy as np

# First we calculate the euclidean distances to measure the geometric closeness
def euclidean_distance(X_test, X_train):
    
    X_test_sq = np.sum(X_test ** 2, axis=1).reshape(-1,1)
    X_train_sq = np.sum(X_train ** 2, axis=1)
    cross_term = X_test @ X_train.T

    return np.sqrt(X_test_sq - 2*cross_term + X_train_sq)


class KNNRegression:
    def __init__(self, n_neighbors=3, weights='uniform'):
        self.n_neighbors = n_neighbors  # Controls smoothness of predictions
        self.weights = weights  # Performs uniform or weighted distance prediction

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        distances = euclidean_distance(X, self.X_train)

        # Selecting the index of the neighbors with closest distances and their values 
        neighbor_idx = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbor_values = self.y_train[neighbor_idx]

        if self.weights == 'uniform':
            return np.mean(neighbor_values, axis=1)
        
        elif self.weights == 'distance':
            neighbor_distances = np.take_along_axis(
                distances, neighbor_idx, axis=1
            )
            # Creates a list of the neighbor distances from the distances array (index is taken from neighbor_index) 

            weights = 1/ (neighbor_distances + 1e-8)    # Preventing division by zero

            weighted_sum = np.sum(weights * neighbor_values, axis=1)
            weight_total = np.sum(weights, axis=1)

            return weight_total/weighted_sum
