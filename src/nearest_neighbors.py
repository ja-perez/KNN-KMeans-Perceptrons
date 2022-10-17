import numpy as np
from scipy.spatial import distance


def KNN_test(X_train, Y_train, X_test, Y_test, K):
    calculated_dist = {}
    k_dist, k_labels = [], []
    results = []
    for i, test_point in enumerate(X_test):
        for j, train_point in enumerate(X_train):
            dist = round(distance.euclidean(test_point, train_point), 2)
            calculated_dist[dist] = Y_train[i]
        k_dist = sorted(calculated_dist.keys())[:K]
        k_labels.append(sum([calculated_dist[dist] for dist in k_dist]))
        results.append(1 if (k_labels[-1] > 0 and Y_test[i] > 0) or (k_labels[-1] < 0 and Y_test[i] < 0) else 0)
    return sum(results) / len(results)


def choose_K(X_train, Y_train, X_val, Y_val):
    calculated_dist = {}
    k_dist, k_labels = [], []
    results = []
    k_results = {}
    for curr_k in range(3, len(X_train), 2):
        for i, test_point in enumerate(X_val):
            for j, train_point in enumerate(X_train):
                dist = round(distance.euclidean(test_point, train_point), 2)
                calculated_dist[dist] = Y_train[i]
            k_dist = sorted(calculated_dist.keys())[:curr_k]
            k_labels.append(sum([calculated_dist[dist] for dist in k_dist]))
            results.append(1 if (k_labels[-1] > 0 and Y_val[i] > 0) or (k_labels[-1] < 0 and Y_val[i] < 0) else 0)
        accuracy = sum(results) / len(results)
        k_results[accuracy] = curr_k
    return k_results[max(k_results.keys())]
