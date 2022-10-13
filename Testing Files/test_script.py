import numpy as np
import src.nearest_neighbors as nn
import src.clustering as kmeans
import src.perceptron as percept


def load_data(file_data):
    data = np.genfromtxt(file_data, skip_header=1, delimiter=',')
    X = []
    Y = []
    for row in data:
        temp = [float(x) for x in row]
        temp.pop(-1)
        X.append(temp)
        Y.append(int(row[-1]))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


X, Y = load_data("nearest_neighbors_1.csv")
acc = nn.KNN_test(X, Y, X, Y, 1)
print("KNN:", acc)
# Added by me
best_k = nn.choose_K(X, Y, X, Y)
print("Optimal K:", best_k)

X = np.genfromtxt("clustering_1.csv", skip_header=1, delimiter=',')
mu = np.array([[1], [5]])
mu = kmeans.K_Means(X, 2, mu)
print("KMeans:", mu)

X, Y = load_data("perceptron_1.csv")
W = percept.perceptron_train(X, Y)
acc = percept.perceptron_test(X, Y, W[0], W[1])
print("Percept:", acc)
