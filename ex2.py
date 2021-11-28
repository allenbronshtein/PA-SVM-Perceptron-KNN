import numpy as np
from operator import itemgetter
from sys import argv

# Globals

# TRAIN_X_DIR = 'PA-SVM-Perceptron-KNN/train_x.txt'
# TRAIN_Y_DIR = 'PA-SVM-Perceptron-KNN/train_y.txt'
# TEST_X_DIR = 'PA-SVM-Perceptron-KNN/test_x.txt'
# OUTPUT_DIR = 'PA-SVM-Perceptron-KNN/output.txt'

TRAIN_X_DIR = argv[1]
TRAIN_Y_DIR = argv[2]
TEST_X_DIR = argv[3]
OUTPUT_DIR = argv[4]


# min-max normalization
def normalize():
    max_v, min_v = np.amax(train_x, axis=0), np.amin(train_x, axis=0)
    for i in range(train_size):
        for j in range(num_att):
            train_x[i][j] = (train_x[i][j] - min_v[j]) / (max_v[j] - min_v[j])
            if i < test_size:
                test_x[i][j] = (test_x[i][j] - min_v[j]) / \
                    (max_v[j] - min_v[j])


# Shuffle training set
def shuffle():
    np.random.shuffle(s)
    return zip(train_x[s], train_y[s])


# logs output to file
def log():
    msg = ''
    for i in range(test_size):
        msg += f"knn: {knn_test_y[i]}, perceptron: {perceptron_test_y[i]}, svm: {svm_test_y[i]}, pa: {pa_test_y[i]}\n"
    with open(OUTPUT_DIR, 'w') as out:
        out.write(msg)


# KNN learning algorithm
def knn():
    KN = 7
    for x in test_x:
        dist, classes = [], {0: 0, 1: 0, 2: 0}
        for i in range(train_size):
            dist.append([np.linalg.norm(x - train_x[i]), train_y[i]])
        dist.sort(key=itemgetter(0))
        nearest_neighbors = dist[:KN]
        for i in range(KN):
            classes[nearest_neighbors[i][1]] += 1
        knn_test_y.append(max(classes, key=classes.get))


# Perceptron learning algorithm
def perceptron():
    LEARNING_RATE, EPOCHS = 0.2, 5
    w = np.array([np.zeros(num_att), np.zeros(num_att), np.zeros(num_att)])
    for _ in range(EPOCHS):
        train_set = shuffle()
        for x, y in train_set:
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[y, :] = w[y, :] + LEARNING_RATE * x
                w[y_hat, :] = w[y_hat, :] - LEARNING_RATE * x
    for x in test_x:
        perceptron_test_y.append(np.argmax(np.dot(w, x)))


# Passive agressive learning algorithm
def passive_agressive():
    EPOCHS = 1
    w = np.random.random((3, num_att))
    for _ in range(EPOCHS):
        train_set = shuffle()
        for x, y in train_set:
            w_out = np.delete(w, y, 0)
            y_hat = np.argmax(np.dot(w_out, x))
            if y == 0 or (y == 1 and y_hat == 1):
                y_hat += 1
            loss = max(0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x))
            if loss:
                tau = 1
                denominator = 2 * (np.linalg.norm(x) ** 2)
                if denominator:
                    tau = loss / denominator
                w[y, :] = w[y, :] + tau * x
                w[y_hat, :] = w[y_hat, :] - tau * x
    for x in test_x:
        pa_test_y.append(np.argmax(np.dot(w, x)))


# SVM learning algorithm
def svm():
    EPOCHS, ETA, LAMBDA = 20, 1.1, 0.1
    w = np.random.random((3, num_att))
    for e in range(EPOCHS):
        train_set = shuffle()
        ETA /= (e + 1)
        for x, y in train_set:
            y_hat = np.argmax(np.dot(w, x))
            w *= (1 - ETA * LAMBDA)
            w_out = np.delete(w, y_hat, 0)
            second = np.argmax(np.dot(w_out, x))
            loss = max(0, 1 - np.dot(w[y, :], x) + np.dot(w_out[second, :], x))
            if loss > 0:
                w[y, :] = w[y, :] + ETA * x
                w[y_hat, :] = w[y_hat, :] - ETA * x
    for x in test_x:
        svm_test_y.append(np.argmax(np.dot(w, x)))


# Main
train_x = np.genfromtxt(TRAIN_X_DIR, delimiter=',')
train_y = np.genfromtxt(TRAIN_Y_DIR).astype(int)
test_x = np.genfromtxt(TEST_X_DIR, delimiter=',')
num_att, train_size, test_size = train_x[0].size, train_x.shape[0], test_x.shape[0]
knn_test_y, perceptron_test_y, svm_test_y, pa_test_y = [], [], [], []
s = np.arange(train_x.shape[0])
normalize()
knn(), perceptron(), passive_agressive(), svm()
log()
# End
