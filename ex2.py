import numpy as np

# min-max normalization


def normalize(train_x, test_x):
    max_v, min_v = np.amax(train_x, axis=0), np.amin(train_x, axis=0)
    for i in range(train_size):
        for j in range(num_att):
            train_x[i][j] = (train_x[i][j] - min_v[j]) / (max_v[j] - min_v[j])
            if i < test_size:
                test_x[i][j] = (test_x[i][j] - min_v[j]) / \
                    (max_v[j] - min_v[j])


# KNN learning algorithm
def knn(train_x, train_y, test_x):
    KN = 7
    test_y = []
    # find the distance for each test x from every train x
    for x in test_x:
        dist = []
        classes = {'0': 0, '1': 0, '2': 0}
        # save distance and matching label in a pair
        for i in range(test_size):
            dist.append([np.linalg.norm(x - train_x[i]), train_y[i]])
        dist_np = np.array(dist)
        # find k nearest neighbors
        sorted_array = dist_np[np.argsort(dist_np[:, 0])]
        nearest_neighbors = sorted_array[:KN]
        # find the number of occurrences for each class
        for line in range(KN):
            classes[str(int(nearest_neighbors[line][1]))] += 1
        # predict the majority class and assign to current x example
        max_class = max(classes, key=classes.get)
        test_y.append(int(max_class))
    return test_y


# Perceptron learning algorithm
def perceptron(train_x, train_y, test_x):
    LEARNING_RATE = 0.2
    EPOCHS = 20
    w = np.array([np.zeros(num_att), np.zeros(num_att), np.zeros(num_att)])
    test_y = []
    for _ in range(EPOCHS):
        # shuffle train data every epoch
        s = np.arange(train_x.shape[0])
        np.random.shuffle(s)
        train_x_shuffled = train_x[s]
        train_y_shuffled = train_y[s]
        # predict and update
        for x_i, y_i in zip(train_x_shuffled, train_y_shuffled):
            y_hat = np.argmax(np.dot(w, x_i))
            if y_hat != int(y_i):
                w[int(y_i), :] = w[int(y_i), :] + LEARNING_RATE * x_i
                w[y_hat, :] = w[y_hat, :] - LEARNING_RATE * x_i
    # use the trained weight vectors to assign best label prediction to current x example
    for x in range(test_size):
        test_y.append(np.argmax(np.dot(w, test_x[x])))
    return test_y


# Passive agressive learning algorithm
def passive_agressive(train_x, train_y, test_x):
    EPOCHS = 1
    w = np.array([np.random.rand(num_att), np.random.rand(
        num_att), np.random.rand(num_att)])
    test_y = []
    for _ in range(EPOCHS):
        # shuffle train data every epoch
        s = np.arange(train_x.shape[0])
        np.random.shuffle(s)
        train_x_shuffled = train_x[s]
        train_y_shuffled = train_y[s]
        # predict and update
        for x_i, y_i in zip(train_x_shuffled, train_y_shuffled):
            w_out = np.delete(w, int(y_i), 0)
            y_hat = np.argmax(np.dot(w_out, x_i))
            # get the correct label by it's place in original w
            if y_i == 0 or (y_i == 1 and y_hat == 1):
                y_hat += 1
            loss = max(0, 1 - np.dot(w[int(y_i)], x_i) + np.dot(w[y_hat], x_i))
            if loss:
                tau = 1
                denominator = 2 * (np.linalg.norm(x_i) ** 2)
                # when denominator is not 0 change initialization by formula
                if denominator:
                    tau = loss / denominator
                # update w
                w[int(y_i), :] = w[int(y_i), :] + tau * x_i
                w[y_hat, :] = w[y_hat, :] - tau * x_i
    # use the trained weight vectors to assign best label prediction to current x example
    for x in range(test_size):
        test_y.append(np.argmax(np.dot(w, test_x[x])))
    return test_y


# SVM learning algorithm
def svm(train_x, train_y, test_x):
    test_y = []
    return test_y


# Main
train_x = np.genfromtxt('PA-SVM-Perceptron-KNN/train_x.txt', delimiter=',')
train_y = np.genfromtxt('PA-SVM-Perceptron-KNN/train_y.txt')
test_x = np.genfromtxt('PA-SVM-Perceptron-KNN/test_x.txt', delimiter=',')
num_att = train_x[0].size
train_size = int(train_x.size/num_att)
test_size = int(test_x.size/num_att)
knn_test_y, perceptron_test_y, svm_test_y, pa_test_y = [], [], [], []


normalize(train_x, test_x)
knn_test_y = knn(train_x, train_y, test_x)
perceptron_test_y = perceptron(train_x, train_y, test_x)
pa_test_y = passive_agressive(train_x, train_y, test_x)

msg = ''
for i in range(test_size):
    msg += f"knn: {knn_test_y[i]}, perceptron: {perceptron_test_y[i]}, svm: 0, pa: {pa_test_y[i]}\n"
_out = open('PA-SVM-Perceptron-KNN/output.txt', "w")
_out.write(msg)
_out.close()
# End
