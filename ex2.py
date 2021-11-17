import numpy as np

def normalize(train_x,test_x):
    # min-max normalization
    max_v, min_v = np.amax(train_x, axis=0), np.amin(train_x, axis=0)
    for i in range(train_size):
        for j in range(num_att):
            train_x[i][j] = (train_x[i][j] - min_v[j]) / (max_v[j] - min_v[j])
            if i < test_size:
                test_x[i][j] = (test_x[i][j] - min_v[j]) / (max_v[j] - min_v[j])

def knn(train_x,test_x):
    k = 7
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
        nearest_neighbors = sorted_array[:k]
        # find the number of occurrences for each class
        for line in range(k):
            classes[str(int(nearest_neighbors[line][1]))] += 1
        # predict the majority class and assign to current x example
        max_class = max(classes, key=classes.get)
        test_y.append(int(max_class))
    return test_y

def passive_agressive(x,y):
    print("unimplemented pa")

def svm(x,y):
    print("unimplemented svm")

def perceptron(x,y):
    print("unimplemented perceptron")

def loss(x,y,w):
    print("unimplemeted loss")

train_x = np.genfromtxt('PA-SVM-Perceptron-KNN/train_x.txt', delimiter=',')
train_y = np.genfromtxt('PA-SVM-Perceptron-KNN/train_y.txt')
test_x = np.genfromtxt('PA-SVM-Perceptron-KNN/test_x.txt', delimiter=',')
num_att = train_x[0].size
train_size = int(train_x.size/num_att)
test_size = int(test_x.size/num_att)

normalize(train_x,test_x)

knn(train_x,test_x)