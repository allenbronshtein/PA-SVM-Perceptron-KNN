import numpy as np
TEST_SIZE = 0

def normalize(x,test_x):
    # min-max normalization
    max_v = np.amax(x, axis=0)
    min_v = np.amin(x, axis=0)
    for i in range(240):
        for j in range(5):
            x[i][j] = (x[i][j] - min_v[j]) / (max_v[j] - min_v[j])
            if i < TEST_SIZE:
                test_x[i][j] = (test_x[i][j] - min_v[j]) / (max_v[j] - min_v[j])

def knn(x,y):
    print("unimplemented knn")

def passive_agressive(x,y):
    print("unimplemented pa")

def svm(x,y):
    print("unimplemented svm")

def perceptron(x,y):
    print("unimplemented perceptron")

def loss(x,y,w):
    print("unimplemeted loss")

x = np.genfromtxt('PA-SVM-Perceptron-KNN/train_x.txt', delimiter=',')
y = np.genfromtxt('PA-SVM-Perceptron-KNN/train_y.txt')
test_x = np.genfromtxt('PA-SVM-Perceptron-KNN/test_x.txt', delimiter=',')
TEST_SIZE = test_x.size/5
normalize(x,test_x)