import numpy as np

X = np.genfromtxt('PA-SVM-Perceptron-KNN/train_x.txt', delimiter=',')
Y = np.genfromtxt('PA-SVM-Perceptron-KNN/train_y.txt')
test_X = np.genfromtxt('PA-SVM-Perceptron-KNN/test_x.txt', delimiter=',')


def knn(X,Y):
    print("unimplemented knn")

def passive_agressive(X,Y):
    print("unimplemented pa")

def svm(X,Y):
    print("unimplemented svm")

def perceptron(X,Y):
    print("unimplemented perceptron")
