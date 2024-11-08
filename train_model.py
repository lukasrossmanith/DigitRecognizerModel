import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl 


# this will be  a model that can recognize handwritten digits 
# it will be trained on a given dataset (see train.csv or https://www.kaggle.com/competitions/digit-recognizer/data)
# it will be a simple neural network (input of 784 pixel values, 2 layers of 10 neurons, argmax output)
# activation functions will be ReLU within the 1st and softmax in the 2nd layer (to convert the values to probabilities)
# parameters: two 10x1 bias vectors, one 784x10 weight matrix, one 10x10 weight matrix


# formatting training and testing data
data = pd.read_csv('train.csv')

data = np.array(pd.read_csv('train.csv'))
data_test = data[:1000].T
data_train = data[1000:].T
X_test, y_test, X_train, y_train = data_test[1:] / 255, data_test[0], data_train[1:] / 255, data_train[0]


# rectified linear unit and its derivative
def relu(x):
    return np.maximum(x, 0)


def relu_deriv(x):
    return (x > 0)


# softmax function
def softmax(x):
    return np.exp(x) / sum(np.exp(x), 0)


# one hot encoding digits 0,1,...,9 (e.g. 3 = [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0])
def one_hot(y):
    one_hot = np.zeros((y.size, 10))
    for i in range(y.size):
        one_hot[i, y[i]] = 1
    return one_hot.T


# initialize parameters
def init_params():
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2


# forward propagation
def forward_prop(X, W1, b1, W2, b2):
    L1 = np.dot(W1, X) + b1
    A1 = relu(L1)
    L2 = np.dot(W2, A1) + b2
    A2 = softmax(L2)
    return L1, A1, L2, A2


# turn probability vector into prediction (digit)
def predict(y_fit):
    return np.argmax(y_fit, 0)


# get accuracy of predictions 
def get_acc(predictions, y):
    return np.sum(predictions == y) / y.size


# compute error for parameters
def backprop(L1, A1, L2, A2, W1, W2, X, y):
    p = y.size
    one_hot_y = one_hot(y)
    dL2 = A2 - one_hot_y
    dW2 = 1 / p * np.dot(dL2, A1.T)
    db2 = 1 / p * np.sum(dL2)
    dL1 = np.dot(W2.T, dL2) * relu_deriv(L1)
    dW1 = 1 / p * np.dot(dL1, X.T)
    db1 = 1 / p * np.sum(dL1)
    return dW1, db1, dW2, db2


# update parameters according to error
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, a):
    W1 = W1 - a*dW1
    b1 = b1 - a*db1
    W2 = W2 - a*dW2 
    b2 = b2 - a*db2
    return W1, b1, W2, b2


# perform gradient descent algorithm for a given number of iterations
def gradient_descent(X_train, X_test, y_train, y_test, iterations=500, alpha=0.1, print_out=False, save_report=False, plot_report=False, xlsx_dir='', xlsx_name='', plt_dir='', plt_name=''):
    iterations = int(iterations)  # handles case that iteration is given in scientific notation
    iters, train_acc_arr, test_acc_arr = np.array([]), np.array([]), np.array([])
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        L1, A1, L2, y_fit = forward_prop(X_train, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backprop(L1, A1, L2, y_fit, W1, W2, X_train, y_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)  
        # getting accuracy on training data
        train_predictions = predict(y_fit)
        train_accuracy = get_acc(train_predictions, y_train)
        # getting accuracy on testing data
        _, _, _, y_test_fit = forward_prop(X_test, W1, b1, W2, b2)
        test_predictions = predict(y_test_fit)
        test_accuracy = get_acc(test_predictions, y_test)
        iters = np.hstack((iters, np.array([i])))
        train_acc_arr = np.hstack((train_acc_arr, np.array([train_accuracy])))
        test_acc_arr = np.hstack((test_acc_arr, np.array([test_accuracy])))
        if print_out:
                if (i+1) % 10 == 0:
                    print(f'iteration: {i+1}')
                    print(f'training accuracy: {train_accuracy}')
                    print(f'test accuracy: {test_accuracy}')
    # creating columns for report csv
    iters_col = [iters[i] + 1 for i in range(0, iterations) if (i+1)%10==0]
    train_acc_col = [train_acc_arr[i] for i in range(0, iterations) if i%10==0]
    test_acc_col = [test_acc_arr[i] for i in range(0, iterations) if i%10==0]
    if save_report:    
        num_data = np.array([iters_col, train_acc_col, test_acc_col]).T
        col_names = np.array(['Iterations', 'Training Accuracy', 'Testing Accuracy'])
        report_data = list(np.vstack((col_names, num_data)))
        report = pd.DataFrame(columns = report_data)
        report.to_excel(excel_writer=xlsx_dir+xlsx_name+'.xlsx', sheet_name='Accuracy Report')
    if plot_report:
        plt.plot(iters, train_acc_arr)
        plt.plot(iters, test_acc_arr)
        plt.title('Accuracy Report')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.xlim(left=0, right=iterations+10)
        plt.ylim(bottom=0, top=1.1)
        plt.xticks(np.arange(0, iterations + 1, int(iterations/10)))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='upper left')
        plt.savefig(plt_dir + plt_name + '.pdf')

    return W1, b1, W2, b2









