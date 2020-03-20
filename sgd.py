import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing


def helper():
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"  # 0,8
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


"""
  Implements Hinge loss using SGD.
  returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
  """
def SGD(data, labels, C, eta_0, T):
    # Start with the all-zeroes weight vector
    w = np.zeros(data.shape[1])
    for t in range(1, T + 1):
        i = np.random.randint(0, len(data) - 1)
        eta_t = eta_0 / t
        x = data[i]
        y = labels[i]
        # max 0, ywx-1
        if y * (np.dot(w, x)) < 1:
            w = (1 - eta_t) * w + eta_t * C * y * x
        else:
            w = (1 - eta_t) * w
    return w


def calc_accuracy(w, data, labels):
    err = 0
    for i in range(len(data)):
        if np.dot(w, data[i]) > 0:
            prediction = 1
        else:
            prediction = -1
        if prediction != labels[i]:
            err += 1
    accuracy = 1 - err / len(data)
    return accuracy


def find_best_eta():
    T = 1000
    C = 1
    # 10**5 causes overflow
    # higher resolution
    eta_0_list = [10 ** (i * 0.5) for i in range(-10, 12)]
    mean = list()
    for eta_0 in eta_0_list:
        accuracy = [0] * 10
        for i in range(10):
            w = SGD(train_data, train_labels, C, eta_0, T)
            accuracy[i] = calc_accuracy(w, validation_data, validation_labels)
        mean.append(np.mean(accuracy))
    best_eta = eta_0_list[mean.index(np.max(mean))]
    # plot eta performance
    plt.plot(eta_0_list, mean, color='r')
    plt.xscale('log')
    plt.xlabel("eta_0")
    plt.ylabel("accuracy")
    red_patch = mpatches.Patch(color='red', label='Accuracy as function of eta-0')
    plt.legend(handles=[red_patch])
    plt.show()

    return best_eta


def find_best_C():
    best_eta = find_best_eta()
    T = 1000
    # 10**5 causes overflow
    # higher resolution
    C_list = [10 ** (i * 0.5) for i in range(-10, 12)]
    mean = list()
    for C in C_list:
        accuracy = [0] * 10
        for i in range(10):
            w = SGD(train_data, train_labels, C, best_eta, T)
            accuracy[i] = calc_accuracy(w, validation_data, validation_labels)
        mean.append(np.mean(accuracy))
    best_C = C_list[mean.index(np.max(mean))]
    # plot C performance
    print("C = ", best_C, "eta_0 = ", best_eta)
    plt.plot(C_list, mean, color='r')
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("accuracy")
    red_patch = mpatches.Patch(color='red', label='Accuracy as function of C, given eta-0')
    plt.legend(handles=[red_patch])
    plt.show()

    return best_C, best_eta


if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

    C, eta_0 = find_best_C()
    print("C = ", C, "eta_0 = ", eta_0)
    T = 20000
    w = SGD(train_data, train_labels, C, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.title("w_star_SGD")
    # show SGD weight vector as a pic
    plt.show()

    print("The accuracy of the SGD classifier with best C, eta_0 is " + str(calc_accuracy(w, test_data, test_labels)))
