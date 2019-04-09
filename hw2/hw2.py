import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import decomposition
from sklearn import datasets
from matplotlib.pyplot import *
import seaborn as sns

import os, re
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier

class DataSet:
    def __init__(self, data_set):
        """
        Initialize a data set and load both training and test data
        DO NOT MODIFY THIS FUNCTION
        """
        self.name = data_set

        # The training and test labels
        self.labels = {'train': None, 'test': None,'valid':None}

        # The training and test examples
        self.examples = {'train': None, 'test': None,'valid':None}

        # Load all the data for this data set
        for data in ['train', 'test','valid']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]
        self.num_test = self.examples['test'].shape[0]
        self.num_valid = self.examples['valid'].shape[0]
        self.dim = self.examples['train'].shape[1]

    def load_file(self, dset_type):
        """
        Load a training set of the specified type (train/test). Returns None if either the training or test files were
        not found. NOTE: This is hard-coded to use only the first seven columns, and will not work with all data sets.
        DO NOT MODIFY THIS FUNCTION
        """
        path = './data/{0}.{1}'.format(self.name, dset_type)
        try:
            file_contents = np.genfromtxt(path, missing_values=0, skip_header=0,
                                          dtype=float, delimiter=",")
            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))


def get_normed_mean_cov(X):
    X_std = StandardScaler().fit_transform(X)
    X_mean = np.mean(X_std, axis=0)
    
    ## Automatic:
    #X_cov = np.cov(X_std.T)
    
    # Manual:
    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0]-1)
    
    return X_std, X_mean, X_cov


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    length = len(y_true)

    error_cnt = 0

    for i in range (length):
        if y_true[i] != y_pred[i]:
            # print(y_true[i],y_pred[i])
            error_cnt = error_cnt+1
    error = (1/length) * error_cnt
    return error

if __name__ == '__main__':

    # Load a data set
    data = DataSet('usps')
    
    # loading train and test sets
    X_train = data.examples['train']
    Y_train = data.labels['train']

    X_test = data.examples['test']
    Y_test = data.labels['test']

    X_valid = data.examples['valid']
    Y_valid = data.labels['valid']

    X_std, X_mean, X_cov = get_normed_mean_cov(X_train)
    X_std_validation, _, _ = get_normed_mean_cov(X_test)
    eig_vals, eig_vecs = np.linalg.eig(X_cov)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # print('Eigenvalues in descending order:')
    # for i in range(0,1):
    #     print(eig_pairs[i])
    #     mat = eig_pairs[i][1].reshape(16,16)
    #     print(mat)
    #     plt.imshow(mat)
    #     gca().grid(False)
    #     plt.show()


    clf = PCA (n_components=256)
    #clf = PCA (0.9)
    clf.fit(X_train)
    X_pca = clf.transform(X_train)
    x_70 = clf.inverse_transform(X_pca)
    
    clf2 = SGDClassifier(loss="hinge", penalty="l2", alpha = 0.001)
    clf2 = clf2.fit(x_70, Y_train)  

    pred = clf2.predict(X_test)
    error = compute_error(Y_test, pred)
    print(error)

    variance = clf.explained_variance_ratio_

    var = np.cumsum(np.round(clf.explained_variance_ratio_, decimals=3)*100)
    # print(var)

    # plt.ylabel('cumulative variance (%)')
    # plt.xlabel('number of components')
    # plt.title('PCA Analysis')
    # plt.ylim(14,100)
    
    # plt.style.context('seaborn-whitegrid')
    # plt.plot(var)
    # plt.plot(17,70,'o',label='k70') 
    # plt.plot(28,80,'o',label='k80') 
    # plt.plot(52,90,'o',label='k90') 
    # plt.text(21,70,  'k70', fontsize=9) 
    # plt.text(31,80, 'k80', fontsize=9) 
    # plt.text(57,90,  'k90', fontsize=9) 


    # plt.show()

    # myNumber = 91
    # idx = (np.abs(var-myNumber)).argmin()
    # print(idx)


    idx = eig_vals.argsort()[::-1]   
    eigenValues = eig_vals[idx]
    eigenVectors = eig_vecs[:,idx]
 
    # print(newNewMat[0])
    # print(newNewMat.shape)
    # print(newMat[0])
    
    # 17, 29, 56
    newMat = eigenVectors[0:,0:]
    newNewMat = X_train.dot(newMat.T)
    validMat = X_valid.dot(newMat.T)
    testMat = X_test.dot(newMat.T)
    # 0.0001; 0.001; 0.01; 0.1
    clf2 = SGDClassifier(loss="hinge", penalty="l2", alpha = 0.0001)
    clf2 = clf2.fit(newNewMat, Y_train)  
    # pred = clf2.predict(validMat)
    # error = compute_error(Y_valid, pred)
    pred = clf2.predict(testMat)
    error = compute_error(Y_test, pred)
    # print(error)
    # print(cross_val_score(clf2,newNewMat, Y_train))

    