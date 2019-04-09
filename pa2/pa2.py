from numpy import array
import operator
import random
import numpy as np


class DataSet:
    def __init__(self, data_set):
        """
        Initialize a data set and load both training and test data
        DO NOT MODIFY THIS FUNCTION
        """
        self.name = data_set

        # The training and test labels
        self.labels = {'train': None, 'test': None}

        # The training and test examples
        self.examples = {'train': None, 'test': None}

        # Load all the data for this data set
        for data in ['train', 'test']:
            self.load_file(data)

        # The shape of the training and test data matrices
        self.num_train = self.examples['train'].shape[0]
        self.num_test = self.examples['test'].shape[0]
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
                                          dtype=int, delimiter=",")
            self.labels[dset_type] = file_contents[:, 0]
            self.examples[dset_type] = file_contents[:, 1:]

        except RuntimeError:
            print('ERROR: Unable to load file ''{0}''. Check path and try again.'.format(path))


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    elements,counts = np.unique(x,return_counts = True)
    partitioned_dict = {}

    for row in elements:
        partitioned_dict[row] = []
        for index, item in enumerate(x):
            if item == row:
                partitioned_dict[row].append(index)

    return partitioned_dict

def mutual_information(x, y, w):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    

    total_entropy = entropy(y,w)

    partitioned_x = partition(x)
    weighted_entropy = 0
    # calculate the weighted entropy over the partition of x
    vals,counts= np.unique(x,return_counts=True)
    for key in partitioned_x:
        weighted_entropy += np.sum([(np.sum(w[partitioned_x[key]])/np.sum(w)) * entropy(y[partitioned_x[key]],w[partitioned_x[key]])])

    information_gain = total_entropy - weighted_entropy
    return information_gain


def entropy(y,w):

    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

	# my original entropy function commented below is not working as desired. The below implementation is based on from Sai Ram Chappidi's explanation

    # y_partition = partition(y)
    # elements,counts = np.unique(y,return_counts = True)
    # entropy=0

    # for i in range(len(elements)):
    #     entropy += ((-(np.sum(w[y_partition[i]])))/np.sum(w))*np.log2(np.sum(w[y_partition[i]])/np.sum(w))
    # return entropy

    entropy = 0
    # two hypothesis cases 0,1
    h = {0: 0, 1: 0}
    leny = len(y)
    for i in range(leny):
        # if y is 0 add 0 to the weight
        if y[i] == 0:
            h[0] += w[i]
        # if y is 1 add 1 to the weight
        elif y[i] == 1:
            h[1] += + w[i]
    # summing all the weighted values           
    val_sum = h[0] + h[1]

    # entropy calculation
    for j in range(len(h)):
        h[j] = h[j]/val_sum
        # to prevent divide by zero
        if h[j] != 0:
            entropy += h[j] * np.log2(h[j])
    entropy = -(entropy)
    return entropy



def id3(x, y, attributes, max_depth, weight, depth=0):
    """
    Extending the classical ID3 algorithm given training data (x), training labels (y)and an array of attributes
    to consider with weight. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attributes is empty (there is nothing to split on), then return the most common value of y
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y
    Otherwise the algorithm selects the next best attribute using INFORMATION GAIN as the splitting criterion and
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    See https://gkunapuli.github.io/files/cs6375/04-DecisionTrees.pdf (Slide 18) for more details.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current level.
    * The subtree itself can be nested dictionary, or a single label (leaf node).

    Returns a decision tree represented as a nested dictionary 
    of feature, value, prediction, weight, for example
    
    {(17, 4, 'True'): {(9, 0, 'True'): {(4, 5, 'True'): 0,
     (4, 5, 'False'): 1}, (9, 0, 'False'): 1},
     (17, 4, 'False'): {(16, 2, 'True'): 1,
     (16, 2, 'False'): 0}}          
    """
    tree = {}
    new_attr = []
    arr, count = np.unique(y, return_counts=True)
    # checking edge cases - reached maximum depth, or no attributes
    if len(attributes) == 0 or depth == max_depth or len(x) == 0:
        return np.bincount(y).argmax()
    # if all the values of y are one return one
    elif len(np.unique(y)) == 1:
        return arr[0]
    else:
        # calculating mutual information values
        info_values = {} 
        # over number of columns
        for i in range(data.dim):
            oldX = partition(x[:,i])
            oldKeys = oldX.keys()
            # check in attributes recieved from bagging
            for attr in attributes:
                binX = []
                key , value = attr
                # check for key and value
                if i == key and value in oldKeys:
                    # get the index
                    index = oldX[value]
                    for n in range(len(x)):
                        if n in index:
                            # if match binary classification 1 / 0 and appending to binX list
                            binX.append(1)
                        else:
                            binX.append(0)
                    # adding to a dictionary        
                    info_values[(i, value)] = mutual_information(binX, y, weight)
        
        # getting the maximum feature value
        best_feature_index = max(info_values, key=info_values.get)     
        best_feature, best_val = best_feature_index
        # creating the best partition
        x_best_part = partition(x[:,best_feature])
        #selecting other than the best feature value from the dictionary
        new_attr = list(filter(lambda x: x!= (best_feature, best_val), attributes))
        # increasing depth
        depth += 1

        # Calling id3 recursively, checking over 0,1 making a prediction as True / False 
        for n in range(0,2):
            if n == 0:
                # recursively calling id3 over the best values of the x partition
                bestX = x[x_best_part[best_val]]
                bestY = y[x_best_part[best_val]]
                tree[best_feature, best_val, 'True'] = id3(bestX, bestY, new_attr, max_depth,weight, depth)
            else:
    	        # recursively calling id3 selecting other than best features
                othr_idx = []
                for i in x_best_part:
                    if i != best_val:
                        othr_idx.extend(x_best_part[i])

                otherX = x[othr_idx]
                otherY = y[othr_idx]
                tree[best_feature, best_val, 'False'] = id3(otherX, otherY, new_attr, max_depth,weight, depth)
    return tree 


def bagging(x, y, max_depth, num_trees):
    """ Bagging function taking in x, y, max_dept 
    is sent to id3 and looped over num_trees (bag size) 
    returns weighted pair of hypotheses"""

    hypotheses = {}
    # attribute_idx = np.array(range(data.dim))
    # generating attributes
    attributes = []
    cols  = data.dim
    for i in range(cols):
        arr = np.unique(x[:, i])
        for value in arr:
            attributes.append((i, value))
    lena = len(x)
    # initializing weights to 1 for boosting
    alpha = 1
    w = np.ones((lena, 1), dtype=int)    
    # iterating over j number of trees
    for j in range(num_trees):
        # generating a random array of indicies with replacement over the length of x
        new_array = np.random.choice(lena,size =lena,replace=True)
        #calling id3 over the indices of the new array
        tree = id3(x[new_array],y[new_array],attributes, max_depth, w)
        # appending to a global tree as a weighted pair
        hypotheses[j] = (alpha, tree)

    return hypotheses

def boosting(x, y, max_depth, num_stumps):
    """ Boosting function taking in x, y, max_dept (stumps) is sent to id3 
    and looped over num_stumps returns weighted pair of hypotheses"""

    attribute_idx = np.array(range(data.dim))
    # generating attributes
    attributes = []
    cols  = data.dim
    for i in range(cols):
        arr = np.unique(x[:, i])
        for value in arr:
            attributes.append((i, value))
    lenx = len(x)
    # calculating weight column over the length of x
    w = np.ones(lenx) / lenx
    hypotheses = {}
    # looping over the number of stumps
    for t in range(num_stumps):
        #calling id3 over the indices of the new array
        tree = id3(x,y,attributes, max_depth, w)
        # getting the predictions over the tree(stump) returned
        trn_pred = [predict_example(x[i, :], tree) for i in range(data.num_train)]
        # weighted training error
        eps = w.dot(trn_pred != y) / w.sum()
        alpha = (np.log(1 - eps) - np.log(eps)) / 2
        # normalizing the weights
        for i in range(len(trn_pred)):
            if trn_pred[i] == y[i]:
                w[i] = w[i] * np.exp(- alpha)
            else:
                w[i] = w[i] * np.exp(alpha)
        # renormalizing over the distribtribution using a normalization factor
        w = w * np.exp(alpha)
        w = w / w.sum()
        # adding weighted pair to the hypothesis
        hypotheses[t] = (alpha, tree)

    return hypotheses

def predict_bagging_example(x, h_ens):
    """  predicting over ensemble of weighted hypotheses for bagging """
    arr = []
    for y in h_ens:
    	# calls predict example repeatedly and stores them in an array
        tst_pred = predict_example(x, h_ens[y][1])
        arr.append(tst_pred)
    # returning the maximum voted
    predict_egz = max(set(arr), key=arr.count)
    return predict_egz

def predict_boosting_example(x, h_ens):
    """  predicting over ensemble of weighted hypotheses for boosting """

    arr = []
    sum_alpha = 0

    for y in h_ens:
        # splitting hypothesis, weight pairs
        alpha, tree = h_ens[y]
        tst_pred = predict_example(x, tree)
        # appending prediction
        arr.append(tst_pred*alpha)
        sum_alpha += alpha
    predict_egz = np.sum(arr) / sum_alpha
    # weak learner
    if predict_egz >= 0.5:
        return 1
    else:
        return 0

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    # generatng a list of keys of the tree
    keys = list(tree.keys())
    for key in keys:
    	# seperating our attribute, value and prediction - true/false
        attribute, value, pred = key
        for i in range(0, len(x)):
            # checking if attribute matches
            if i == attribute:
                # checking if value matches
                if x[i] == value:
                    # descend if subtree
                    if  isinstance(tree[key],dict):
                        return predict_example(x, tree[key])
                    else:
                        return tree[key]
                else:
                    # else classify false preds
                    elsekey = (attribute, value, 'False')
                    # descend if subtree
                    if isinstance(tree[elsekey],dict):
                        return predict_example(x, tree[elsekey])
                    else:
                        return tree[elsekey]

def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    length = len(y_true)

    error_cnt = 0

    for i in range (length):
        if y_true[i] != y_pred[i]:
            error_cnt = error_cnt+1
    error = (1/length) * error_cnt
    return error


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

if __name__ == '__main__':

    """ 
    Discussion : Comparing the quality of my naïve implementation to Scikit
    It is evidnet that scikit's performance is much better and robust. 
    It computes significantly faster and has better prediction implying better learning ability.
    As Professor discussed in class it is always better to use well tested and 
    implemented algorithms as such instead of implemeting our own solutions."""
    


    # Load a data set
    data = DataSet('mushroom')
    
    # loading train and test sets
    X_train = data.examples['train']
    Y_train = data.labels['train']
    X_test = data.examples['test']
    Y_test = data.labels['test']

    # bagging
   	
   	# depth d
    max_depth = 3
    # bag size 
    num_trees = 10
    # returns the ensemble of weighted hypotheses.[(alpha i, h i)]
    h_ens_bag = bagging(X_train, Y_train, max_depth,num_trees)
    # sending the ensemble to predict_egzample where each pair is sent to predict_example function
    trn_pred_bag = [predict_bagging_example(X_train[i, :], h_ens_bag) for i in range(data.num_train)]
    # Compute the training error
    trn_err_bag = compute_error(Y_train, trn_pred_bag)
    tst_pred_bag = [predict_bagging_example(X_test[i, :], h_ens_bag) for i in range(data.num_test)]
    # Compute the test error
    tst_err_bag = compute_error(Y_test, tst_pred_bag)
    print("bagging results")
    print('trn={0}, tst={1}'.format(trn_err_bag, tst_err_bag))
    

    # actual = np.array(data.labels['test'])
    # predicted = np.array(tst_pred_bag)

    # # calculate the confusion matrix; labels is numpy array of classification labels
    # cm = np.zeros((2, 2))
    # for a, p in zip(actual, predicted):
    #     if a == 1 and p == 1:
    #         cm[0][0] += 1
    #     if a == 1 and p == 0:
    #         cm[0][1] += 1
    #     if a == 0 and p == 1:
    #         cm[1][0] += 1
    #     if a == 0 and p == 0:
    #         cm[1][1] += 1
    # print ("confusion matrice for depth:"+ str(max_depth) + "and bag size: " + str(num_trees))
    # print(cm)

    #boosting

    # depth d
    max_depth = 1
    # number of stumps 
    num_stumps = 20
    # returns the ensemble of weighted hypotheses.[(alpha i, h i)]
    h_ens_boost = boosting(X_train, Y_train,max_depth,num_stumps)
    # sending the ensemble to predict_egzample where each pair is sent to predict_example function
    trn_pred_boost = [predict_boosting_example(X_train[i, :], h_ens_boost) for i in range(data.num_train)]
    # Compute the training error
    trn_err_boost = compute_error(Y_train, trn_pred_boost)
    tst_pred_boost = [predict_boosting_example(X_test[i, :], h_ens_boost) for i in range(data.num_test)]
    # Compute the test error
    tst_err_boost = compute_error(Y_test, tst_pred_boost)
    
    print("boosting results")
    print('trn={0}, tst={1}'.format(trn_err_boost, tst_err_boost))

    # actual = np.array(data.labels['test'])
    # predicted = np.array(tst_pred_boost)

    # # calculate the confusion matrix; labels is numpy array of classification labels
    # cm = np.zeros((2, 2))
    # for a, p in zip(actual, predicted):
    #     if a == 1 and p == 1:
    #         cm[0][0] += 1
    #     if a == 1 and p == 0:
    #         cm[0][1] += 1
    #     if a == 0 and p == 1:
    #         cm[1][0] += 1
    #     if a == 0 and p == 0:
    #         cm[1][1] += 1
    # print ("confusion matrice for depth:"+ str(max_depth) + "and stumps: " + str(num_stumps))
    # print(cm)

