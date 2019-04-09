# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu).
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package.


import numpy as np


# A simple utility class to load data sets for this assignment
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
                                          usecols=(0, 1, 2, 3, 4, 5, 6), dtype=int)

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

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """


    elements,counts = np.unique(y,return_counts = True)
    entropy = 0
    for i in range(len(elements)):
        entropy += np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))])
    return entropy


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    total_entropy = entropy(y)

    partitioned_x = partition(x)

    vals,counts= np.unique(x,return_counts=True)
    weighted_entropy = 0
    
    for key in partitioned_x:
        weighted_entropy += np.sum(len(partitioned_x[key]/np.sum(counts)) * entropy(y[partitioned_x[key]]))

    information_gain = total_entropy - weighted_entropy

    return information_gain


def id3(x, y, attributes, max_depth, depth=0):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of attributes
    to consider. This is a recursive algorithm that depends on three termination conditions
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

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1): 1,
     (4, 2): {(3, 1): 0,
              (3, 2): 0,
              (3, 3): 0},
     (4, 3): {(5, 1): 0,
              (5, 2): 0},
     (4, 4): {(0, 1): 0,
              (0, 2): 0,
              (0, 3): 1}}
    """

    if len(np.unique(y)) == 1:
        return 1
    elif len(np.unique(y)) == 0:
        return 0
    elif len(attributes) == 0 or depth == max_depth :
        return np.bincount(y).argmax()
    else:       
        info_values = {}
        for i in range(len(x[0])):
            info_values[i] = mutual_information(x[:,i],y)

        best_feature_index = max(info_values, key=info_values.get)
        new_indices = partition(x[:,best_feature_index])
        depth = depth + 1
        new_attr = list(filter(lambda x : x != best_feature_index, attributes))

        tree = {}
        for i in new_indices:
            newX = x[new_indices[i]]
            newY = y[new_indices[i]]
            tree[(best_feature_index,i)] = id3(newX,newY,new_attr,max_depth,depth)
                    
    return(tree) 


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    for key, subtree in tree.items():
        attribute, value =  key
        for i in range(len(x)):
            if i == attribute and x[i] == value:
                if isinstance(subtree, dict):
                    return predict_example(x,subtree)
                else:
                    return subtree



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
    #
    # Below is an example of how a decision tree can be trained and tested on a data set in the folder './data/'. Modify
    # this function appropriately to answer various questions from the Programming Assignment.
    #

    # Load a data set
    data = DataSet('monks-1')

    # Get a list of all the attribute indices
    attribute_idx = np.array(range(data.dim))

    # Learn a decision tree of depth 3
    d = 3
    decision_tree = id3(data.examples['train'], data.labels['train'], attribute_idx, d)
    visualize(decision_tree)

    # Compute the training error
    trn_pred = [predict_example(data.examples['train'][i, :], decision_tree) for i in range(data.num_train)]
    trn_err = compute_error(data.labels['train'], trn_pred)

    # Compute the test error
    tst_pred = [predict_example(data.examples['test'][i, :], decision_tree) for i in range(data.num_test)]
    tst_err = compute_error(data.labels['test'], tst_pred)

    # Print these results
    print('d={0} trn={1}, tst={2}'.format(d, trn_err, tst_err))