from cProfile import label
from mysklearn import myutils
from mysklearn.myutils import *
import numpy as np


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if shuffle:
        randomize_in_place(X, y, random_state)

    if type(test_size) == float:
        split_index = len(X) - int(test_size * len(X)) - 1

    if type(test_size) == int:
        split_index = len(X) - test_size

    return X[0:split_index], X[split_index:], y[0:split_index], y[split_index:]


def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    all_indexes = [x for x in np.arange(len(X))]

    if shuffle:
        randomize_in_place(all_indexes, random_state=random_state)

    n_samples = len(X)
    fold_size = int(n_samples / n_splits)

    extra_indexes = n_samples % n_splits
    fold_sizes = []

    for i in range(extra_indexes):
        fold_sizes.append(fold_size + 1)

    for i in range(n_splits - extra_indexes):
        fold_sizes.append(fold_size)

    index_ct = 0
    train_folds = []
    test_folds = []
    for fold_size in fold_sizes:
        train_indexes = all_indexes[:]
        fold_indexes = []
        for i in range(fold_size):
            fold_indexes.append(all_indexes[index_ct])
            train_indexes.remove(all_indexes[index_ct])
            index_ct += 1
        test_folds.append(fold_indexes)
        train_folds.append(train_indexes)

    return train_folds, test_folds


def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if shuffle:
        randomize_in_place(X, y, random_state=random_state)

    all_indexes = [x for x in np.arange(len(X))]

    n_samples = len(X)
    fold_size = int(n_samples / n_splits)

    extra_indexes = n_samples % n_splits
    fold_sizes = []

    for i in range(extra_indexes):
        fold_sizes.append(fold_size + 1)

    for i in range(n_splits - extra_indexes):
        fold_sizes.append(fold_size)

    partition_indexes = group_by_class_label(y)

    train_folds = []
    test_folds = []

    for fold_size in fold_sizes:
        test_folds.append([" " for i in range(fold_size)])
    fold_num = 0
    for group in partition_indexes:
        for label_index in group:
            x = label_index
            for i in range(fold_sizes[fold_num]):
                if test_folds[fold_num][i] == " ":
                    test_folds[fold_num][i] = x
                    break
            fold_num += 1
            if fold_num == n_splits:
                fold_num = 0

    for test_fold in test_folds:
        train_fold = []
        for index in all_indexes:
            if index not in test_fold:
                train_fold.append(index)
        train_folds.append(train_fold)

    return train_folds, test_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """

    if not random_state:
        random_state = 0
    np.random.seed(random_state)

    if not n_samples:
        n_samples = len(X)

    X_sample = []
    X_out_of_bag = X[:]

    if y:
        y_sample = []
        y_out_of_bag = y[:]
    else:
        y_sample = None
        y_out_of_bag = None

    number_of_instances = len(X)

    for i in range(n_samples):
        random_index = np.random.randint(0, number_of_instances)

        X_sample.append(X[random_index])

        if X[random_index] in X_out_of_bag:
            X_out_of_bag.remove(X[random_index])

        if y:
            y_sample.append(y[random_index])

            if y[random_index] in y_out_of_bag:
                y_out_of_bag.remove(y[random_index])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    num_instances = len(y_true)
    matrix = []

    for actual in labels:
        matrix_row = []
        for predicted in labels:
            ct = 0
            for i in range(num_instances):
                if y_true[i] == actual and y_pred[i] == predicted:
                    ct += 1
            matrix_row.append(ct)
        matrix.append(matrix_row)

    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    labels = []

    for val in y_true:
        if val not in labels:
            labels.append(val)

    matrix = confusion_matrix(y_true, y_pred, labels)

    correct_instances = []

    for i in range(len(labels)):
        correct_instances.append(matrix[i][i])

    if normalize:
        return sum(correct_instances) / len(y_pred)
    return sum(correct_instances)


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if not labels:
        labels, freq = myutils.get_frequencies(y_true)

    if not pos_label:
        pos_label = labels[0]

    TP = 0
    P = 0
    for i in range(len(y_true)):
        label = y_pred[i]
        if label == pos_label:
            P += 1
            if label == y_true[i]:
                TP += 1

    if P == 0:
        return 0

    return TP / P


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if not labels:
        labels, freq = myutils.get_frequencies(y_true)

    if not pos_label:
        pos_label = labels[0]

    TP = 0
    FN = 0

    for i in range(len(y_true)):
        label = y_pred[i]
        if label == pos_label:
            if label == y_true[i]:
                TP += 1
        else:
            if y_true[i] == pos_label:
                FN += 1

    if TP == 0:
        return 0.0
    return TP / (TP + FN)


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """

    if not labels:
        labels, freq = myutils.get_frequencies(y_true)

    if not pos_label:
        pos_label = labels[0]

    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)
