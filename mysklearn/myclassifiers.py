import operator
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

import copy
from mysklearn import myutils
from mysklearn.mypytable import MyPyTable




class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.calculate_priors(y_train)  # priors are stored as a nested dict
        self.calculate_posteriors(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        all_class_labels = list(self.posteriors.keys())
        num_cols = len(X_test[0])

        for test_instance in X_test: 
            highest_prob_class = all_class_labels[0]
            highest_prob = 0
            for class_label in all_class_labels:
                class_probability = 1
                for i in range(num_cols):
                    try:
                        class_probability *= self.posteriors[class_label][i][test_instance[i]]
                    except KeyError:
                        class_probability = 0
                class_probability *= self.priors[class_label]
                if class_probability > highest_prob:
                    highest_prob_class = class_label
            
            y_predicted.append(class_label)
                
        return y_predicted

    def calculate_priors(self, y_train):
        labels = copy.copy(y_train)
        priors_dict = {}
        num_instances = len(y_train)
        values, counts = myutils.get_frequencies(labels)

        for i in range(len(values)):
            value = values[i]
            freq = counts[i]
            prior = freq / num_instances
            priors_dict[value] = prior

        self.priors = priors_dict

    def calculate_posteriors(self, X_train, y_train):
        labels = copy.copy(y_train)
        posteriors_dict = {}

        class_labels, label_freq = myutils.get_frequencies(labels)
        groups_dict = myutils.group_by_label(X_train, y_train)
        

        for i in range(len(class_labels)):
            label = class_labels[i]
            freq = label_freq[i]

            posteriors_dict[label] = freq
            denominator = freq 

            class_instances = groups_dict[label]

            col_ct = 0
            inner_dict = {}
            for i in range(len(class_instances[0])):
                col = MyPyTable([x for x in range(len(class_instances[0]))], class_instances).get_column(i)

                attribute_values, value_freqs = myutils.get_frequencies(copy.copy(col))
                innermost_dict = {}
                for i, attribute_value in enumerate(attribute_values):
                    innermost_dict[attribute_value] = value_freqs[i] / denominator

                inner_dict[col_ct] = innermost_dict
                col_ct += 1

            posteriors_dict[label] = inner_dict

        self.posteriors = posteriors_dict


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).
    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data
    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.
        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if not self.regressor:
            self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = self.regressor.predict(X_test)
        return [self.discretizer(num) for num in y_predicted]


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distance = []
        indices = []
        for test_instance in X_test:
            row_indexes_dists = []
            distance.append([])
            indices.append([])

            for i, train_instance in enumerate(self.X_train):
                if isinstance(train_instance[0], str):
                    dist = myutils.compute_catagorical_distance(train_instance, test_instance)
                else:
                    dist = myutils.compute_euclidean_distance(
                        train_instance, test_instance)
                row_indexes_dists.append([i, round(dist, 8)])

            row_indexes_dists.sort(key=operator.itemgetter(-1))

            top_k = row_indexes_dists[:self.n_neighbors]

            for pair in top_k:
                distance[-1].append(pair[1])
                indices[-1].append(pair[0])

        return distance, indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        distances, kneighbors = self.kneighbors(X_test)

        for test_instance_kneighbors in kneighbors:
            y_labels = [self.y_train[num] for num in test_instance_kneighbors]
            values, counts = myutils.get_frequencies(y_labels)
            prediction = myutils.majority_vote_winner(values, counts)
            y_predicted.append(prediction)

        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """

        values, count = myutils.get_frequencies(y_train)
        self.most_common_label = myutils.majority_vote_winner(values, count)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for test_instance in X_test]
            