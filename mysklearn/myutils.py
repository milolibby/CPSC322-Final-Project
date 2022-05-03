import numpy as np
import mysklearn.myevaluation as myevaluation
import math

import random

def random_forest_split_test_and_training_sets_stratified(X, y):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    partition_indexes = group_by_class_label(y)

    for group in partition_indexes:
        train_idxs = group[:(len(group) * 2) // 3]
        test_idxs = group[(len(group) * 2) // 3:]

        for idx in train_idxs:
            X_train.append(X[idx])
            y_train.append(y[idx])

        for idx in test_idxs:
            X_test.append(X[idx])
            y_test.append(y[idx])

    return X_train, X_test, y_train, y_test



def random_attribute_subset(attributes, F):
    # shuffle and pick first F
    shuffled = attributes[:]  # make a copy
    random.shuffle(shuffled)
    return shuffled[:F]


def randomize_in_place(alist, parallel_list=None, random_state=0):
    np.random.seed(random_state)
    for i in range(len(alist)):
        # generate a random index to swap values with
        rand_index = np.random.randint(0, len(alist))  # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]


def ratings_discretizer(num):
    if num > 79:
        return "special"
    return "basic"


def reb_discretizer(num):
    if num > 9.0:
        return "high volume"
    if num > 4.0:
        return "average volume"
    return "low volume"


def fg_discretizer(num):
    if num > 45:
        return "efficient"
    return "inefficient"


def win_percent_discretizer(num):
    if num > .7:
        return "high"
    elif num > .5:
        return "average"
    else:
        return "low"


def pts_discretizer(num):
    if num > 21:
        return "top"
    elif num > 12:
        return "mid"
    return "bottom"


def threePTRS_made_discretizer(num):
    if num > 2.0:
        return "high volume"
    return "low volume"


def mins_played_discretizer(mins):
    if mins > 30:
        return "high"
    elif mins > 15:
        return "moderate"
    else:
        return "low"


def group_by_class_label(labels):
    partitions = []
    partitions_indexs = []

    for label_index, label in enumerate(labels):
        new = True
        if partitions == []:
            partitions.append([label])
            partitions_indexs.append([label_index])
        else:
            for group_num, group in enumerate(partitions):
                if label == group[0]:
                    partitions[group_num].append(label)
                    partitions_indexs[group_num].append(label_index)
                    new = False
                    break

            if new:
                partitions.append([label])
                partitions_indexs.append([label_index])

    return partitions_indexs


def high_low_discretizer(num):
    if num >= 100:
        return "high"
    return "low"


def shirt_size_discretizer(num):
    if num > 17:
        return "L"
    elif num > 14:
        return "M"
    else:
        return "S"


def free_throw_discretizer(num):
    if num > 90:
        return "great"
    elif num > 75:
        return "decent"
    else:
        return "poor"


"""
  Instances (2D list)
  labels (1D list)

  returns a dictionary that maps the labels to the instances with the labels
"""


def group_by_label(Instances, labels):
    values = []
    groups_dict = {}

    for i, label in enumerate(labels):
        instance = Instances[i]
        if label not in values:
            groups_dict[label] = [instance]
            values.append(label)
        else:
            groups_dict[label].append(instance)

    return groups_dict


def age_discretizer(age):
    if age > 28:
        return "old"
    return "young"


def majority_vote_winner(values, frequencies):
    max_frequency = max(frequencies)
    max_index = frequencies.index(max_frequency)

    return values[max_index]


def compute_euclidean_distance(v1, v2):
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))


def compute_catagorical_distance(v1, v2):
    total = 0
    for i in range(len(v1)):
        if v1[i] is not v2[i]:
            total += 1
    return total


def get_frequencies(col):
    col.sort()  # inplace
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values:  # seen it before
            counts[-1] += 1  # okay because sorted
        else:  # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts  # we can return multiple values in python


def determineMPGRating(mpg):
    if mpg >= 45:
        return 10
    elif mpg >= 37:
        return 9
    elif mpg >= 31:
        return 8
    elif mpg >= 27:
        return 7
    elif mpg >= 24:
        return 6
    elif mpg >= 20:
        return 5
    elif mpg >= 17:
        return 4
    elif mpg >= 15:
        return 3
    elif mpg >= 14:
        return 2
    else:
        return 1


def min_max_normalize(xs):
    normalized = []
    for x in xs:
        normalized.append((x - min(xs)) / ((max(xs) - min(xs)) * 1.0))
    return normalized


"""
Takes in train and test folds and a classifier
returns accuracy, error, precision, recall, f1, confusion matrix
"""


def get_scores_from_folds(X, y, X_train_folds, X_test_folds, classifier):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    y_true_all = []
    predictions_all = []
    for i in range(len(X_train_folds)):
        X_train = [X[train_index] for train_index in X_train_folds[i]]
        y_train = [y[train_index] for train_index in X_train_folds[i]]

        X_test = [X[test_index] for test_index in X_test_folds[i]]
        y_true = [y[test_index] for test_index in X_test_folds[i]]

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        y_true_all += y_true
        predictions_all += predictions

        precision_scores.append(myevaluation.binary_precision_score(
            y_true, predictions, ["yes", "no"], "yes"))
        recall_scores.append(myevaluation.binary_recall_score(
            y_true, predictions, ["yes", "no"], "yes"))
        f1_scores.append(myevaluation.binary_f1_score(
            y_true, predictions, ["yes", "no"], "yes"))

    accuracy = round(myevaluation.accuracy_score(
        y_true_all, predictions_all), 2)
    error = round(1 - accuracy, 2)
    precision = round(np.average(precision_scores), 2)
    recall = round(np.average(recall_scores), 2)
    f1 = round(np.average(f1_scores), 2)

    matrix = myevaluation.confusion_matrix(
        y_true_all, predictions_all, ["yes", "no"])

    return accuracy, error, precision, recall, f1, matrix


#used in Plot_Utils
def convert_list_to_dict(lst):
    """
    Accepts a list of values and converts it into a dictionary which 
    records all possible values along with it's frequencies
    """
    dictionary = {}
    for value in lst:
        if value in dictionary:
            dictionary[value] += 1
        else:
            dictionary[value] = 1

    return dictionary


def get_e(labels, total_instances):
    e = []
    values, counts = get_frequencies(labels)
    for ct in counts:
        pi = ct / len(labels)
        e.append(-(pi * math.log(pi, 2)))

    return sum(e) * (len(labels) / total_instances)


def get_enew(partitions, total_instances):
    enew = []
    for group in list(partitions.values()):
        labels = [row[-1] for row in group]
        e = get_e(labels, total_instances)
        enew.append(e)
    return sum(enew)


def select_attribute(instances, attributes, attribute_domains, header):
    # with the smallest Enew
    best_attribute = None
    smallest_enew = 2
    for attribute in attributes:
        partition = partition_instances(
            instances, attribute, attribute_domains, header)
        enew = get_enew(partition, len(instances))
        if enew < smallest_enew:
            best_attribute = attribute
            smallest_enew = enew

    return best_attribute


def all_same_class(att_partition):
    class_value = att_partition[0][-1]
    for row in att_partition:
        if row[-1] is not class_value:
            return False
    return True


def partition_instances(instances, split_attribute, attribute_domains, header):
    # lets use a dictionary
    partitions = {}  # key (string): value (subtable)
    att_index = header.index(split_attribute)  # e.g. 0 for level
    # e.g. ["Junior", "Mid", "Senior"]
    att_domain = attribute_domains[att_index]
    for att_value in att_domain:
        partitions[att_value] = []
        # task: finish
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions


def majority_vote_winner(values, frequencies):
    max_frequency = max(frequencies)
    max_index = frequencies.index(max_frequency)

    return values[max_index]


def total_instances_in_partition(partitions):
    total = 0
    for key in partitions.keys():
        total += len(partitions[key])
    return total


def determine_majority_vote_winner_dict(partitions):
    partitions = list(partitions.values())
    all_instances = []

    for group in partitions:
        if group == []:
            continue
        for instance in group:
            all_instances.append(instance[-1])

    values, counts = get_frequencies(all_instances)

    winner = majority_vote_winner(values, counts)

    return winner