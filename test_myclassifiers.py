from mysklearn.myclassifiers import MyDecisionTreeClassifier
import numpy as np
from sklearn.linear_model import LinearRegression

from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier, MyNaiveBayesClassifier
from mysklearn.mypytable import MyPyTable

# note: order is actual/received student value, expected/solution


def test_naive_bayes_classifier_fit():
    my_nb_classifier = MyNaiveBayesClassifier()
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]

    X_train_inclass_example = [
        [1, 5],  # yes
        [2, 6],  # yes
        [1, 5],  # no
        [1, 5],  # no
        [1, 6],  # yes
        [2, 6],  # no
        [1, 5],  # yes
        [1, 6]  # yes
    ]

    y_train_inclass_example = ["yes", "yes",
                               "no", "no", "yes", "no", "yes", "yes"]

    my_nb_classifier.fit(X_train_inclass_example, y_train_inclass_example)
    assert np.isclose(my_nb_classifier.priors["no"], 3 / 8)
    assert np.isclose(my_nb_classifier.priors["yes"],  5 / 8)

    assert my_nb_classifier.posteriors["no"][0][1] == 2 / 3
    assert my_nb_classifier.posteriors["yes"][1][6] == 3 / 5

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status",
                        "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]

    iphone_table = MyPyTable(iphone_col_names, iphone_table)
    buys_iphone_results = iphone_table.pop_column("buys_iphone")
    my_nb_classifier.fit(iphone_table.data, buys_iphone_results)

    assert np.isclose(my_nb_classifier.priors["no"], 1 / 3)
    assert np.isclose(my_nb_classifier.priors["yes"], 2 / 3)

    assert my_nb_classifier.posteriors["no"] == {0: {1: .6, 2: .4}, 1: {
        1: .2, 2: .4, 3: .4}, 2: {'excellent': .6, 'fair': .4}}
    assert my_nb_classifier.posteriors["yes"] == {0: {1: .2, 2: .8}, 1: {
        1: .3, 2: .4, 3: .3}, 2: {'excellent': .3, 'fair': .7}}

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]

    train_table = MyPyTable(train_col_names, train_table)
    y_labels = train_table.pop_column("class")
    my_nb_classifier.fit(train_table.data, y_labels)

    priors = my_nb_classifier.priors
    posteriors = my_nb_classifier.posteriors

    assert np.isclose(priors["on time"], 14 / 20)
    assert np.isclose(priors["cancelled"], 1 / 20)

    assert np.isclose(posteriors["on time"][0]["weekday"], 9 / 14)
    assert np.isclose(posteriors["very late"][2]["normal"], 2 / 3)


def test_naive_bayes_classifier_predict():
    my_nb_classifier = MyNaiveBayesClassifier()
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5],  # yes
        [2, 6],  # yes
        [1, 5],  # no
        [1, 5],  # no
        [1, 6],  # yes
        [2, 6],  # no
        [1, 5],  # yes
        [1, 6]  # yes
    ]

    y_train_inclass_example = ["yes", "yes",
                               "no", "no", "yes", "no", "yes", "yes"]

    my_nb_classifier.fit(X_train_inclass_example, y_train_inclass_example)
    y_predicted = my_nb_classifier.predict([[2, 5]])

    assert y_predicted == ["yes"]

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status",
                        "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]

    iphone_table = MyPyTable(iphone_col_names, iphone_table)
    buys_iphone_results = iphone_table.pop_column("buys_iphone")
    my_nb_classifier.fit(iphone_table.data, buys_iphone_results)
    result = my_nb_classifier.predict([[2, 2, "fair"]])

    assert result == ["yes"]

    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]

    train_table = MyPyTable(train_col_names, train_table)
    y_labels = train_table.pop_column("class")
    my_nb_classifier.fit(train_table.data, y_labels)
    y_predicted = my_nb_classifier.predict(
        [["weekday", "winter", "high", "heavy"]])

    assert y_predicted == ["very late"]


def test_simple_linear_regression_classifier_fit():
    np.random.seed(0)

    # test case 1: class data set
    X_train = [[val] for val in range(100)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]  # 1D
    X_test = [[150], [105], [99], [22], [0]]

    my_linear_classifier = MySimpleLinearRegressionClassifier(
        myutils.high_low_discretizer, MySimpleLinearRegressor())
    my_linear_classifier.fit(X_train, y_train)
    sk_linear_regressor = LinearRegression()
    sk_linear_regressor.fit(X_train, y_train)

    assert np.isclose(my_linear_classifier.regressor.slope,
                      sk_linear_regressor.coef_[0])
    assert np.isclose(my_linear_classifier.regressor.intercept,
                      sk_linear_regressor.intercept_)

    # test case 2: weight to shirt size
    X_train = [[val] for val in range(100, 200, 5)]
    y_train = [row[0] / 10 for row in X_train]
    X_test = [[122], [157], [200], [177]]

    my_linear_classifier.discretizer = myutils.shirt_size_discretizer
    my_linear_classifier.fit(X_train, y_train)
    sk_linear_regressor.fit(X_train, y_train)

    assert np.isclose(my_linear_classifier.regressor.slope,
                      sk_linear_regressor.coef_[0])
    assert np.isclose(my_linear_classifier.regressor.intercept,
                      sk_linear_regressor.intercept_)


def test_simple_linear_regression_classifier_predict():
    np.random.seed(0)
    # test case 1: class data set
    X_train = [[val] for val in range(100)]  # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]  # 1D
    X_test = [[150]]

    my_linear_classifier = MySimpleLinearRegressionClassifier(
        myutils.high_low_discretizer, MySimpleLinearRegressor())
    my_linear_classifier.fit(X_train, y_train)
    my_predictions = my_linear_classifier.predict(X_test)

    sk_linear_regressor = LinearRegression()
    sk_linear_regressor.fit(X_train, y_train)
    sk_predictions = sk_linear_regressor.predict(X_test)
    sk_predictions = [myutils.high_low_discretizer(
        num) for num in sk_predictions]

    assert my_predictions == sk_predictions

    # desk calculations

    # case 2:
    my_linear_classifier = MySimpleLinearRegressionClassifier(
        myutils.high_low_discretizer, MySimpleLinearRegressor(7.77, 30))

    X_test = [[1], [5], [6.6], [26], [35], [100]]
    y_correct = [37.7, 68.5, 80.82, 230.2, 299.5, 800.0]
    y_correct = [myutils.high_low_discretizer(y) for y in y_correct]
    predictions = my_linear_classifier.predict(X_test)

    assert predictions == y_correct

    # case 3:
    my_linear_classifier = MySimpleLinearRegressionClassifier(
        myutils.shirt_size_discretizer, MySimpleLinearRegressor(3, -100))

    X_test = [[199], [0], [208], [55], [40], [500]]
    y_correct = [497, -100, 524, 65, 20, 1400]
    y_correct = [myutils.shirt_size_discretizer(y) for y in y_correct]
    predictions = my_linear_classifier.predict(X_test)

    assert predictions == y_correct


def test_kneighbors_classifier_kneighbors():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test = [[.33, 1]]

    my_kn_classifier = MyKNeighborsClassifier()
    my_kn_classifier.fit(X_train_class_example1, y_train_class_example1)

    actual_distances = [[.67, 1.0, 1.05304]]
    actual_kneighbors = [[0, 2, 3]]
    my_distances, my_kneighbors = my_kn_classifier.kneighbors(X_test)

    assert np.allclose(my_kneighbors, actual_kneighbors)
    assert np.allclose(my_distances, actual_distances)

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes",
                              "no", "no", "yes", "no", "yes", "yes"]
    X_test = [[2, 3]]

    my_kn_classifier.fit(X_train_class_example2, y_train_class_example2)

    sk_distances, sk_kneighbors = ([1.41421356, 1.4142121356, 2.0], [0, 4, 6])
    my_distances, my_kneighbors = my_kn_classifier.kneighbors(X_test)

    assert np.allclose(my_kneighbors, sk_kneighbors)
    assert np.allclose(my_distances, sk_distances)

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
                              "-", "-", "+", "+", "+", "-", "+"]

    X_test = [[9.1, 11.0]]

    my_kn_classifier.fit(X_train_bramer_example, y_train_bramer_example)

    actual_distances, actual_kneighbors = ([.608, 1.237, 2.202], [6, 5, 7])
    my_distances, my_kneighbors = my_kn_classifier.kneighbors(X_test)

    assert np.allclose(my_kneighbors, actual_kneighbors)
    assert np.allclose(my_distances, actual_distances, .001)


def test_kneighbors_classifier_predict():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test = [[.33, 1]]

    my_kn_classifier = MyKNeighborsClassifier()
    my_kn_classifier.fit(X_train_class_example1, y_train_class_example1)
    my_prediction = my_kn_classifier.predict(X_test)

    assert my_prediction == ["good"]

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes",
                              "no", "no", "yes", "no", "yes", "yes"]
    X_test = [[2, 3]]

    my_kn_classifier.fit(X_train_class_example2, y_train_class_example2)
    prediction = my_kn_classifier.predict(X_test)

    assert prediction == ["yes"]

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",
                              "-", "-", "+", "+", "+", "-", "+"]

    X_test = [[9.1, 11.0]]

    my_kn_classifier.fit(X_train_bramer_example, y_train_bramer_example)
    prediction = my_kn_classifier.predict(X_test)

    assert prediction == ["+"]


def test_dummy_classifier_fit():
    X_train = list(np.arange(100))
    y_train = list(np.random.choice(
        ["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_test = [[10], [100], [1]]
    my_dummy_classifier = MyDummyClassifier()
    my_dummy_classifier.fit(X_train, y_train)

    assert my_dummy_classifier.most_common_label == "yes"

    X_train = list(np.arange(100))
    y_train = list(np.random.choice(
        ["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_test = [[10], [100], [1]]

    my_dummy_classifier = MyDummyClassifier()
    my_dummy_classifier.fit(X_train, y_train)

    assert my_dummy_classifier.most_common_label == "no"


def test_dummy_classifier_predict():
    X_train = list(np.arange(100))
    y_train = list(np.random.choice(
        ["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_test = [[10], [100], [1]]
    my_dummy_classifier = MyDummyClassifier()
    my_dummy_classifier.fit(X_train, y_train)
    my_prediction = my_dummy_classifier.predict(X_test)

    assert my_prediction == ["yes", "yes", "yes"]

    X_train = list(np.arange(100))
    y_train = list(np.random.choice(
        ["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_test = [[10], [100], [1]]

    my_dummy_classifier = MyDummyClassifier()
    my_dummy_classifier.fit(X_train, y_train)
    my_prediction = my_dummy_classifier.predict(X_test)

    assert my_prediction == ["no", "no", "no"]


# interview dataset
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]

X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False",
                     "True", "False", "True", "True", "True", "True", "True", "False"]

tree_interview = \
    ["Attribute", "att0",
     ["Value", "Junior",
      ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "True", 3, 5]
                     ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                     ]
       ]
      ],
     ["Value", "Mid",
      ["Leaf", "True", 4, 14]
      ],
     ["Value", "Senior",
      ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                     ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                     ]
       ]
      ]
     ]

# bramer degrees dataset
header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
X_train_degrees = [
    ['A', 'B', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'A', 'B', 'B'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'A', 'B', 'B', 'A'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B'],
    ['A', 'A', 'A', 'A', 'A'],
    ['B', 'A', 'A', 'B', 'B'],
    ['B', 'A', 'A', 'B', 'B'],
    ['A', 'B', 'B', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'B', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['A', 'A', 'B', 'B', 'B'],
    ['B', 'B', 'B', 'B', 'B'],
    ['A', 'A', 'B', 'A', 'A'],
    ['B', 'B', 'B', 'A', 'A'],
    ['B', 'B', 'A', 'A', 'B'],
    ['B', 'B', 'B', 'B', 'A'],
    ['B', 'A', 'B', 'A', 'B'],
    ['A', 'B', 'B', 'B', 'A'],
    ['A', 'B', 'A', 'B', 'B'],
    ['B', 'A', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B']
]
y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                   'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                   'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                   'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                   'SECOND', 'SECOND']

tree_degrees = ['Attribute', 'att0',
                ['Value', 'A',
                 ['Attribute', 'att4',
                  ['Value', 'A',
                   ['Leaf', 'FIRST', 5, 14]],
                  ['Value', 'B',
                   ['Attribute', 'att3',
                    ['Value', 'A',
                            ['Attribute', 'att1',
                             ['Value', 'A',
                              ['Leaf', 'FIRST', 1, 2]],
                                ['Value', 'B',
                                 ['Leaf', 'SECOND', 1, 2]]]],
                    ['Value', 'B',
                     ['Leaf', 'SECOND', 7, 9]]]]]],
                ['Value', 'B',
                 ['Leaf', 'SECOND', 12, 26]]]

# RQ5 (iphone) dataset
iphone_col_names = ["standing", "job_status",
                    "credit_rating", "buys_iphone"]
iphone_table = [
    [1, 3, "fair", "no"],
    [1, 3, "excellent", "no"],
    [2, 3, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [2, 1, "fair", "yes"],
    [2, 1, "excellent", "no"],
    [2, 1, "excellent", "yes"],
    [1, 2, "fair", "no"],
    [1, 1, "fair", "yes"],
    [2, 2, "fair", "yes"],
    [1, 2, "excellent", "yes"],
    [2, 2, "excellent", "yes"],
    [2, 3, "fair", "yes"],
    [2, 2, "excellent", "no"],
    [2, 3, "fair", "yes"]
]


# decision tree classifiers
interview_tree_classifier = MyDecisionTreeClassifier()
bramer_tree_classifier = MyDecisionTreeClassifier()
iphone_tree_classifier = MyDecisionTreeClassifier()


def test_decision_tree_classifier_fit():

    interview_tree_classifier.fit(X_train_interview, y_train_interview)
    assert interview_tree_classifier.tree == tree_interview

    bramer_tree_classifier.fit(X_train_degrees, y_train_degrees)
    assert bramer_tree_classifier.tree == tree_degrees

    iphone_mypytable = MyPyTable(iphone_col_names, iphone_table)
    y_train_iphone = iphone_mypytable.pop_column("buys_iphone")
    X_train_iphone = iphone_mypytable.data

    iphone_tree_classifier.fit(X_train_iphone, y_train_iphone)
    print(iphone_tree_classifier.tree)

    assert iphone_tree_classifier.tree == ['Attribute', 'att0',
                                           ['Value', 1,
                                            ['Attribute', 'att1',
                                             ['Value', 1,
                                                        ['Leaf', 'yes', 1, 5]],
                                             ['Value', 2,
                                              ['Attribute', 'att2',
                                                            ['Value', 'excellent',
                                                                ['Leaf', 'yes', 1, 2]],
                                                            ['Value', 'fair',
                                                                ['Leaf', 'no', 1, 2]]]],
                                             ['Value', 3,
                                              ['Leaf', 'no', 2, 5]]]],
                                           ['Value', 2,
                                            ['Attribute', 'att2',
                                             ['Value', 'excellent',
                                              ['Leaf', 'no']],
                                             ['Value', 'fair',
                                              ['Leaf', 'yes', 6, 10]]]]]


def test_decision_tree_classifier_predict():
    interview_preds = interview_tree_classifier.predict(
        [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]])

    assert interview_preds == ["True", "False"]

    bramer_preds = bramer_tree_classifier.predict(
        [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]])

    assert bramer_preds == ["SECOND", "FIRST", "FIRST"]

    iphone_preds = iphone_tree_classifier.predict(
        [[2, 2, "fair"], [1, 1, "excellent"]])
    assert iphone_preds == ["yes", "yes"]
