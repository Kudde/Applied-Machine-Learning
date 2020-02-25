import ToyData as td
import ID3

from sklearn import datasets , svm, metrics
import matplotlib.pyplot as plt
from sklearn import datasets , svm, metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus

import numpy as np
from sklearn import tree, metrics, datasets


def main():

    # attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    #
    # print("attributes")
    # print(attributes)
    # print("classes")
    # print(classes)
    # print("data")
    # print(data)
    # print("target")
    # print(target)
    #
    # id3 = ID3.ID3DecisionTreeClassifier()
    #
    # myTree = id3.fit(data, target, attributes, classes)
    # print(myTree)
    # plot = id3.make_dot_data()
    # plot.render("testTree")
    # predicted = id3.predict(data2, myTree, attributes)
    # print("Predicted Result :")
    # print(predicted)
    #
    # print("Target Result : ")
    # print(target2)

    # ---------------------------------------
    # print("------- Images")
    # digits = datasets.load_digits()
    # num_examples = len(digits.data)
    # num_split = int(0.7*num_examples)
    #
    # print("data pre")
    # print(digits.data[2])
    # print("target pre")
    # print(digits.target[:num_split])
    # print("Shape")
    # print(len(digits.data[-1]))
    #
    # attributes = {}
    # for i in range(0, 64):
    #     attributes[i] = list([x / 1.0 for x in range(0, 17, 1)])
    #
    # classes = tuple(range(0, 10))
    #
    # train_features = []
    # for sample in digits.data[:num_split]:
    #     train_features.append(tuple(sample))
    #
    # train_labels = tuple(digits.target[:num_split])
    #
    # print("attributes")
    # print(attributes)
    # print("classes")
    # print(classes)
    # print("data")
    # print(train_features)
    # print("target")
    # print(train_labels)
    #
    # test_features = digits.data[num_split:]
    # test_labels = digits.target[num_split:]
    #
    # id3 = ID3.ID3DecisionTreeClassifier()
    # myTree = id3.fit(train_features, train_labels, attributes, classes)
    # plot = id3.make_dot_data()
    # plot.render("digitTree")
    # predicted = id3.predict(data2, myTree, attributes)
    # print("Predicted Result :")
    # print(predicted)
    #
    # print("Target Result : ")
    # print(target2)

    # --------------------------------------- light , grey, dark
    print("------- Images - Hue")
    digits = datasets.load_digits()
    num_examples = len(digits.data)

    num_split = int(0.7*num_examples)

    attributes = {}
    for i in range(0, 64):
        attributes[i] = ['light', 'grey', 'dark']

    classes = tuple(range(0, 10))
    # classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


    train_features = []
    for sample in digits.data[:num_split]:
        hue_list = []
        for point in sample:
            if point < 5 :
                hue_list.append('light')
            elif point > 10 :
                hue_list.append('dark')
            else :
                hue_list.append('grey')

        train_features.append(tuple(hue_list))

    train_labels = tuple(digits.target[:num_split])

    print("attributes")
    print(attributes)
    print("classes")
    print(classes)
    print("data")
    print(train_features)
    print("target")
    print(train_labels)

    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]

    test_features_formated = []
    for sample in test_features:
        hue_list = []
        for point in sample:
            if point < 5 :
                hue_list.append('light')
            elif point > 10 :
                hue_list.append('dark')
            else :
                hue_list.append('grey')

        test_features_formated.append(tuple(hue_list))

    test_labels_formated = tuple(test_labels)
    print(test_features_formated)
    print(test_labels_formated)

    id3 = ID3.ID3DecisionTreeClassifier()
    myTree = id3.fit(train_features, train_labels, attributes, classes)
    plot = id3.make_dot_data()
    plot.render("digitHueTree")

    predicted = id3.predict(test_features_formated, myTree, attributes)
    print("Predicted Result :")
    print(predicted)

    print("Target Result : ")
    print(test_labels_formated)

    print("Classification report for classifier %s:\n%s\n"
          % (myTree, metrics.classification_report(test_labels, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))


if __name__ == "__main__": main()