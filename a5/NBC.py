import numpy as np

"""
uses a dict on the form 
{label:{possible x value per index:[probability of value per index]}}
"""

class NBC:
    def __init__(self):
        self.features_by_label = {}
        self.p_label = {}
        self.available_labels = set()
        self.xn_if_yi_dict = {}
        self.probability_of_label = {}
        self.predictions = []

    """
    probability of a label in the dataset
    """

    def label_probability_calc(self, labels):
        total_number = len(labels)
        for l1 in self.available_labels:
            number_of_label = 0
            for l2 in labels:
                if l2 == l1:
                    number_of_label += 1
            self.p_label[l1] = number_of_label/total_number

    """
    splits the dataset based on label, puts into dict with label as key
    and all features belonging to it as value
    """

    def split_features_into_labels(self, train_features, train_labels):
        for f, l in zip(train_features, train_labels):
            if l in self.features_by_label.keys():
                self.features_by_label[l].append(f)
            else:
                self.features_by_label[l] = []
                self.features_by_label[l].append(f)
    
    """
    dis clusterfuck right here tho
    """
    def probability_by_column_per_label(self):
        x_available_in_column = set()
        for k, v in self.features_by_label.items():
            v_n = np.array(v)
            for i in range(v_n.shape[1]):
                x_available_in_column.update(set(v_n[:,i]))
        
        for k, v in self.features_by_label.items():
            # setup structure for later use, easier this way
            self.xn_if_yi_dict[k] = {}
            for x in x_available_in_column:
                self.xn_if_yi_dict[k][x] = list()
            v_n = np.array(v)
            num_arrays = v_n.shape[0]
            for index in range(v_n.shape[1]):
                for x in x_available_in_column:
                    num_of_x = 0
                    for x_i in v_n[:,index]:
                        if x_i == x:
                            num_of_x += 1
                    self.xn_if_yi_dict[k][x].append((num_of_x/num_arrays)+0.0001)

        return

    def fit(self, train_features, train_labels):
        self.available_labels = set(train_labels)
        self.label_probability_calc(train_labels)
        self.split_features_into_labels(train_features, train_labels)
        self.probability_by_column_per_label()
        return

    def predict(self, test_features):
        for feature in test_features:
            for label in self.available_labels:
                prob_label = 1
                for index in range(len(feature)):
                    prob_label *= self.xn_if_yi_dict[label][feature[index]][index]
                self.probability_of_label[label] = prob_label
            self.predictions.append(max(self.probability_of_label, key=lambda k: self.probability_of_label[k]))

        return self.predictions