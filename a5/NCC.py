import numpy as np
from scipy.spatial import distance


class NCC:
    
    def __init__(self):
        self.some_data_structure = {}
        self.some_predictions = []
        self.centroid_dict = {}
    
    def fit(self, train_features, train_labels):

        for feature, label in zip(train_features, train_labels):
            if label in self.some_data_structure.keys():
                self.some_data_structure[label].append(feature)
            else:                
                self.some_data_structure[label] = []
                self.some_data_structure[label].append(feature)

        for label in sorted(self.some_data_structure.keys()):
            array = np.array(self.some_data_structure[label])
            self.centroid_dict[label] = array.mean(axis=0)

    def get_closest_centroid(self, test_feature):
        current_min = ('', float('inf'))
        for label in sorted(self.centroid_dict.keys()):
            d = distance.euclidean(self.centroid_dict[label], test_feature)
            if d < current_min[1]:
                current_min = (label, d)

        return current_min

    def predict(self, test_features):
        for feature in test_features:
            prediction = self.get_closest_centroid(feature)
            self.some_predictions.append(prediction[0])

        return self.some_predictions

