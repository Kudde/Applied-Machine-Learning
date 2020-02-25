import numpy as np


# Summaries the samples by feature
# Returns mean, std and length for each feature
def sum_features(features):
    features_sums = []
    epsilon = 0.1
    for feature in zip(*features):
        features_sums.append((np.mean(feature), np.std(feature) + epsilon, len(feature)))

    return features_sums


# Returns the Gaussian probability for x
def calc_prob(x, my, sigma):
    return (1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-((x - my) ** 2 / (2 * sigma ** 2)))


# Returns the probabilities for each label for a given vector of features
def calc_labels_prob(summaries, predict_feature_vector):
    # print(summaries[0][0])
    total_samples = 0
    for label in summaries:
        total_samples += summaries[label][0][2]

    probabilities = dict()

    for label, feature_summaries in summaries.items():
        probabilities[label] = summaries[label][0][2] / float(total_samples)

        for feature_stats, predict_vector in zip(feature_summaries, predict_feature_vector):
            mean, std, length = feature_stats
            probabilities[label] *= calc_prob(predict_vector, mean, std) # improve vanish with log stuff

    return probabilities


class GBN:

    def __init__(self):
        self.features_by_label_dict = {}
        self.feature_summaries_by_label = {}

    def fit(self, train_features, train_labels):
        samples_by_label = {}

        for feature, label in zip(train_features, train_labels):
            if label not in samples_by_label.keys():
                samples_by_label[label] = []

            samples_by_label[label].append(feature)

        # print(samples_by_label[0])
        # print(type(samples_by_label[0]))
        # print(type(samples_by_label[0][0]))

        for label, features in samples_by_label.items():
            self.feature_summaries_by_label[label] = sum_features(features)

    def predict(self, test_features):
        # print(test_features[0])
        predictions = []

        # probabilities = calc_labels_prob(self.feature_summaries_by_label, test_features[0])
        # print(probabilities)
        # print(max(probabilities, key=probabilities.get))

        for feature_vector in test_features:
            probabilities = calc_labels_prob(self.feature_summaries_by_label, feature_vector)
            predictions.append(max(probabilities, key=probabilities.get))

        return predictions
