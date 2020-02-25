from sklearn import metrics, datasets
from sklearn.naive_bayes import GaussianNB
import MNIST
from NCC import NCC
from NBC import NBC
from GBN import GBN


def main_boring_gnb():
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')

    train_features, test_features, train_labels, test_labels = mnist.get_data()

    mnist.visualize_random()

    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)

    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn BORING GNB:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn BORING GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    mnist.visualize_wrong_class(y_pred, 8)


def main_ncc():
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')

    train_features, test_features, train_labels, test_labels = mnist.get_data()

    # mnist.visualize_random()

    ncc = NCC()
    ncc.fit(train_features, train_labels)

    y_pred = ncc.predict(test_features)

    print("Classification report NCC:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix NCC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    # mnist.visualize_wrong_class(y_pred, 8)


def main_fun_gnb():
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))

    num_examples = len(digits.data)
    num_split = int(0.7 * num_examples)

    train_features = digits.data[:num_split]
    train_labels = digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]

    gbn = GBN()
    gbn.fit(train_features, train_labels)

    y_pred = gbn.predict(test_features)

    print("Classification report GBN:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))

    print("Confusion matrix GBN:\n%s" % metrics.confusion_matrix(test_labels, y_pred))


def value_to_category(samples):
    hue_samples = []

    for sample in samples:
        hue_sample = []
        for point in sample:

            if point < (1 / 3):
                hue_sample.append(0)  # Light

            elif point > (2 / 3):
                hue_sample.append(2)  # Dark

            else:
                hue_sample.append(1)  # Grey
        hue_samples.append(hue_sample)

    return hue_samples


def main_hue_gnb():
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))

    num_examples = len(digits.data)
    num_split = int(0.7 * num_examples)

    train_features = value_to_category(digits.data[:num_split])
    train_labels = digits.target[:num_split]
    test_features = value_to_category(digits.data[num_split:])
    test_labels = digits.target[num_split:]

    gbn = GBN()
    gbn.fit(train_features, train_labels)
    y_pred = gbn.predict(test_features)

    print("Classification report HUE GBN:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix HUE GBN:\n%s" % metrics.confusion_matrix(test_labels, y_pred))


def main_nbc():
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))

    num_examples = len(digits.data)
    num_split = int(0.7 * num_examples)

    train_features = digits.data[:num_split]
    train_labels = digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]

    nbc = NBC()
    nbc.fit(train_features, train_labels)
    y_pred = nbc.predict(test_features)

    print("Classification report NBC:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix NBC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))


def main_hue_nbc():
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))

    num_examples = len(digits.data)
    num_split = int(0.7 * num_examples)

    train_features = value_to_category(digits.data[:num_split])
    train_labels = digits.target[:num_split]
    test_features = value_to_category(digits.data[num_split:])
    test_labels = digits.target[num_split:]

    nbc = NBC()
    nbc.fit(train_features, train_labels)
    y_pred = nbc.predict(test_features)

    print("Classification report HUE NBC:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix HUE NBC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

def maihn_MNIST_gbn():
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')

    train_features, test_features, train_labels, test_labels = mnist.get_data()


    gnb = GBN()
    gnb.fit(train_features, train_labels)

    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn MNIST GNB:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn MNIST GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))


if __name__ == "__main__":
    # main_boring_gnb()
    main_ncc()
    # main_nbc()
    # main_hue_nbc()
    # main_fun_gnb()
    # main_hue_gnb()


