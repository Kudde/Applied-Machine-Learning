from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, datasets
import seaborn as sns

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import completeness_score



digits = load_digits()
print(digits.data.shape)
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)

print(clusters.shape)

# fig, ax = plt.subplots(2, 5, figsize=(8, 3))
# centers = kmeans.cluster_centers_.reshape(10, 8, 8)
# for axi, center in zip(ax.flat, centers):
#     axi.set(xticks=[], yticks=[])
#     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
#

labels = np.zeros_like(clusters)

for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

print("Accuracy")
print(accuracy_score(digits.target, labels))

print("Completeness")
print(completeness_score(digits.target, labels))


mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label');

plt.show()






