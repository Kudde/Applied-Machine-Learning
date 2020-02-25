from sklearn import datasets , svm, metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
import collections

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 5, index +1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
#plt.show()

num_examples = len(digits.data)

num_split = int(0.7*num_examples)


train_features = digits.data[:num_split]
train_labels =  digits.target[:num_split]
test_features = digits.data[num_split:]
test_labels = digits.target[num_split:]
print(train_features)

print("Number of training examples: ",len(train_features))
print("Number of test examples: ",len(test_features))
print("Number of total examples:", len(train_features)+len(test_features))

tree_clf = DecisionTreeClassifier(max_depth=14)
tree_clf.fit(train_features, train_labels)

predicted = tree_clf.predict(test_features)
#print(predicted)
print("Classification report for classifier %s:\n%s\n"
      % (tree_clf, metrics.classification_report(test_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))

dot_data = tree.export_graphviz(tree_clf,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')