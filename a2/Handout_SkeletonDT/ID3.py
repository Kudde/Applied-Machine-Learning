from collections import Counter
from graphviz import Digraph
import math


class ID3DecisionTreeClassifier:

    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0
        self.__currentNode = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))

    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    def most_common_class(self, target):
        score_dict = {}
        for item in target:
            if item in score_dict:
                score_dict[item] += 1
            else:
                score_dict[item] = 1
        return max(score_dict, key=score_dict.get)

    def find_entropy(self, target, classes):
        dict = {}
        n = len(target)
        entropy = 0

        for c in classes:
            i = target.count(c)
            dict[c] = i
        diversity = 0
        for k, v in dict.items():
            if v is not 0:
                diversity += 1
        if diversity is 1:
            return entropy
        for k, v in dict.items():
            # print("V : " , v)
            # print("N : " , n)
            # print("log : " , len(classes))
            if v is 0 :
                entropy += 0
            else:
                entropy += -(v/n) * math.log(v/n, len(classes))
        return entropy
    
    def yet_another_entropy(self, class_dict):
        n = sum(class_dict.values())
        entropy = 0
        if len(class_dict) is 1:
            return entropy
        for k, v in class_dict.items():
            entropy += -(v/n)*math.log(v/n, len(class_dict))
        return entropy

    def find_average_information(self, tuple_list, classes):
        number_of_entries = len(tuple_list)
        attribute_dict = {}
        for t in tuple_list:
            attribute_key = t[0]
            attribute_value = t[1]
            if attribute_key in attribute_dict :
                if attribute_value in attribute_dict[attribute_key]:
                    attribute_dict[attribute_key][attribute_value] += 1
                else:
                    attribute_dict[attribute_key][attribute_value] = 1
                    
            else:
                attribute_dict[attribute_key] = {attribute_value : 1}
        average_information = 0
        for k, v in attribute_dict.items():
            entropy = self.yet_another_entropy(v)
            average_information += (sum(v.values())/number_of_entries)*entropy
            
        return average_information

    def find_information_gain(self, data, target, classes, entropy):
        information_gain_list = []
        attribute_list = []
        for i in range(len(data[0])):
            for j in range(len(data)):
                attribute_list.append((data[j][i],target[j]))
            information_gain_list.append(entropy - self.find_average_information(attribute_list, classes))
            attribute_list.clear()
        return information_gain_list

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes, classes):
        
        # Change this to make some more sense
        entropy = self.find_entropy(target, classes)
        information_gain_list = self.find_information_gain(data, target, classes, entropy)
        ziped_information_gain_dict = {}
        for a, i in zip(attributes.keys(), information_gain_list):
            ziped_information_gain_dict[a] = i

        return max(ziped_information_gain_dict, key=ziped_information_gain_dict.get)

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):

        # fill in something more sensible here... root should become the output of the recursive tree creation
        # print("data: " , data)
        # print("target: " , target)
        # print("attributes: " , attributes)
        # print("classes: " , classes)

        root = self.new_ID3_node()

        # used to print class distribution on graph nodes
        distribution_dict = {}
        for item in target:
            if item in distribution_dict:
                distribution_dict[item] += 1
            else:
                distribution_dict[item] = 1
        
        # Base cases
        # If all samples belong to one class <class_name>
        if len(distribution_dict) == 1: #
            root['label'] = next(iter(distribution_dict))
            self.add_node_to_graph(root, self.__currentNode)
            return root

        # If Attributes is empty, then
        if len(attributes) == 0 :
            root['label'] = self.most_common_class(target)
            self.add_node_to_graph(root,  self.__currentNode)
            return root

        # begin split
        split_attribute = self.find_split_attr(data, target, attributes, classes)
    
        root['attribute'] = split_attribute
        root['nodes'] = {}
        root['entropy'] = self.find_entropy(target, classes)
        root['classCounts'] = distribution_dict

        if self.__nodeCounter is 1:
            self.add_node_to_graph(root)
        else:
            self.add_node_to_graph(root,  self.__currentNode)

        # branching here
        for attribute_value in attributes[split_attribute]:

            this_new_dict_change_me = {}
            for c in classes:
                this_new_dict_change_me[c] = 0

            branch_data = []
            branch_target = []
            branch_attributes = {}
            attribute_index = list(attributes.keys()).index(split_attribute)

            for k, v in zip(data, target):
                if k[attribute_index] is attribute_value:
                    this_new_dict_change_me[v] += 1

            number_in_subset = sum(this_new_dict_change_me.values())

            # leaf due to empty subset
            if number_in_subset == 0:
                leaf_node = self.new_ID3_node()
                leaf_node['label'] = max(distribution_dict, key=distribution_dict.get)
                root['nodes'][leaf_node['id']] = leaf_node
                self.add_node_to_graph(leaf_node, root['id'])

            # create new data for next iteration
            else:
                for d, t in zip(data, target):
                    if d[attribute_index] is attribute_value:
                        pruned_tuple = d[:attribute_index] + d[attribute_index + 1:]
                        branch_data.append(pruned_tuple)
                        branch_target.append(t)
                
                for a in attributes:
                    if a is not split_attribute:
                        branch_attributes[a] = attributes[a]

                # print("-----")
                # print("Split attribute : " , split_attribute)
                # print("Attribute value : " , attribute_value)
                # print("Index : " , attribute_index)
                # print("Nodes : " , self.__nodeCounter)
                # print(attributes)
                # print(branch_attributes)
                # print("Data . " , data)
                # print("Branch data: %s \nBranch targets: %s \nBranch attributes: %s" % (branch_data, branch_target, branch_attributes))
                
                self.__currentNode = root['id']
                temp_node = self.fit(branch_data, branch_target, branch_attributes, classes)
                root['nodes'][temp_node['id']] = temp_node

        #self.find_split_attr(zip(data, target))
        # for x,y in zip(data, target):
        #     print(x, y)
        # print(root)
        # new_node = self.new_ID3_node()
        # self.add_node_to_graph(new_node, 0)
        # new_node_2 = self.new_ID3_node()
        # self.add_node_to_graph(new_node_2, 0)
        return root

    def predict(self, data, tree, attributes):
        predicted = list()
        # print(attributes)
        # print(tree)
        print("lets go!!!")

        for sample in data:
            sample_data = {}
            # print(sample)
            for attribute in attributes.keys():
                sample_data[attribute] = sample[list(attributes.keys()).index(attribute)]

            predicted.append(self.predict_rec(tree, sample_data, attributes))

        return predicted

    def predict_rec(self, node, x, attributes):
        # print(node['label'])

        # print(current_attribute)
        # print(current_attribute_value)
        # print(attributes)
        # print(x)
        # print("-----")
        # for c in child_list:
        #     print(c)
        # print("Selecting...")
        # print(next_node)

        # if node is leaf
        #  return the class label of node
        if node['label'] is not None:
            return node['label']

        # else
        #  find the child c among the children of node
        #  representing the value that x has for
        #  the split_attribute of node
        #  return predict_rek( c, x)
        else:
            current_attribute = node['attribute']
            current_attribute_value = x[current_attribute]
            print("Current Attribute", current_attribute)
            print("Current Attribute Value", current_attribute_value)

            # Find branch for current attribute value
            branch_index = attributes[current_attribute].index(current_attribute_value)
            # print(branch_index)
            print("Branch index", branch_index)


            # Select branch
            child_list = []
            print("Parent : " , node)
            for child in node['nodes']:
                print("Child : ", child)
                child_list.append(node['nodes'][child])

            # print("Child list")
            # print(child_list)
            # print("Child list size")
            # print(len(child_list))
            # print("Branch index" , branch_index)
            next_node = child_list[branch_index]
            return self.predict_rec(next_node, x, attributes)


