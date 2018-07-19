import numpy as np
from collections import Counter

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify1(inX, dataSet, labels, k):
    dist = np.sum((inX - dataSet) ** 2, axis=1) ** 0.5
    k_labels = [labels[index] for index in dist.argsort()[0: k]]
    label = Counter(k_labels).most_common()[0][0]
    return label

def test1():
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify1([0.1, 0.1], group, labels, 3))

test1()
