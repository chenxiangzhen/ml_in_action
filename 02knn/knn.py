from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(intX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet  # numpy.tile(A,B)重复A，B次
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDisIndicies = distances.argsort()  # 从小到大排列，返回index
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def test1():
    """
    第一个例子演示
    """
    group, labels = createDataSet()
    print('group:'+str(group))
    print('labels:'+str(labels))
    print(classify0([0.1, 0.1], group, labels, 3))


test1()