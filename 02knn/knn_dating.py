import numpy as np
import operator
import matplotlib.pyplot as plt
from knn import classify0

def file2matrix(filename):
    fr = open(filename, 'r')
    numberOfLines = len(fr.readlines())  # 行数
    returnMat = np.zeros((numberOfLines, 3))  # 准备返回的矩阵
    classLabelVector = []  # 准备返回的标签

    fr = open(filename, 'r')
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index] = listFromLine[0 : 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化特征值，消除属性之间量级不同导致的影响
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # -------第一种实现方式---start-------------------------
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
    # -------第一种实现方式---end---------------------------------------------

    # # -------第二种实现方式---start---------------------------------------
    # norm_dataset = (dataset - minvalue) / ranges
    # # -------第二种实现方式---end---------------------------------------------

    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i], normMat[numTestVecs : m], datingLabels[numTestVecs : m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        errorCount += classifierResult != datingLabels[i]
    print("the total error rate is: %f" % (errorCount / numTestVecs))
    print(errorCount)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()

datingClassTest()
