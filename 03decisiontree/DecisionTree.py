# -*- coding: UTF-8 -*-
import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter


def createDataSet():
    '''
    创建数据集
    :return: 数据集，标签
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet: 数据集
    :return: 香农熵
    '''
    label_count = Counter(data[-1] for data in dataSet)
    probs = [p[1] / len(dataSet) for p in label_count.items()]
    shannonEnt = sum([-p * log(p, 2) for p in probs])
    return shannonEnt


def splitDataSet(dataSet, index, value):
    '''
    划分数据集
    :param dataSet: 待划分的数据集
    :param index: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return: index列为value的数据集（该数据集需要排除index列）
    '''
    retDataSet = []
    for featVec in dataSet: 
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择切分数据集的最佳特征
    :param dataSet: 需要切分的数据集
    :return: 切分数据集的最优的特征列
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    选择出现次数最多的一个结果
    :param classList: label列的集合
    :return: 最优的特征列
    '''
    major_label = Counter(classList).most_common(1)[0]
    return major_label


def createTree(dataSet, labels):
    '''
    创建决策树
    :param dataSet: 训练数据集
    :param labels: labels，不是目标变量
    :return: 创建完成的决策树
    '''
    classList = [example[-1] for example in dataSet]
    print('classList[0]', classList[0])
    print('classList.count(classList[0])', classList.count(classList[0]))
    print('len(classList)', len(classList))
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''
    对新数据进行分类
    :param inputTree: 已经训练好的决策树模型
    :param featLabels: Feature标签对应的名称，不是目标变量
    :param testVec: 测试输入的数据
    :return: 分类的结果值
    '''
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    '''
    将之前训练好的决策树模型存储起来，使用 pickle 模块
    :param inputTree: 以前训练好的决策树模型
    :param filename: 要存储的名称
    :return:
    '''
    import pickle
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    '''
    将之前存储的决策树模型使用 pickle 模块 还原出来
    :param filename: 之前存储决策树模型的文件名
    :return: 将之前存储的决策树模型还原出来
    '''
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def fishTest():
    '''
    对动物是否是鱼类分类的测试函数，并将结果使用 matplotlib 画出来
    :return:
    '''
    myDat, labels = createDataSet()
    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    print(classify(myTree, labels, [1, 1]))
    # 画图可视化展现
    dtPlot.createPlot(myTree)


def ContactLensesTest():
    '''
    预测隐形眼镜的测试代码，并将结果画出来
    :return:
    '''
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    dtPlot.createPlot(lensesTree)


if __name__ == "__main__":
    fishTest()
    #ContactLensesTest()
