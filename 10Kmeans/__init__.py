#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "Ng WaiMing"

from kMeans import kMeans
from kMeans import loadDataSet
from kMeans import randCent
from kMeans import distEclud
from kMeans import biKmeans
from numpy import *

if __name__ == '__main__':
    dataMat = mat(loadDataSet('testSet.txt'))
    print('min(dataMat[:, 0])', min(dataMat[:, 0]), '\n')
    print('min(dataMat[:, 1])', min(dataMat[:, 1]), '\n')
    print('max(dataMat[:, 0])', max(dataMat[:, 0]), '\n')
    print('max(dataMat[:, 1])', max(dataMat[:, 1]), '\n')
    print(randCent(dataMat, 2), '\n')
    print(distEclud(dataMat[0], dataMat[1]))
    centroids, clusterAssment = kMeans(dataMat, 4)
    print('centroids:\n', centroids, '\n')
    print('clusterAssment:\n', clusterAssment, '\n')
    dataMat3 = mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(dataMat3, 3)
    print('centList: \n', centList, '\n')
    # fileName = '../../../../data/k-means/places.txt'
    # imgName = '../../../../data/k-means/Portland.png'
    # kMeans.clusterClubs(fileName=fileName, imgName=imgName, numClust=5)
