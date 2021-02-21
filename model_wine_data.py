import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix


def file2matrix(filename):
    data = pd.read_csv(filename, names=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species'])
    dataframe = data.drop(columns='Species')
    dataMat = dataframe.to_numpy()
    # add labels matrix
    classLabels = []
    for line in range(len(data) - 1):
        if data.iloc[line][4] == 'Iris-setosa':
            classLabels.append(1)
        elif data.iloc[line][4] == 'Iris-versicolor':
            classLabels.append(2)
        elif data.iloc[line][4] == 'Iris-virginica':
            classLabels.append(3)
    return dataMat, classLabels


def data_visualize(dataMat, datalabel):
    pass


def KNNclasstest(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classcount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classcount[voteLabel] = classcount.get(voteLabel, 0) + 1
    sortedClasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClasscount[0][0]


def GMMclasstest():

    pass


def NormData(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVal, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVal


def classifyData():
    resultsList = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    # test input
    Sepallength = float(input("sepal length in cm:"))
    Sepalwidth = float(input("sepal width in cm:"))
    Petallength = float(input("petal length in cm:"))
    Petalwidth = float(input("sepal width in cm:"))
    # open file
    filename = "iris.data"
    DataMat, dataLabels = file2matrix(filename)
    # data normalization
    normMat, ranges, minVals = NormData(DataMat)
    inArr = np.array([Sepallength, Sepalwidth, Petallength, Petalwidth])
    # test normalization
    norminArr = (inArr - minVals) / ranges
    classifierResult = KNNclasstest(norminArr, normMat, dataLabels, 3)
    # result
    print("this would be %s" % (resultsList[classifierResult-1]))


def classTest():
    filename = 'iris.data'
    data, classLabels = file2matrix(filename)
    horatio = 0.20
    normMat, ranges, minVals = NormData(data)
    m = normMat.shape[0]
    numTestVecs = int(m*horatio)
    errorcount = 0.0
    for i in range(numTestVecs):
        classifierResult = KNNclasstest(normMat[i, :], normMat[numTestVecs:m, :],
                                        classLabels[numTestVecs:m], 4)
        print("Classify result:%d\tTrue:%d" % (classifierResult, classLabels[i]))
        if classifierResult != classLabels[i]:
            errorcount += 1.0
    print("error rate:%f%%" % (errorcount / float(numTestVecs) * 100))


if __name__ == '__main__':
    classTest()
