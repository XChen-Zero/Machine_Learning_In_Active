import numpy as np
import operator


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []  # 标签值
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromline = line.split('\t')
        #print(listFromline)
        returnMat[index, :] = listFromline[0: 3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(range, (m, 1))
    return normDataSet, ranges, minVals

