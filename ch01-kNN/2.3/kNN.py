import numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetShape = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetShape, 1)) - dataSet
    distances = ((diffMat ** 2).sum(axis=1)) ** 0.5
    classCount = {}
    sortedDistIndicia = distances.argsort()
    for i in range(k):
        voteLabel = labels[sortedDistIndicia[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def img2Vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i * 32 + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    trainingDigitsFilename = 'trainingDigits'
    digitsFilename = listdir(trainingDigitsFilename)
    traingingLen = len(digitsFilename)
    labels = []
    dataSetMat = np.zeros((traingingLen, 1024))
    for i in range(traingingLen):
        digitLabel = (digitsFilename[i].split('.'))[0].split('_')[0]
        labels.append(digitLabel)
        dataSetMat[i, :] = img2Vector(trainingDigitsFilename+'/'+digitsFilename[i])

    errors = 0

    testDigitsFilename = 'testDigits'
    testFilename = listdir(testDigitsFilename)
    testLen = len(testFilename)
    for i in range(testLen):
        testDigit = testFilename[i].split('.')[0].split('_')[0]
        inX = img2Vector(testDigitsFilename + '/' + testFilename[i])
        # print('%s %s' % (testDigit, testFilename[i]))
        forecastDigit = classify0(inX, dataSetMat, labels, k=6)
        # print('test %d , forecast %d' % (int(testDigit), int(forecastDigit)))
        if(testDigit != forecastDigit):
            errors += 1

    print('errors: %d, percent: %f' % (errors, errors / testLen))

if __name__ == '__main__':
    handwritingClassTest()