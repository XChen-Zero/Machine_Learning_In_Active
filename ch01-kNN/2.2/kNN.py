import numpy as np
import operator


def classify0(inX, dataSet, labels, k):
    # k近邻
    # inX为用于分类的输入向量  dataSet为训练样本集  labels为标签向量  k表示用于选择最近邻居的数目
    dataSetSize = dataSet.shape[0] #得到训练样本集的第一维大小
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet #使inX广播到dataSetSize大小，然后和样本做差
    sqDiffMat = diffMat ** 2 #平方
    sqDistance = sqDiffMat.sum(axis=1) #距离差的平方求和，注意axis=1
    distances = sqDistance ** 0.5 #距离差的平方和 进行开方
    sortedDistIndicies = distances.argsort() #返回从小到大排序的 下标
    classCount = {}
    for i in range(k):
        #取前k小的样本对应的标签
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #找到标签数量最大的那一个标签
    return sortedClassCount[0][0]


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
        returnMat[index, :] = listFromline[0: 3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    # 归一化特征值
    # 因为特征值的取值范围不一样，很可能出现某一个大的特征值的影响结果远大于其他特征值，所以进行归一化
    # newValue = (oldValue - min)/(max - min)
    # 其中min和max分别是数据集中的最小特征值和最大特征值

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("you will probably like this person: ", resultList[classifierResult - 1])


if __name__ == '__main__':
    classifyPerson()
