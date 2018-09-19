

#k近邻算法
from numpy import *
import operator
import matplotlib
import matplotlib,pylab as plt

def createDataSst():
    group = array([[1.0,1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet         #样本数据减去每一条已知数据
    print("测试数据对应减去训练数据{}".format(diffMat),sep='\n')
    sqDiffMat = diffMat **2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances**0.5             #平方后相加再开根得到距离
    sortedDistIndicies = distance.argsort()    #返回从小到大排序的下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1      #返回字典classCount中键对应的值，否则返回默认值0,
                                                                        #这里是将离样本数据最近的3个数据对应的类进行统计
        print(classCount)
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

#将文本内容转换成数据矩阵和标签矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if filename == 'datingTestSet.txt':
            if listFromLine[-1] == 'largeDoses':
                classLabelVector.append(3)
            elif listFromLine[-1] == 'smallDoses':
                classLabelVector.append(2)
            else:
                classLabelVector.append(1)
        else:
            classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

#特征归一化,每列用最大最小法进行标准化
def autoNorm(dataSet):
    minVals = dataSet.min(0)        #得到每列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

#测试函数
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        calssifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],3)
        print('预测的类别为：{}，真实类别为：{}'.format(calssifierResult, datingLabels[i]))
        if(calssifierResult !=datingLabels[i]):
            errorCount += 1
    print('错误率为：%f' %(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['不喜欢', '有好感', '喜欢']
    percentTats = float(input('你玩游戏所消耗的时间？'))
    ffMiles = float(input('每年获得的飞行常客里程数？'))
    iceCream = float(input('每周消费的冰淇淋公升数？'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('这个人我认为你：{}'.format(resultList[classifierResult - 1]))

if __name__ == '__main__':
    # group, labels = createDataSst()
    # class1 = classify0([0,0], group, labels, 3)
    # datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels),15.0*array(datingLabels) )#描绘散点图，不同类别，15是让它的标记点更大些
    # plt.show()
    # normDataSet, ranges, minVals = autoNorm(datingDataMat)
    # print(normDataSet, ranges, minVals, sep = '\n')
    # print(class1, datingDataMat, datingLabels,sep='\n')
    datingClassTest()
    classifyPerson()