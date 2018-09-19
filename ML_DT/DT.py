#决策树的算法实现，用信息增益(互信息)的ID3算法：

#计算给定数据集的香农熵,这里其实就是将数据分为N类，
# 计算每类占总数据的比例，在算它的自信息，最后所有类的自信息相加

#本节内容为重点，需要仔细推敲每个方法的作用******，
#但完整的决策树算法应该含有剪枝的内容，以后一定要写完
from math import log
import operator
from treePlotter import plotNode,createPlot,\
    getNumLeafs,getTreeDepth,plotMidText,plotTree,retrieveTree

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] +=1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

#划分数据集
#待划分的数据集， 划分数据集的特征(可以理解为树的节点在原数据的下标)， 特征的返回值(节点满足的条件值)
#这里很有意思，选取节点分开两半，再将两半合并，来达到决策树的核心结构判断
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:      #相当于判断是否满足节点条件
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])     #合并a, b列表 ->[a,b]
            retDataSet.append(reducedFeatVec)           #合并啊a, b列表->[a, [b]]
    return retDataSet

def createDataSet():
    dataSet = [[1,1,'yes'], [1,1,'yes'], [1,0,'no'], [0,1,'no'], [0,1,'no']]
    labels = ['no surfacing', 'flippers']           #此处表示特别标签的属性，如no surfacing下分为0,1两类
    return dataSet, labels

#可以参考《数据挖掘概念与技术》P218-P219页
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)       #计算整体数据的自信息
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #选取每条记录的第i个特征取值
        uniqueVals = set(featList)          #在特征i的选择下分为几类
        newEntropy = 0.0
        #计算在当前第i个特征下的期望信息需求，先算在第i特征下分为几类，且各自所占的比例为prob
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy   #此处才叫信息增益，越大越好
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classCount:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #树叶子的情况，这两种情况不是很明白
    if classList.count(classList[0]) == len(classList): #如果分类后的数据属于同一类
        return classList[0]
    if len(dataSet[0]) == 1:                            #每条数据只剩最后一列了
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)        #得到最好的特征下标作为数节点
    bestFeatLabel = labels[bestFeat]                    #找到该特征的值
    myTree = {bestFeatLabel:{}}                         #节点诞生
    del(labels[bestFeat])                               #在备选节点(所有特征的列表)中删除被选中的特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)                        #在该节点下，数据被分为几类，分别取值为
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        #此处将节点判别后的数据重新调用该方法，先用splitDataSet方法判别分类，再将分类的数据重新构造决策树
    return myTree

#想象一下一颗成型的决策树，引进一条新的记录，怎样运用决策树来对该条数据标注
def classify(inputTree, featLabels, testVec):
    print('ubu {}'.format(featLabels))
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)      #确定最优的判别特征在特征列表中下标为多少
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:   classLabel = secondDict[key]
    return classLabel

#存储决策树模块
#因为每次使用时都重复构建决策树，我们将构建看好的决策树存储到文件中
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
#打开模型
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    # myData, labels = createDataSet()
    # print( labels)
    # myTree = createTree(myData, labels)
    # myTree = retrieveTree(0)
    # print(myTree)
    # storeTree(myTree, 'MyDt.txt')
    # print(grabTree('MyDt.txt'))
    # print(splitDataSet(myData, 0, 1))
    # print(splitDataSet(myData, 0, 0))
    # print(calcShannonEnt(myData))
    # print(chooseBestFeatureToSplit(myData))
    # print(myData)
    # myTree = retrieveTree(0)
    # print(classify(myTree, labels, [1,0]))
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    createPlot(lensesTree)


