#朴素贝叶斯分类

from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'flea','problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]      #1代表侮辱性言论，0代表正常言论
    return postingList, classVec
#将文本中出现的单词做成集合
def createVocaList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#将文本转换成向量基于上面的函数createVocaList
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1        #存在就为1，否则为0
        else:print('{}不在我的词汇范围内'.format(word))
    return returnVec

#trainMatrix：词向量，trainCategory：类别
def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #防止有有一个词的概率为0，我们将初始值设为1
    #并将初始总词数设为2
    pONum = ones(numWords)
    plNum = ones(numWords)
    pODenom = 2.0
    plDenom = 2.0
    #遍历每一条记录与对应的标签，将同类词向量对应相加得plNum或pONum， 并各自向量求和。
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            plNum +=trainMatrix[i]
            plDenom +=sum(trainMatrix[i])
        else:
            pONum += trainMatrix[i]
            pODenom += sum(trainMatrix[i])
    # print(plNum, plDenom, pONum, pODenom, sep ='\n')
    plVect = log(plNum/plDenom)            #每个类别中每个词的次数除以该类中的总词数(防止太多太小的数相乘导致下溢取对数)
    pOVect = log(pONum/pODenom)
    return pOVect, plVect, pAbusive
#分别计算输入的数据在不同类的概率(词概率的乘积)
def classifyNB(vec2Classify, p0vec, p1vec, pClass1):
    p1 = sum(vec2Classify * p1vec) + log(pClass1)       #向量对应数据相乘
    p0 = sum(vec2Classify * p0vec) + log(1- pClass1)
    if p1 > p0:
        return 1
    else: return 0
def testingNB():
    list0Posts, listClasses = loadDataSet()
    myVocabList = createVocaList(listOPosts)
    trainMat = []
    for postinDoc in list0Posts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNBO(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))     #得到输入数据的词向量
    print(testEntry,'属于：',classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'属于：',classifyNB(thisDoc, p0V, p1V, pAb))

#词袋模型，当一条数据中某个词出现多次，这词向量中中对应得数就是几
def bag0fWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] +=1
    return returnVec

#文本解析及垃圾邮件测试
def textParse(bigString):
    import re
    list0fTokens = re.split(r'\W*', bigString)
    return [token.lower() for token in list0fTokens if len(token) > 2]

def spamTest():
    docList = [];classList = [];fullText = []
    for i in range(1, 26):
        #读取文件数据，去掉长度小于2的单词，并将文件的数据转换词集合
        #每条数据都对应它的类或1或0
        wordList = textParse(open('spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        print(i)
        wordList = textParse(open('ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocaList(docList)
    trainingSet = list(range(50)); testSet = []
    #在50条数据中随机选取10条作为测试集
    for  i in range(10):
        rangeIndex = int(random.uniform(0, len(trainingSet)))       #在0-49中随机选一个记录下标加入测试集，并在原列表中删除
        testSet.append(trainingSet[rangeIndex])
        del(trainingSet[rangeIndex])
        trainMat = [];trainClass = []
        #训练模型
        for docIndex in trainingSet:
            #选取每条记录变成词向量(词袋模型)
            trainMat.append(bag0fWords2VecMN(vocabList, docList[docIndex]))
            trainClass.append(classList[docIndex])
        #得到词概率
        p0V, p1V, pSpam = trainNBO(array(trainMat), array(trainClass))
        errorCount = 0
        #测试模型
        for docIndex in testSet:
            wordVector = bag0fWords2VecMN(vocabList, docList[docIndex])
            if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1
            print("预测为：{}， 实际为{}".format(classifyNB(array(wordVector), p0V, p1V, pSpam), classList[docIndex]))
        print('错误率为：{}'.format(float(errorCount)/len(testSet)))





if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocaList(listOPosts)
    print(myVocabList)
    trainMat = []
    #postingList的每句话转换成向量
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print(trainMat)
    p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    print(p0V, p1V, pAb, sep = '\n')
    testingNB()
    spamTest()
