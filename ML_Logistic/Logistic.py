#Logistic回归

from numpy import *

def loadDataSet():
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#找到回归系数
def gradAscent(dataMatIn, classLabels):
    #mat将数据转换成Numpy矩阵， transpose为矩阵的转置
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500         #最大迭代次数
    weights = ones((n, 1))      #初始设置参数为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        #梯度上升，函数对weights为数据矩阵，但不知道为啥要乘以error

        weights = weights + alpha * dataMatrix.transpose() * error

    return weights
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    if type(wei).__name__ == 'ndarray':
        weights = [array(i) for i in wei]
    else:
        weights = wei.getA()
    print(weights)
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    #将不同类别的点分开，并画出它们的散点图
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s =30, c ='red', marker = 's')
    ax.scatter(xcord2, ycord2, s =30, c = 'green')
    x = list((-3.0, 3.0, 0.1))
    #对于sigmoid函数的分割点为0.5, 对应的y取值为0，所以在y = 0 的情况下，求解x2关于x1的表达式
    y =(-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('x1');plt.ylabel('x2')
    plt.show()
#改进的梯度上升法，使它的迭代次数更少
#随机选取值计算，并更新w值
#每次迭代的步长改变，这点与最优化很像
def srocGradAscent0(dataMatrix, classLabels, numIter=150):

    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            #更新迭代步长
            alpha = 4 / (1.0+j+i)+0.01
            randInddex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randInddex] * weights))
            error = classLabels[randInddex] - h
            weights = weights + array(alpha*error)*dataMatrix[randInddex]
    return weights
#最终的判断类别函数
def classifyVector(inX, weights):
    print(len(inX), len(weights), sep='\n')
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    #导入训练集与测试集
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    #读取训练数据中的每一行，简单处理，
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        #将训练集数据与对应的类标签分开在两个列表中。
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    #进行训练，得到系数w
    trainWeights = srocGradAscent0(trainingSet, trainingLabels, 500)
    errorCount = 0.0; numTestVec = 0.0
    #读取训练数据中的每一行，简单处理。
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        #将‘x’提出来，放到同一列表中
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # print(lineArr, trainWeights, sep = '\n')
        #预测每条记录属于哪一类，与真实类比较，分错就将errorCount + 1。
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    #分错总数除以数据总数，得到错误率。
    errorRate = (float(errorCount)/numTestVec)
    print('错误率为：{}'.format(errorRate))
    return errorRate

#模型重复十次，算平均得到精确错误率。
def multiTest():
    numTestVec = 10; errorSum  = 0.0
    for k in range(numTestVec):
        errorSum += colicTest()
    print("重复十次的平均错误率为：{}".format(float(errorSum)/numTestVec))


if __name__ == "__main__":
    data, label = loadDataSet()
    weights = gradAscent(data, label)
    print(type(weights))
    weights2 = srocGradAscent0(data, label)
    print(type(weights2))
    #getA()将Numpy矩阵转换成数组，与mat()相反
    plotBestFit(weights)
    plotBestFit(weights2)
    multiTest()