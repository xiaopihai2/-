#AdaBoost算法
#2018/5/10

from numpy import *
def loadSimpData():
    #构建数据矩阵
    dataMat = matrix([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels
#通过阈值比较对数据进行分类
#dimen判断节点
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min(); rangeMax = dataMatrix[:, i].max()
        #这里可以看做对阈值的更新基础
        stepSize = (rangeMax - rangeMin)/numSteps
        #j也是阈值更新的基础，并且也是迭代次数的控制
        for j in range(-1, int(numSteps)+1):
            #遍历大于或小于阈值
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                #预测值与真实值比较，正确就是0，否则为1
                errArr[predictedVals == labelMat] = 0
                #计算误差
                weightedError = D.T * errArr
                print('split: dim %d, thresh %.2f, thresh ineqal:%s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))
                #更新误差，找到误差最小的弱分类器，并加入到字典中
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst
def adaBoostTrainDs(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D;',D.T)
        #计算a,相当于每个弱分类器的权重
        alpha = float(0.5*log((1.0 - error) / max(error, 10^-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:', classEst.T)
        #保证当预测值对时为e^a,预测错时为e^-a
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        #更新D
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst
        print('aggClassEst:', aggClassEst)
        #当x>0，sign(x)=1;当x=0，sign(x)=0; 当x<0， sign(x)=-1
        #sign(aggClassEst) != mat(classLabels).T生成一个布尔矩阵
        #其实就是为了累计错误率
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print('total error:', errorRate, sep='\n')
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

#测试算法：
def adaClassify(dataToClass, classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    #将每棵单层决策树模型用来预测数据类别
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        #加权求和
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

#用Logistic中的马疝病数据测试
#加载数据
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#ROC曲线的绘制与AUC计算
#将截断点不同取值下对应的TPR和FPR结果画于二维坐标系中得到的曲线，就是ROC曲线
def plotRoc(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    #数据中正例的数目，len(classLabels) - numPosClas为负例的数目
    numPosClas = sum(array(classLabels) == 1.0)
    #x,y轴的步长
    ySetp = 1/float(numPosClas)
    xSetp = 1/float(len(classLabels) - numPosClas)
    #argsort将数据从小到大排列，返回它们的下标
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #转换为矩阵，但当一维时一定要用tolist()[0]
    #这里不明白为啥要将predStrengths排序，取下标？？？？？？
    for index in sortedIndicies.tolist()[0]:
        #m,为正例个数，n，为反例个数
        #设前一个坐标为(x,y)，若当前为真正例，对应标记点为(x,y+1/m)，若当前为假正例，则标记点为（x+1/n,y），然后依次连接各点。
        if classLabels[index] == 1.0:
            delX = 0; delY = ySetp
        else:
            delX = xSetp; delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c = 'b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rat'); plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Hourse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print('the Area Under the Curve is:', (ySum*xSetp))

if __name__ == '__main__':
    # dataMat, classLabels = loadSimpData()
    # classifierArray = adaBoostTrainDs(dataMat, classLabels, 30)
    # print(classifierArray)
    # print(adaClassify([0,0], classifierArray))
    # print(adaClassify([[5,5], [0, 0]], classifierArray))
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDs(dataArr, labelArr, 10)
    print(aggClassEst)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    plotRoc(aggClassEst.T, labelArr)
    # prediction10 = adaClassify(testArr, classifierArray)
    # errArr = mat(ones((67, 1)))
    # print(errArr[prediction10 != mat(testLabelArr).T].sum())





