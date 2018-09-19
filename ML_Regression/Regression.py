#回归：
#2018/5/12
from numpy import *
import matplotlib.pyplot as plt

#加载数据并处理，并分割数据集与标签。
def loadDataSet(fileName):
    numFet = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFet-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#最小二乘法求解w向量
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    #linalg.det：计算方阵的行列式值
    if linalg.det(xTx) == 0.0:
        print('这个矩阵是奇异的，不能做逆。')
        return
    ws = xTx.I * xMat.T * yMat
    return ws

#局部加权线性回归函数
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        #高斯核函数，计算权重
        weights[j, j] = exp(diffMat * diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print('xTx为奇异矩阵，不可逆')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

#预测鲍鱼的年龄
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

#数据的特征大于样本点，即方程数目小于未知数的个数，有无穷多解
#需要使用岭回归算法：
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print('denom矩阵是奇异矩阵，不可逆')
        return
    ws = denom.I * xMat.T *yMat
    return ws
#数据标准化：所有特征减去各自的均值并除以方差
def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat

#前向逐步线性回归：
#数据标准化
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            #进行两次改变，向前或向后
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat



if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    xMat = mat(xArr); yMat = mat(yArr)
    print(xArr, yArr, sep = '\n')
    ws = standRegres(xArr, yArr)
    yHat2 = xMat * ws
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    #a = [[1,3],[2,4],[3,5]]
    # a = array(a)
    # a.flatten()
    # array([1, 3, 2, 4, 3, 5])
    #用于矩阵时要加上A[0]，转换成array，并且单层
    print(xMat[:, 1])
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    #计算相关系数矩阵
    print(corrcoef(yHat2.T, yMat))
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    #与上文的xCopy一样都是对数据进行排序
    strInd = xMat[:, 1].argsort(0)
    xSort = xMat[strInd][:,0,:]
    ax1 = fig.add_subplot(2,2,2)
    ax1.plot(xSort[:,1], yHat[strInd])
    ax1.scatter(xMat[:, 1].flatten().A[0], yMat.T.flatten().A[0], s =2, c = 'red')
    #鲍鱼的预测
    abX, abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat02 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat03 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print(rssError(abY[0:99], yHat01.T), rssError(abY[0:99], yHat02.T), rssError(abY[100:199], yHat03.T), sep = '\n')
    #岭回归
    ridgeWeights = ridgeTest(abX, abY)
    print(ridgeWeights)
    ax2 = fig.add_subplot(2,2,3)
    ax2.plot(ridgeWeights)
    plt.show()
    #前向逐步回归
    # print(stageWise(abX, abY, 0.01, 200))
    print(stageWise(abX, abY, 0.001, 5000))



