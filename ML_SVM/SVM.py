#SVM模型，基于SMO算法

from numpy import *

#读取数据的每一行，生成数据矩阵与标签矩阵。
def loadDataSet(fileName):
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[0])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

#SOM算法随机选取参数
#i为参数的下标，m为参数得个数
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

#选取大于H小于L的参数aj
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L >aj:
        aj = L
    return aj

#简化的SMO算法：
#dataMatIn:数据集；classLabels：类别标签
#C：常数C；toler：容错率；maxIter：最大循环次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    #转换成矩阵，*为矩阵乘法
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m, n = shape(dataMatrix)
    #初始化参数alphas为0列向量
    alphas = mat(zeros((m, 1)))
    #在不改变alphas中的任意个参数alpha时，遍历数据的次数
    iter = 0
    #当我们使用ufunc函数对两个数组进行计算时，ufunc函数会对这两个数组的对应元素进行计算，
    # 因此它要求这两个数组有相同的大小(shape相同)。如果两个数组的shape不同的话，会进行如下的广播(broadcasting)处理
    #此处指的是multiply()函数，对数组或矩阵做对应相乘
    while(iter < maxIter):
        alphaPairsChange = 0
        for i in range(m):
            #对Xi预测它的类，为+1， 或-1
            fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            # print(multiply(alphas, labelMat).T, (dataMatrix*dataMatrix[i, :].T), sep = '\n')
            #预测值与实际值的误差
            Ei = fXi - float(labelMat[i])
            #满足KKT条件
            if ((labelMat[i] *Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] >0)):
                #选择一个和 i 不相同的待改变的alphas[j]
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                #保存旧值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #这里是判断yi与yj是否相等，如果不等约束条件：(更新上下限)
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] -C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:      #上下限一样结束，本次循环
                    print('L == H')
                    continue
                #发现这里eta<=0
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                #更新alphas[j]
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                #限制范围
                alphas[j] = clipAlpha(alphas[j], H, L)
                #如果alphas[j]没怎么改变，结束本次循环
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print('j不能在变化了')
                    continue
                #更新alphas[i]
                alphas[i] += labelMat[j] *labelMat[i] * (alphaJold  - alphas[j])
                #更新系数b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                #b的几种选择机制
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2
                #确定更新了，记录一次
                alphaPairsChange += 1
                print('ietr:{}, i:{}, pairs改为：{}'.format(iter, i, alphaPairsChange))
        #没有实行alpha交换，迭代加1，交换就清零
        if (alphaPairsChange == 0):
            iter += 1
        else:
            iter = 0
        print('itertion 数字为：%d' % iter)

    return b, alphas




if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alpha = smoSimple(dataArr, labelArr, 0.6, 0.001, 1000)
    print(b, alpha[alpha>0],sep = '\n')
    print(dataArr, labelArr, sep = '\n')