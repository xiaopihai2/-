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
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C =C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros(self.m, 2))
def calcEK(oS, K):
    fXK = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[K, :].T) + oS.b)
    EK = fXK - float(oS.labelMat[K])
    return EK

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) >1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEK(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEK(oS, j)
    return j, Ej
def updateEk(oS, k):
    Ek = calcEK(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEK(oS, i)
    if ((oS.labelMat[i] *Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C +oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L等于H')
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print('eta>=0')
            return 0
        oS.alphas[j] -=oS.labelMat[j] * (Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j不能再改变了')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] *(alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j]*\
            (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j]*\
            (oS.alphas[j] - oS.alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


