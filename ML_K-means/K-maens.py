#K-means聚类：
#2018/5/14

from numpy import *
import matplotlib
import matplotlib.pyplot as plt


#加载数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

#计算向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    #提取每一个特征的列向量
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        #最小值加上极差的(0, 1)倍，确定随机选择的中心在范围内
        #这里就能得到k*n的中心数据矩阵
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    #两列，一列是簇的索引下标，二列为误差(我认为是该点到对应簇的距离的平方)
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChange = True
    while clusterChange:
        clusterChange = False
        #遍历数据中得每一条数据
        for i in range(m):
            minDist = inf; minIndex = -1
            #遍历k个中心点数据，这里得到每条数据到那个中心点距离最近，并返回它的下标
            for j in range(k):
                #计算样本数据到中心点的矩离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            #当簇的中心还在变化就继续迭代       
            if clusterAssment[i,0] != minIndex: clusterChange = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):
            #分割属于同一簇的数据，并求平均值来，更新当前k的值
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssment

#二分K-means
def biKmeans(dataSet, k, distMeans = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    #第一次数据集的均值作为中心
    centroid0 = mean(dataSet, axis = 0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        #计算距离
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            #尝试划分每一簇，第一次进来就是整个数据集
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            if len(ptsInCurrCluster) == 0:
                continue
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            sseSplit = sum(splitClustAss[:, 1])
            #不是该簇的数据计算它们的距离和(误差和)
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit 和 sseNotSplit:', sseSplit, sseNotSplit)
            #判断分开前后数据的误差是否更小
            if (sseSplit + sseNotSplit) < lowestSSE:
                #一分为二的原始类下标
                bestCentToSplit = i
                #一分为二的中心点
                bestNewCents = centroidMat
                #分开后的clusterAssment
                bestClustAss = splitClustAss.copy()
                #更新lowestSSE
                lowestSSE = sseSplit + sseNotSplit
        #这里可以这样理解：当二分后二分中0类继承父类标签，1类赋予新标签即原标签列表的长度
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentTopSplit is:', bestCentToSplit)
        print('the len of bestClustAss is:', len(bestClustAss))
        #更新父类的中心为分开后0类的中心
        centList[bestCentToSplit] = bestNewCents[0, :]
        #加上新类的中心
        centList.append(bestNewCents[1,:])
        #最后更新距离(误差)
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment


#实例
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1]*pi/180) * sin(vecB[0, 1]*pi/180)
    b = cos(vecA[0, 1]*pi/180) * cos(vecB[0, 1]*pi/180) * cos(pi * (vecB[0, 0] - vecA[0, 0]/180))
    return arccos(a + b)*6371.0

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    print("数据：", datMat)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeans=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    print(myCentroids)
    ax1.scatter(array(myCentroids[:,0]), array(myCentroids[:,1]), marker='+', s=300)
    plt.show()



if __name__ == '__main__':
    # datMat = mat(loadDataSet('testSet.txt'))
    # myCentroids, clustAssing = kMeans(datMat, 4)
    # print(myCentroids, clustAssing,sep = '\n')
    # datMat3 = mat(loadDataSet('testSet2.txt'))
    # centList, myNewAssments = biKmeans(datMat3, 3)
    # print(centList, myNewAssments)
    clusterClubs(5)