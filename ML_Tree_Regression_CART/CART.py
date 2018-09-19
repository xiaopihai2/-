#树回归，本次实现CART算法：

from numpy import *
from tkinter import *

#建立树节点
class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left
#加载数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #将每行映射为浮点数
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    #nonzero：返回数据中非零数据的索引
    #nonzero(dataSet[:, feature] > value)[0]：返回满足条件的数据第所有行
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1
#建立叶节点函数
def regLeaf(dataSet):
    return mean(dataSet[:, -1])
#误差估计函数
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]
#我们选取当前特征中的每一个值(当前列中每一个值)，取该值作为判断节点，比较分类后的总方差，取最小的对应的节点值
#分开左右树后，再分别在两支树中重复上述操作
def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = ( 1,4)):
    #tolS：为容许的误差下降值；tolN：切分的最小的样本数
    tolS = ops[0] ; tolN = ops[1]
    #不同剩余特征值的数目
    # print(len(set(dataSet[:, -1].T.tolist()[0])))
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    #计算样本的总方差
    S = errType(dataSet)
    #初始化最优特征下标，和最优判断值
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        #遍历当前特征中的每个值作为判断值
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            #计算左右树的总方差
            news = errType(mat0) + errType(mat1)
            if news < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = news
    #在本例中，起初我们思考应该不只一层，这里起到了限制，误差下限值
    #即误差减少不大则退出。就是说分不分开这个数据集，它们的总方差差不多，没必要分开
    # print(S, bestS,sep = '\n')
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

#回归树剪枝函数（后剪枝
#判断当前处理的节点是否是叶节点
def isTree(obj):
    return (type(obj).__name__ == 'dict')
#从上往下遍历树到叶节点为止，找到两个叶节点计算平均值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        print(tree, testData, sep = '\n')
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        print('\n')
        print(lSet, rSet, sep = '\n')
        #power(x,y)：返回x的y次方数
        #返回找到的最底层的左右枝节点，算它们的误差和
        #这里是求(x - x的平均值)^2；左右节点的和
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        #整棵树的均值
        print(tree['left'], tree['right'], sep = '\n')
        treeMean = (tree['left'] +tree['right'])/ 2.0
        #整棵树的(x - x的平均值)^2；
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        #比较分开与合并的高低
        if errorMerge < errorNoMerge:
            print('合并')
            return treeMean
        else: return tree
    else: return tree
#将X与Y中的数据格式化
#树模型，将树的节点变成线性模型
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n))); Y = mat(ones((m ,1)))
    X[:, 1:n] = dataSet[:, 0:n-1]; Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('奇异矩阵，不可逆')
    ws = xTx.I * X.T*Y
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

#用树回归进行预测
#回归树
def regTreeEval(model, inDat):
    return float(model)

#模型树
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X * model)
#这里默认为回归树
def treeForeCats(tree, inData, modelEval = regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    #取出特征中的值与判断值对比，分出左右树
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):        #如果是树，就递归调用函数
            return treeForeCats(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCats(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

#将测试数据的每一行拿出来调用treeForeCats函数
def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCats(tree, mat(testData[i]), modelEval)
    return yHat

if __name__ == '__main__':
    # myDat = loadDataSet('ex00.txt')
    # myMat = mat(myDat)
    # tree = createTree(myMat)
    # myDat1 = loadDataSet('ex0.txt')
    # myMat1 = mat(myDat1)
    # tree2 = createTree(myMat1)
    #
    # #因为ex2.txt中第二列的数据比较大，则tolS就比较敏感
    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    # tree3 = createTree(myMat2, ops = (10000, 4))        #改变容错率，使树更简洁（预剪枝）
    # print(tree, tree2,tree3, sep ='\n' )

    #后剪枝,太坑了，没注意
    myTree = createTree(myMat2, ops = (0, 1))
    print('未修剪的树：{}'.format(myTree))
    myDataTest = loadDataSet('ex2test.txt')
    myMatTest = mat(myDataTest)
    myTree2 = prune(myTree, myMatTest)

    #模型树
    myMat3 = mat(loadDataSet('exp2.txt'))
    print('修剪后的树：{}'.format(myTree2))
    myTree2_2 = createTree(myMat3, modelLeaf, modelErr, (1, 10))
    print('模型树：{}'.format(myTree2_2))

    #预测：
    #默认回归树
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops = (1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    #corrcoef(rowvar = 0),计算列的相关系数,这里形成2*2的相关系数矩阵
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0,1])

    #模型树
    myTree = createTree(trainMat, modelLeaf, modelErr, (1,20))
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print(corrcoef(yHat, testMat[:,1], rowvar=0)[0,1])

    #Tkinter
    root = Tk()
    myLabel = Label(root, text = 'Hello, World')
    myLabel.grid()
    #与plt.show()一样，显示出来
    root.mainloop()
