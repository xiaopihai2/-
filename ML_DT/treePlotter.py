import matplotlib.pyplot as plt

#定义文本框与箭头格式
decisionNode = dict(boxstyle = 'sawtooth', fc = '0.8')#定义判断节点形态
leafNode = dict(boxstyle = 'round4', fc = '0.8')      #定义叶节点形态
arrow_args = dict(arrowstyle = '<-')                    #定义箭头

#绘制带箭头的注解
#nodeTxt：节点文字标注，centerPt：节点中心位置
#parentPt：箭头起点位置(上一节点位置)， nodeType：节点属性
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])

    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)       #当函数的参数不确定时，可以使用*args 和**kwargs，*args 没有key值，**kwargs有key值
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')        #设置初始节点为(0.5,1.0)
    plt.show()

#获取叶节点的数目
#无限迭代，找到所有叶子节点
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': #若该子节点为字典类型，则该子节点也是判断节点
            numLeafs  += getNumLeafs(secondDict[key])
        else:numLeafs +=1
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': #若该子节点为字典类型，则该子节点也是判断节点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:thisDepth =1               #这里很有意思，可以仔细想想******
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth
#在父子节点间填充文本信息
#cntrPt:子节点位置，parentPt：父节点位置，txtString：标注内容
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
#绘制树图形
#myTree：树的字典， parentPt:父节点位置，nodeTxt：节点的文字标注
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    #计算当前节点的位置,第一次运行得到父节点，其实就是得到每次的判别节点
    #但不明白这个算式是怎么来的？
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)          #在父子节点间添加文本内容
    plotNode(firstStr, cntrPt, parentPt, decisionNode)#绘制带箭头的注解
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD         #更新下一级别的y
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':        #判断是不是字典
            plotTree(secondDict[key], cntrPt, str(key))      #递归绘制树图形
        else:                                                #如果是叶节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD      #不明白为什么要将y值还原为初始状态？***

def retrieveTree(i):
    listOfTree = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                {'no surfacing': {0: 'no', 1: {'flippers': {0:{'head':{0:'no', 1:'yes'}},1:'no'}}}}]
    return listOfTree[i]
if __name__ == '__main__':
    myTree = retrieveTree(0)
    createPlot(myTree)
    myTree['no surfacing'][3] = 'maybe'
    createPlot(myTree)
    # print(myTree)
    # print(getTreeDepth(myTree))
    # print(getNumLeafs(myTree))

