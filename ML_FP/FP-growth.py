#FP-growth算法来高效的发现频繁项集：
#(1)构建FP树：第一遍统计出现的频率，小于最小支持度的删除
#             第二遍扫描只考虑那些频繁项集
#(2)从FP树中挖掘频繁项集

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind = 1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            #递归调用disp函数
            child.disp(ind + 1)


#FP树构建函数

def createTree(dataSet, minSup = 1):
    headerTable = {}
    #trans：我以为是整个item，没想到是keys
    for trans in dataSet:
        for item in trans:
            #获取字典中关键字为item的值headerTable.get(item, 0)，没找到就默认为0(主要为了初次构建字典)
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    #删除不满足最小支持度的元素
    #字典在遍历时不能进行修改，需改为list
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    #得到满足最小支持度的元素集合
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0: return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    #创建根节点
    retTree = treeNode('Null Set', 1, None)
    #再次遍历数据，这次只关心频繁项集
    #遍历数据，找到里面的频繁项，再根据频繁项值的大小，进行排序
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            #根据字典中的values排序，再返回排序后的keys.
            #得到每条记录去除非频繁项后，按值从大到小排序后的keys列表
            ordereItems = [v[0] for v in sorted(localD.items(), key = lambda p: p[1], reverse = True)]
            #ordereItems：如:['z', 'r'];retTree：根节点；headerTable：频繁项集字典；count：初始计数
            updateTree(ordereItems, retTree, headerTable, count)
    return retTree, headerTable

#count:是每条记录都为1，可以看该条数据中每个元素都出现了一次，这样，每次更新都可以加一
def updateTree(items, inTree, headerTable, count):
    #如果孩子节点存在，就+1,不存在就创建一个
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #这里理解为头指针表，headerTable中的每个元素的出现的地方都必须有所指向
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        #将items剩下的数据重新回带到函数中，但此时父节点变成它上面创造的子节点
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

#每个headerTable中元素的第2个keys存的一个treeNode对象，它表示指向FP树中该元素出现的地方（父节点定位）
#而每一个treeNode中的nodeLink存放下一个指向，整个思想很有意思
def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

#简单的数据集测试

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

#这里将原始数据集做成了字典，应该是为了方便计数
def creteInitSet(dataSet):
    reDict = {}
    for trans in dataSet:
        reDict[frozenset(trans)] = 1
    return reDict


#从一棵FP树挖掘频繁项集
#(1)从FP树中获取条件模式基
#(2)利用条件模式基，构建一个条件FP树
#(3)迭代1,2步，直到树包含一个元素为止
#了解前缀路径：所查找的元素与根节点之间的所有内容

#用上面的头指针表：headerTable来发现以给定元素项结尾的所有路径函数
def ascendTree(leafNode, prefixPath):
    #判断是否具有父节点
    if leafNode.parent != None:
        #初次添加一定是本节点的名字
        prefixPath.append(leafNode.name)
        #向上迭代
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    condPats = {}
    #判断初次指向不为None
    while treeNode != None:
        prefixPath = []
        #指向的对象，向上迭代
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            #赋予本指向对象的上溯对象树路径值：由当前计数决定
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        #指向下一个节点，进行迭代，直到没有指向为止
        treeNode = treeNode.nodeLink
    return condPats

#递归查找频繁项集函数
#先查找满足最小支持度的单元素频繁项，再在该元素出发，查找与其他元素组合是否为频繁项集
#但如何查找，不可能穷举，所以按照指针表来
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    #先排序，将头指针表中的元素按值排序
    bigL = [v[0] for v in sorted(headerTable.items(), key = lambda p:p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #感觉这里有问题，应该先判断是否满足最小支持度，再添加单元素
        freqItemList.append(newFreqSet)
        #寻找条件FP树形式像creteInitSet函数得到的initSet
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #然后构建FP树，考虑为什么没有输入conditional tree for:{‘r’}
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            print('conditional tree for:', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)




if __name__ == '__main__':
    rootNode  = treeNode('pyramid', 9, None)
    rootNode.children['eye'] =treeNode('eye', 13, None)
    rootNode.disp()
    simpDat = loadSimpDat()
    initSet = creteInitSet(simpDat)
    print(initSet)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    print(myFPtree)
    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)
