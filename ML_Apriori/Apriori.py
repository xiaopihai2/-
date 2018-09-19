#关联规则Apriori算法：
#没有实现国会投票

def loadDataSet():
    return [[1,3,4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#提取数据集中的数据的集合
def createC1(dataSet):
    C1 = []
    for transanction in dataSet:
        for item in transanction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #frozenset：与set不同，两者都是集合但frozenset是冻结的，不能添加，移除。可是它有哈希值，可以作为字典的key
    return map(frozenset, C1)

#minSupport：最小支持度
#计算Ck中的每个元素的支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    #这里很奇怪，如果Ck不变成list，就会只运行一次循环
    Ck = list(Ck)
    for tid in D:
        for can in Ck:
            #遍历D，如果当前can是tid的子集
            if can.issubset(tid):
                #字典中存在关键字就加1，不存在就赋初始值
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    #计算数据集的数目
    numItems = float(len(D))
    #频繁集集合
    retList = []
    #存储元素集合中每个元素的支持度
    supportData = {}
    for key in ssCnt.keys():
        support = ssCnt[key] / numItems
        #满足最小支持度就添加到列表中
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

#组合函数
def aprioriGen(Lk, k):
    reList = []
    lenLk = len(Lk)
    print('Lk:',Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            print("L:", list(Lk[i])[:k-2])
            #为了不用遍历去寻找重复值，
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:#此处不是很理解，书上说是为了避免重复，但我认为还有频繁集的关系
                #取并集
                reList.append(Lk[i] | Lk[j])
    print("RE:",reList)
    return reList
#频繁项集的生成函数
def apriori(dataSet, minSupport = 0.5):
    #初次进入函数也是先挑选集合元素的支持度，在进行组合判断
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, SupportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    #这里其实判断上一级的集合中（可以理解为父类数据）是否还存在元素可以用来组合
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supk = scanD(D, Ck, minSupport)
        SupportData.update(supk)
        L.append(Lk)
        k += 1
    return L, SupportData

#关联规则的生成函数
def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    #无法从单元素项集中构建关联规则，所以i要从1开始，而不是0
    for i  in range(1, len(L)):
        #L[1]：[frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})]
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                #当项集中元素个数>2,则可以组合：len(H1)>=3
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #当项集中只有两个元素时：len(H1)<=2
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, br1, minConf = 0.7):
    prunedH = []
    for conseq in H:
        #如：2——>3,则[2, 3]的支持度除以2的支持度
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            #输出关系式， 并添加到列表中：
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

#当len(H)>=3时：
def rulesFromConseq(freqSet, H, supportData, br1, minConf = 0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        #单元素集组合成二元素集
        Hmp1 = aprioriGen(H, m + 1)
        #如：freqSet = [{1, 2, 3}], 则Hmp1 = [{1,2}, {1,3}, {2,3}]
        #计如：3——>[1, 2], 就是freqSet支持度除以3的支持度
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


if __name__ == "__main__":
    dataSet = loadDataSet()
    # C1 = createC1(dataSet)
    # D = list(map(set, dataSet))
    # L1, suppData0 = scanD(D, C1, 0.5)
    # print(L1, suppData0,sep = '\n')
    L, suppData = apriori(dataSet)
    print(L ,suppData, sep='\n')
    rules = generateRules(L, suppData, minConf = 0.5)
    print(rules)

    #毒蘑菇的相似特征：
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L, suppData = apriori(mushDatSet, minSupport=0.3)
    for item in L[1]:
        if item.intersection('2'): print(item)