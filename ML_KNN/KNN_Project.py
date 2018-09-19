from numpy import *
from os import listdir
from KNN import classify0, autoNorm

#将矩阵的每行数据放到第一行
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(linStr[j])
    return returnVect
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')        #得到文件夹下的所有文本的文本名
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])        #得到文本中表示的是哪个数字，在文件名中获取(标签)
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/{}'.format(fileNameStr))
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('预测为数字为：{},真实数字为{}'.format(classifierResult, classNumStr))
        if (classifierResult !=classNumStr):
            errorCount += 1
    print('错误率为：{}'.format(errorCount/float(mTest)))
if __name__ == '__main__':
    handwritingClassTest()

#本次实验，我们可以用于识别二维码。
    # 先将二维码转换成灰度图，变成0,1矩阵。然后运用该方法进行训练和二维码识别