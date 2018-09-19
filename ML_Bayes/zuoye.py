from numpy import *
import pandas as pd


def loadDataSet(fileName):
    data = pd.read_csv(fileName, names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'G'])
    y_data = data['G']
    X_data = data.drop('G',1)
    return X_data, y_data

def DJ(x, mean, S, P):
    d = (x-mean).T *S.I*(x -mean) - 2*log(P)
    return d
def SetS(X_data, y_data):
    X2_data = X_data[nonzero(y_data-1)[1], :]
    X1_data = X_data[nonzero(y_data-2)[1], :]
    X1_mean = mean(X1_data, 0)
    X2_mean = mean(X2_data, 0)
    # m1,n1 = shape(X1_data)
    # m2,n2 = shape(X2_data)
    # ones1 = mat(ones((m1, 1)))
    # ones2 = mat(ones((m2, 1)))
    # dist1 = X1_data - ones1*X1_mean
    # dist2 = X2_data - ones2*X2_mean
    # A = dist1.T * dist1/19
    # B = dist2.T * dist2/6
    A1 = cov(X1_data, rowvar=0)
    B1 = cov(X2_data, rowvar=0)
    print(A1, B1, sep='\n')
    return (19*A1+6*B1)/(20+7-2), X1_mean, X2_mean,X1_data, X2_data

def GP(x,S, X1_mean, X2_mean, p1=20/27, p2=7/27):
    d1 = (x-X1_mean)*mat(S).I*(x-X1_mean).T - 2*log(p1)
    d2 = (x-X2_mean)*mat(S).I*(x-X2_mean).T - 2*log(p2)
    print(d1,d2)
    P = ((exp(-0.5*d1))+(exp(-0.5*d2)))
    P1 = (exp(-0.5*d1))/P
    P2 = (exp(-0.5*d2))/P
    return (P1, P2)
if __name__ == '__main__':
    x, y = loadDataSet('train.csv')
    x = mat(x.values)
    y = mat(y.values)
    S, X1_mean, X2_mean, X1_data, X2_data = SetS(x, y)
    print(X1_mean, X2_mean)
    print('S:',S,sep='\n')
    print("协方差与均值1乘积:", mat(S).I*X1_mean.T)
    print("协方差与均值2乘积:", mat(S).I*X2_mean.T)
    x1 = [7.94, 39.65, 20.97, 20.82, 22.52, 12.41, 1.75, 7.90]
    x2 = [8.28, 64.34, 8.00, 22.22, 20.06, 15.12, 0.72, 22.89]
    x3 = [12.47, 76.39, 5.52, 11.24, 14.52, 22.00, 5.46, 25.50]
    print("西藏：", GP(mat(x1), S, X1_mean, X2_mean))
    print("上海：", GP(mat(x2), S, X1_mean, X2_mean))
    print("广东：", GP(mat(x3), S, X1_mean, X2_mean))
