import pandas as pd
import numpy as  np

print(pd.__version__)
arr = [0, 1, 2, 3, 4]
s1 = pd.Series(arr)
print(s1)

n = np.random.random(5)
index = ['a', 'b', 'c', 'd', 'e']
s2 = pd.Series(index)
print(s2)
d = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5}
s3 = pd.Series(d)
print(s3)
s1.index(list('ABCD'))
print(s1)
s4 = s1.append(s3)
print(s4)
s4 = s4.drop('e')
s4['A'] = 6
print(s4)
print(s4['B'], s4[:3])
print(s4.add(s3), s4.sub(s3), s4.mul(s3), s4.div(s3))
print(s4.median(), s4.sum(), s4.max(), s4.min())
#DataFrame:
dates = pd.date_range('today', periods=6)   #定义时间序列
num_arr = np.random.random(6, 4)    #传入numpy数组
columns = list('ABCD')  #列表名
df1 = pd.DataFrame(num_arr, columns=columns, index=dates)
print(df1)
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = list('abcefightj')
df2 = pd.DataFrame(data, index=labels)
print(df2, df2.dtypes, df2.head(), df2.tail(3), df2.values, df2.describe(),df2.T,  sep = '\n')
print(df2.sort_values(by='age'), df2[1:3], df2['age'], df2[['age', 'animal']], df2.iloc[1:3], sep = '\n')
df3 = df2.copy()
print(df3.isnull())
num = pd.Series([i for i in range(10)], index=df3.index)
df3['NO.'] = num
df3.iat[1, 0] = 2.0 #根据 DataFrame 的下标值进行更改
df3.loc['f', 'age'] = 1.5     #据 DataFrame 的标签对数据进行修改
print(df3, df3.mean(), df3['visits'].sum())
#字符串操作：
string = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(string, string.lower(), string.upper())
df4 = df3.copy()
print(df4)
print(df4.fillna(value=3))
df5 = df3.copy()
print(df5, df5.dropna(how = 'any')) #任何存在nan行都删除
left = pd.DataFrame({'key': ['foo1', 'foo2'], 'one': [1, 2]})
right = pd.DataFrame({'key': ['foo2', 'foo3'], 'two': [4, 5]})
print(left, right, pd.merge(left, right, on='key'))





