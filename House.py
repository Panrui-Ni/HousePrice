from sklearn.ensemble import RandomForestRegressor ##缺失值处理
from sklearn import linear_model ## 线性回归
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from scipy import stats
from scipy.special import boxcox1p

#from xgboost import XGBClassifier
import sklearn.preprocessing as preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

train = pd.read_csv("C:/pythonProjecthouse/Data/train.csv")
test = pd.read_csv("C:/pythonProjecthouse/Data/test.csv")

## 先把 Id 一栏去掉，没用
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


## 根据图表分析出和 SalePrice 相关性最大的为 TotalBsmtSF, GrLivArea, Neighborhood 三个特征
train = train.drop(train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<200000)].index)
train = train.drop(train[(train['GrLivArea']>4000) & (train['GrLivArea']<200000)].index) ## 去除异常值

#print(train.shape)

#train_missing = train.isnull().sum() ## 统计各特征缺失值个数，无缺失值即为0
#train_missing = train_missing.drop(train_missing[train_missing == 0].index).sort_values() ## 把0全部去掉，再排序输出

## 下面补缺失值

# 用 None 补非数值型缺失值
none_lists = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'MasVnrType']
for i in none_lists:
    train[i] = train[i].fillna('None')
    test[i] = test[i].fillna('None')

# 用出现频率最高的值补非数值缺失值
most_lists = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual', 'Electrical']
for i in most_lists:
    train[i] = train[i].fillna(train[i].mode()[0])
    test[i] = test[i].fillna(test[i].mode()[0])

# 特殊缺失值
train['Functional'] = train['Functional'].fillna('Typ')
test['Functional'] = test['Functional'].fillna('Typ')

# 去掉没用的特征
train.drop('Utilities', axis=1, inplace=True)
test.drop('Utilities', axis=1, inplace=True)

# 对特定特征补0
zero_lists = ['GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
for i in zero_lists:
    train[i] = train[i].fillna(0)
    test[i] = test[i].fillna(0)

train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
# 将 LotFrontage 按 neighborhood 分组，再用每组的中位数补上缺失值
for i in test['LotFrontage'][test['LotFrontage'].isnull().values==True].index: # 测试集中 LotFrontage 为空
    x = test['Neighborhood'].loc[i] # x 记录 LotFrontage 为空的 Neighborhood 的值
    test['LotFrontage'].loc[i] = train.groupby('Neighborhood')['LotFrontage'].median()[x] # 用训练集中得到的均值按测试集中 Neighborhood 的值补测试集中的缺失值

#print(train.isnull().sum().any()) # 检查 train 里面有没有缺失值
#print(test.isnull().sum().any())

cate_features = [] # 类别特征
num_featues = [] # 数字特征
for i in test.columns:
    if test[i].dtype == 'object':
        cate_features.append(i)
    else: num_featues.append(i)

for i in cate_features:
    train[i] = train[i].astype(str)
    test[i] = test[i].astype(str)

le_features = ['Street', 'Alley', 'LotShape', 'LandContour', 'LandSlope', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
for i in le_features:
    encoder = LabelEncoder()
    value_train = set(train[i].unique()) # train[i] 中出现的不同类的值
    value_test = set(test[i].unique())
    value_list = list(value_train | value_test) # 生成 List
    encoder.fit(value_list)
    train[i] = encoder.transform(train[i]) # 将类别特征转换为编码
    test[i] = encoder.transform(test[i])

## 处理偏斜度较大的特征
skewness = train[num_featues].apply(lambda x: skew(x)).sort_values(ascending=False) # 计算各数字特征的偏斜度
skewness = skewness[skewness > 0.5] # 偏斜度大于 0.5 的
skew_features = skewness.index # 提取偏斜度大于 0.5 的类别名
for i in skew_features:
    lam = stats.boxcox_normmax(train[i]+1) # 利用 boxcox 处理，+1保证输入值是正的
    train[i] = boxcox1p(train[i], lam)
    test[i] = boxcox1p(test[i], lam)

## 构建新特征
train['IsRemod'] = 1
train['IsRemod'].loc[train['YearBuilt'] == train['YearRemodAdd']] = 0 # YearBuilt=YearRemodAdd 表明没有翻新

train['BltRemodDiff'] = train['YearRemodAdd'] - train['YearBuilt'] # 翻新与建造的时间差
train['BsmtUnfRatio'] = 0
train['BsmtUnfRatio'].loc[train['TotalBsmtSF'] != 0] = train['BsmtUnfSF'] / train['TotalBsmtSF'] # Basment未完成的面积占总面积的比例
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF'] # 总面积
#对测试集做同样处理
test['IsRemod'] = 1
test['IsRemod'].loc[test['YearBuilt'] == test['YearRemodAdd']] = 0 # YearBuilt=YearRemodAdd 表明没有翻新

test['BltRemodDiff'] = test['YearRemodAdd'] - test['YearBuilt'] # 翻新与建造的时间差
test['BsmtUnfRatio'] = 0
test['BsmtUnfRatio'].loc[test['TotalBsmtSF'] != 0] = test['BsmtUnfSF'] / test['TotalBsmtSF'] # Basment未完成的面积占总面积的比例
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF'] # 总面积

## 用 dummy (独热编码) 处理剩余的类别特征
dummy_features = list(set(cate_features).difference(set(le_features))) # 获得剩余特征
all_data = pd.concat((train.drop('SalePrice', axis=1),test)).reset_index(drop=True)
all_data = pd.get_dummies(all_data, drop_first=True) # 独热编码去掉第一个分量，保证剩下的变量是独立的

trainset = all_data[:1456]
y = train['SalePrice'].values
trainset['SalePrice'] = y
testset = all_data[1456:]



trainset.to_csv("C:/pythonProjecthouse/Data/train_data.csv")
testset.to_csv("C:/pythonProjecthouse/Data/test_data.csv")