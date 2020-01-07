from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype,is_numeric_dtype,is_datetime64_dtype
import datetime



def typeof(variate):
    type=None
    if isinstance(variate,int):
        type = "int"
    elif isinstance(variate,str):
         type = "str"
    elif isinstance(variate,float):
         type = "float"
    elif isinstance(variate,list):
         type = "list"
    elif isinstance(variate,tuple):
         type = "tuple"
    elif isinstance(variate,dict):
         type = "dict"
    elif isinstance(variate,set):
         type = "set"
    return type

def count_nan(data):
    nanCols = []
    for col in data.columns:
        colTotal = data[col].isnull().sum()
        if colTotal>0:
            print("%s字段缺失值数量为%f：" %(col,colTotal))
            nanCols.append(col)
    return nanCols



def data_split(X,y,feature_name,label_name,cons=[]):
    '''
    分离数据集
    :param X:
    :param y:
    :param feature_name:
    :param label_name:
    :param cons:
    :return:
    '''
    X[label_name] = y
    special_data = X.loc[X[feature_name].isin(cons)]
    special_y = special_data[label_name]
    special_data.drop(label_name, axis=1, inplace=True)

    bin_data = X.loc[~X[feature_name].isin(cons)]
    bin_y = bin_data[label_name]
    bin_data.drop(label_name, axis=1, inplace=True)
    return special_data,special_y,bin_data,bin_y




def feature_ratio_filter(data):
    '''
    过滤掉单值占比大于90%的字段
    :param data:
    :return:
    '''
    delCols = []
    for col in data.columns:
        valCount = data[col].value_counts(normalize=True, dropna=False)
        if valCount.iloc[0] > 0.9:
            print("%s字段单一值占比为：%f" %(col,valCount.iloc[0]))
            delCols.append(col)
    # remainCols = list(set(data.columns) ^ set(delCols))
    data.drop(delCols,axis=1,inplace=True)
    return data,delCols

def feature_rf_filter(data,target,f_length=50):
    '''
    采用随机森林进行变量筛选，挑选出重要性排前50的字段
    :param data:
    :param input_cols:
    :param target_col:
    :param f_length:
    :return:
    '''
    # 避免LabelEncoding数据覆盖，创建一个新的dataframe
    df = data.copy(deep=True)

    # 对类别类型进行LabelEncoding
    cols = df.select_dtypes(include=['O']).columns.tolist()
    for col in cols:
        df[col] = preprocessing.LabelEncoder().fit_transform(df[col])
    y = target
    x = df
    clf = RandomForestClassifier(random_state=1234)
    clf.fit(x, y)
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]
    features = x.columns
    for f in range(x.shape[1]):
        print(("%2d) %-*s %f" % (f + 1, 20, features[indices[f]], importance[indices[f]])))

    delCols = []
    if len(features)>=f_length:
        delIndices = indices[f_length:]
        for idx in delIndices:
            delCols.append(features[idx])
    data.drop(delCols,axis=1,inplace=True)
    return data,delCols

def num_str_split(data,input_cols):
    '''
    根据数据类型将原始字段分为连续型和离散型字段分组
    :param data:
    :param input_cols:
    :return:
    '''
    strColNames = []
    numColNames = []

    for col in input_cols:
        # if str in data[col].apply(lambda x: type(x)).unique():
        if is_string_dtype(data[col]):
            strColNames.append(col)
        elif is_numeric_dtype(data[col]):
            numColNames.append(col)
    return numColNames,strColNames




def change_percent_to_num(data,input_cols):
    for col in input_cols:
        data[col] = data[col].str.strip("%").astype(float)/100
    return data

def apply_map(x,map_dict):
    '''
    对series列执行映射，返回值
    :param x:
    :param map_dict:
    :return:
    '''
    for key,value in map_dict.items():
        if x==key:
            return value









def change_str_to_time(data,input_cols,format='%b-%Y'):
    '''
    将日期字符串转化为时间格式
    :param data:
    :param input_cols:
    :param format:
    :return:
    '''
    for col in input_cols:
        data[col] = data[col].apply(datetime.datetime.strptime,args=(format,))
    return data


if __name__=='__main__':
    data = pd.read_csv('../datasource/credit_samples.csv')
    cols = ['last_credit_pull_d','next_pymnt_d']
    data = change_str_to_time(data,cols)
    print(data.shape[1])








