#coding:utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
# pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

import pandas as pd
from pandas.api.types import is_string_dtype,is_numeric_dtype,is_datetime64_dtype
import datetime
from utils.config_file import *

def value_range_describe(data):
    counts = [[], [], []]
    cols = data.columns
    for c in cols:
        typ = data[c].dtype
        uniq = len(list(data[c].unique()))
        if uniq == 1:  # uniq==1说明该列只有一个数值
            counts[0].append(c)
        elif uniq == 2 and typ == np.int64:  # uniq==2说明该列有两个数值，往往就是0与1的二类数值
            counts[1].append(c)
        elif typ != np.float64:
            counts[2].append(c)

    print('Constant features: {}\n Binary features: {} \nCategorical features: {}\n'.format(*[len(c) for c in counts]))
    print('Constant features:', counts[0])
    print('Binary features:',counts[1])
    print('Categorical features:', counts[2])
    pal = sns.color_palette()

    for c in counts[2]:
        value_counts = data[c].value_counts(dropna=False)
        if len(value_counts)>20:
            print("字段%s类别数大于20!" %c)
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title('Categorical feature {} - Cardinality {}'.format(c, len(list(data[c].unique()))))
        plt.xlabel('Feature value')
        plt.ylabel('Occurences')
        plt.bar(range(len(value_counts)), value_counts.values, color=pal[1])
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation='vertical')
        plt.savefig('../datasource/fig/%s.png' %(c))
        plt.show()


def value_type_describe(data):
    cols = [c for c in data.columns]  # 返回数据的列名到列表里
    print('Number of features: {}'.format(len(cols)))
    print('Feature types:')
    print(data[cols].dtypes.value_counts())



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
    '''
    统计输入数据表中包含缺失值的列，返回缺失值列名列表
    :param data:
    :return:
    '''
    nanCols = []
    for col in data.columns:
        colTotal = data[col].isnull().sum()
        if colTotal>0:
            print("%s字段缺失值数量为%f：" %(col,colTotal))
            nanCols.append(col)
    return nanCols



def data_split(X,y,feature_name,label_name,cons=[]):
    '''
    分离数据集,将特殊值与待分箱数据分开
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
    '''
    将带'%'的数值字符转为数值
    :param data:
    :param input_cols:
    :return:
    '''
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





def perform_data_transfer(data,transfer_map):
    '''
    对data按照transfer_map中每列设定的转换规则进行数据转换
    :param data:
    :param transfer_map:
    :return:
    '''
    for key,value in transfer_map.items():
        if key == 'change_percent_to_num':
            change_percent_to_num(data,value)
        if key == 'apply_map':
            for itemKey,itemValue in value.items():
                data[itemKey] = data[itemKey].apply(apply_map, args=(itemValue,))





def perform_data_fillna(data):
    '''
    对缺失值进行填充
    :param data:
    :return:
    '''
    nanCols = count_nan(data)  # 缺失值统计
    fillnaMap = {}
    numColWithNan, strColWithNan = num_str_split(data, nanCols)
    for col in numColWithNan:
        fillnaMap[col] = CUSTOMIZE_NUM_VALUE
    for col in strColWithNan:
        fillnaMap[col] = CUSTOMIZE_STR_VALUE
    print("进行缺失值填充的字段-值映射字典：\n", fillnaMap.items())
    data.fillna(value=fillnaMap, inplace=True)
    return data



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
    data = pd.read_csv('../datasource/ms_credit_due.csv')
    print(data.head(20))
    print(data.shape)
    print(data.info())
    uselessCols = ['id_card', 'id','hobby','appl_sbm_tm']
    data.drop(uselessCols,axis=1,inplace=True)
    data,delCols = feature_ratio_filter(data)
    print("单一值占比检测被排除的列：",delCols)
    print("数据表size:",data.shape)
    value_type_describe(data)
    # value_range_describe(data)

    # 字段转换
    # 'sex' 字段缺失值用'保密'替换
    data['sex'].fillna('保密',inplace=True)
    # 'birthday'
    data['birthday'].fillna('0000-00-00', inplace=True)
    data['birthday'].replace(r'(^\d{1,2}-.*)|(^0\d*-.*-.*)|(^\D.*)|(^1[0-8][0-9][0-9]-.*)|(^19[0-5][0-9]-.*)|(^200[3-9]-.*)|(^20[1-9][0-9]-.*)', '0000-00-00', regex=True, inplace=True) #匹配无效数据,及（1999年以前及2003年以后数据）
    data['birthday'].replace(r'(^\d{2})\D.*' ,r'19\1-0-0',regex=True,inplace=True)#匹配不规范年份缩写
    data['birthday'] = data['birthday'].str.extract(r"(^\d{4})")
    data['birthday'] = data['birthday'].apply(lambda x: int(x))
    data['age'] = datetime.date.today().year - data['birthday']
    value_counts = data['age'].value_counts()






    print(data.shape)














