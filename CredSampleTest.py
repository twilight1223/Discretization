import pandas as pd
import numpy as np

from feature_dsct_utils.Dependency_dsct import *
from feature_dsct_utils.dsct_tools import *




if __name__ == "__main__":
    data = pd.read_csv('./credit_samples.csv')
    # print(data.shape)
    # dataInfo = data.info()
    # print(data.info())
    # print(data.dtypes)
    print(data.describe())
    SPECIAL_VALUE = -9999
    TARGET = 'loan_status'
    data = data.fillna(SPECIAL_VALUE)
    print(data.info())
    # 1. 数据EDA
    cols = data.drop(columns=['id', TARGET]).columns
    delCols = []
    remainCols = []
    for col in cols:
        valCount = data[col].value_counts(normalize=True,dropna=False)
        if valCount.iloc[0]>=0.9:
            delCols.append(col)
    remainCols = list(set(cols) ^ set(delCols))

    # 2.数据分箱
    strColNames = []
    numColNames = []

    for col in remainCols:
        if str in data[col].apply(lambda x: type(x)).unique():
            strColNames.append(col)
        else:
            numColNames.append(col)

    print(len(strColNames))
    print(len(numColNames))
    # 报告合并数组
    objs = []
    for col in numColNames:
        # 设置特殊属性值分箱
        df1 = data.loc[data[col] == SPECIAL_VALUE]
        df2 = data.loc[~(data[col] == SPECIAL_VALUE)]
        group = Chi_Discretization(df2, col, TARGET, max_interval=10, binning_method='chiMerge', feature_type=0)
        # 分箱微调
        group, labels = woeConsMerge(group, data, col, TARGET, max_interval=5, feature_type=0)
        woeSummary = woe_summary_report(df1, group, labels,df2, col, 'loan_status', feature_type=0)
        groupList = [[group[i],group[i+1]] for i in range(len(group)-1)]
        if df1.shape[0]!=0:
            groupList.append([-9999])
        data[col+'_bin'] = data[col].apply(convert_numraw_to_bin,args=(groupList,))
        binList = woeSummary['bin']
        woeList = woeSummary['woe']
        binWoeMap = {}
        for key,value in zip(binList,woeList):
            binWoeMap[key] = value
        data[col+'_woe'] = data[col+'_bin'].apply(convert_bin_to_woe,binWoeMap)
        objs.append(woeSummary)
    for col in strColNames:
    # 设置特殊属性值分箱
        df1 = data.loc[data[col] == SPECIAL_VALUE]
        df2 = data.loc[~(data[col] == SPECIAL_VALUE)]
        group = Chi_Discretization(df2, col, 'loan_status',max_interval=10, binning_method='chiMerge', feature_type=1)
        # 分箱微调
        group, labels = woeConsMerge(group, data, col, TARGET, max_interval=5, feature_type=1)
        woeSummary = woe_summary_report(df1,group,labels, df2, col, 'loan_status', feature_type=1)
        groupList = group
        if df1.shape[0] != 0:
            groupList.append([-9999])
        data[col + '_bin'] = data[col].apply(convert_str_to_bin, args=(groupList,))
        binList = woeSummary['bin']
        woeList = woeSummary['woe']
        binWoeMap = {}
        for key, value in zip(binList, woeList):
            binWoeMap[key] = value
        data[col + '_woe'] = data[col + '_bin'].apply(convert_bin_to_woe, binWoeMap)
        objs.append(woeSummary)
    result = pd.concat(objs,axis=0,ignore_index=True)
    result.to_csv("./bin_woe_report.csv")
    data.to_csv("./data_samples.csv")
    print(result.shape[0])










