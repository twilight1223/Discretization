import pandas as pd
from pandas.api.types import is_string_dtype
#显示所有列
pd.set_option('display.max_columns', None)
import numpy as np

import os
import sys

from feature_dsct_utils.Dependency_dsct import *
from feature_dsct_utils.dsct_tools import *
from eda_utils.eda_tools import *
from utils.os_util import Logger


if __name__ == "__main__":
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('../log_file/default.txt')

    data = pd.read_csv('../datasource/credit_samples.csv')
    TARGET = 'loan_status'
    CUSTOMIZE_NUM_VALUE = -9999  # 自定义缺失值填充
    CUSTOMIZE_STR_VALUE = 'unknown'
    SUMMARY_REPORT_PATH = '../datasource/bin_woe_report.csv'
    DATA_BIN_WOE_PATH = '../datasource/data_bin_woe.csv'
    FEATURE_IV_PATH = '../datasource/feature_iv.csv'
    WOE_DATA_PATH = '../datasource/woe_data.csv'

    '''
    1.字段EDA
    ① 去除id列
    ② 过滤掉单值占比大于90%的字段
    ③ 进行数据类型转换，百分号字符转为float
    ④ 自定义缺失值进行填充 连续型字段采用-9999，离散型字段采用'unknown'
    ⑤ 采用随机森林进行变量筛选，挑出排前50%的字段
    '''
    print("原始数据信息：\n",data.describe())
    #去除id列，对剩余列进行EDA
    target = data[TARGET]
    data.drop(columns=['id',TARGET],axis=1,inplace=True)
    # 过滤掉单值占比大于90%的字段
    data,delCols = feature_ratio_filter(data)
    print("滤除单值占比大于90%的字段：\n",delCols)

    # 数据转换
    input_cols = ['revol_util']
    data = change_percent_to_num(data,input_cols)
    # 缺失值填充
    nanCols = count_nan(data)#缺失值统计
    fillnaMap = {}
    numColWithNan, strColWithNan = num_str_split(data,nanCols)
    for col in numColWithNan:
        fillnaMap[col] = CUSTOMIZE_NUM_VALUE
    for col in strColWithNan:
        fillnaMap[col] = CUSTOMIZE_STR_VALUE
    print("进行缺失值填充的字段-值映射字典：\n",fillnaMap.items())
    data.fillna(value=fillnaMap,inplace=True)
    # 采用随机森林进行变量筛选
    data,delCols = feature_rf_filter(data,target) #当前剩余50个字段
    print("滤除随机森林重要性排名在前50之后的字段：\n", delCols)
    '''
    2.对字段进行卡方离散化，按照binrate>5%,badRate单调且不为0/1,进行分箱微调
    '''
    SPECIAL_ATTRIBUTE = [CUSTOMIZE_NUM_VALUE, CUSTOMIZE_STR_VALUE]
    # 报告合并数组
    RAW_COLS = data.columns
    woe_data = pd.DataFrame(columns=[col+'_woe' for col in RAW_COLS])
    objs = []
    featureIv = pd.DataFrame(columns=['feature','iv'])
    # testCols = ['last_pymnt_d']#['grade','loan_amnt']#'loan_amnt', 'term', 'int_rate','installment', 'grade','revol_util','bc_util'
    print("---------------------开始进行卡方分箱--------------------------")
    for col in data.columns:
        feature_type = 1 if is_string_dtype(data[col]) else 0
        special_data,spacial_y,bin_data,bin_y = data_split(data,target,col,TARGET,cons=SPECIAL_ATTRIBUTE)
        group = Chi_Discretization(bin_data, bin_y, col, max_interval=5, binning_method='chiMerge', feature_type=feature_type)
        # 分箱微调
        group, labels = woeConsMerge(group, bin_data, bin_y, col, max_interval=5, feature_type=feature_type)
        # 分箱报告
        woeSummary, binRangeMap, binWoeMap,iv = woe_summary_report(special_data,spacial_y, group, labels, bin_data, bin_y, col,
                                                                feature_type=feature_type)
        # 数据映射
        data[col + '_bin'] = data[col].apply(convert_raw_to_bin, args=(binRangeMap,feature_type,))
        data[col + '_woe'] = data[col + '_bin'].apply(convert_bin_to_woe, args=(binWoeMap,))
        # 将woe数据结果保存在一张新表中
        woe_data[col + '_woe'] = data[col + '_woe']
        woe_data[TARGET] = data[TARGET]
        # 拼接woe报告结果
        objs.append(woeSummary)
        # 保存字段iv值
        featureIv = featureIv.append(pd.DataFrame({'feature': [col], 'iv': [iv]}),ignore_index=True)
    print("---------------------卡方分箱及woe调整完成--------------------------")
    result = pd.concat(objs, axis=0, ignore_index=True)
    result.to_csv(SUMMARY_REPORT_PATH,index=None)
    print("数据分箱报告保存至：",SUMMARY_REPORT_PATH)

    data.to_csv(DATA_BIN_WOE_PATH,index=None)
    print("数据映射结果保存至：", DATA_BIN_WOE_PATH)
    woe_data.to_csv(WOE_DATA_PATH,index=None)
    print('woe分箱数据保存至：',WOE_DATA_PATH)
    featureIv.to_csv(FEATURE_IV_PATH,index=None)
    print("字段iv值结果保存至：", DATA_BIN_WOE_PATH)














