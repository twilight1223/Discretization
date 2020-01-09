import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
from utils.os_util import Logger
from utils.write_data import *
from eda_utils.eda_tools import *
from feature_dsct_utils.Dependency_dsct import *
from utils.config_file import *

if __name__=='__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    # 记录训练过程print信息
    sys.stdout = Logger(BIN_ENCODE_LOG_FILE)
    #导入数据
    data = pd.read_csv(DATA_SET)
    #划分训练集、测试集
    train,test = train_test_split(data,train_size=0.7,random_state=1234)
    test.to_csv(TEST_DATA_SET,index=None)
    ## ----------------------训练集预处理----------------------------------
    '''
    1.字段EDA
    去除id列
    过滤掉单值占比大于90%的字段
    进行数据类型转换，百分号字符转为float
    自定义缺失值进行填充 连续型字段采用-9999，离散型字段采用'unknown'
    日期转换
    采用随机森林进行变量筛选，挑出排前50的字段
    '''
    print("原始数据信息：\n", train.describe())
    # 去除不参与分析的特征列，对剩余列进行EDA
    target = train[TARGET]
    train.drop(columns=DEL_COLUMNS, axis=1, inplace=True)
    # 过滤掉单值占比大于90%的字段
    train, delCols = feature_ratio_filter(train)
    print("滤除单值占比大于90%的字段：\n", delCols)
    #数据转换
    if DATA_TRANSFER:
        perform_data_transfer(train,DATA_TRANSFER)

    # 缺失值填充
    train = perform_data_fillna(train)
    # 采用随机森林进行变量筛选
    train, delCols = feature_rf_filter(train, target,f_length=30)
    print("滤除随机森林重要性排名在前30之后的字段：\n", delCols)

    ##-------------------------------卡方分箱----------------------------------
    '''
    对字段进行卡方离散化，按照binrate>5%,badRate单调且不为0/1,进行分箱微调
    '''
    SPECIAL_ATTRIBUTE = [CUSTOMIZE_NUM_VALUE, CUSTOMIZE_STR_VALUE]
    # 报告合并数组
    RAW_COLS = train.columns
    woe_data = pd.DataFrame(columns=[col + '_woe' for col in RAW_COLS])
    objs = []
    print("---------------------开始进行卡方分箱--------------------------")
    binRangeMapDict = {}
    binWoeMapDict = {}
    featureIVDict = {}
    testCols = ['apply_credibility','latest_six_month_apply']
    for col in train.columns:
        feature_type = 1 if is_string_dtype(train[col]) else 0
        special_data, spacial_y, bin_data, bin_y = data_split(train, target, col, TARGET, cons=SPECIAL_ATTRIBUTE)
        group = Chi_Discretization(bin_data, bin_y, col, max_interval=5, binning_method='chiMerge',
                                   feature_type=feature_type)
        # 分箱微调
        group, labels = woeConsMerge(group, bin_data, bin_y, col, max_interval=5, feature_type=feature_type)
        # 分箱报告
        woeSummary, binRangeMap, binWoeMap, iv = woe_summary_report(special_data, spacial_y, group, labels, bin_data,
                                                                    bin_y, col,
                                                                  feature_type=feature_type)
        # 保存映射关系字典
        binRangeMapDict[col] = binRangeMap
        binWoeMapDict[col] = binWoeMap
        featureIVDict[col] = iv

        # 数据映射
        train[col + '_bin'] = train[col].apply(convert_raw_to_bin, args=(binRangeMap, feature_type,))
        train[col + '_woe'] = train[col + '_bin'].apply(convert_bin_to_woe, args=(binWoeMap,))
        # 将woe数据结果保存在一张新表中
        woe_data[col + '_woe'] = train[col + '_woe']
        woe_data[TARGET] = train[TARGET]
        # 拼接woe报告结果
        objs.append(woeSummary)
    print("---------------------卡方分箱及woe调整完成--------------------------")
    result = pd.concat(objs, axis=0, ignore_index=True)
    result.to_csv(BIN_WOE_REPORT_PATH, index=None)
    print("数据分箱报告保存至：", BIN_WOE_REPORT_PATH)
    train.to_csv(DATA_BIN_WOE_PATH, index=None)
    print("数据映射结果保存至：", DATA_BIN_WOE_PATH)
    woe_data.to_csv(TRAIN_WOE_DATA_PATH, index=None)
    print('woe分箱数据保存至：', TRAIN_WOE_DATA_PATH)

    save_obj(binRangeMapDict, BIN_RANGE)
    save_obj(binWoeMapDict, BIN_WOE)
    save_obj(featureIVDict,FEATURE_IV)








