import numpy as np
import pandas as pd
import os
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.os_util import Logger
from utils.config_file import *
from utils.write_data import save_obj,load_obj
from model_utils.model_train import forward_selection

def iv_coef_filter(iv_dict,woe_data,threshold=0.7):
    '''
    进行iv及相关性筛选，iv>0.02,相关性阈值0.7，返回满足条件的woe编码列
    :param iv_dict: 变量的iv值数据df feature|iv
    :param woe_data: 变量woe编码数据df,包含target列
    :param threshold: 相关系数阈值
    :return:woe_data里需要保存的数据列
    '''
    high_IV = {k: v for k, v in iv_dict.items() if v >= 0.02}
    remainCols = list(high_IV.keys()) #原始
    remainRecord = remainCols.copy()
    woeDataCols = [col + '_woe' for col in remainCols]
    coefData = woe_data[woeDataCols]
    correlation_matrix = np.corrcoef(coefData, rowvar=0)  # 相关性分析
    correlation_matrix_abs = np.abs(correlation_matrix)
    # 记录已删除的字段索引
    deleteIndex = []
    delCols = []
    for i in range(correlation_matrix_abs.shape[0]):
        if i in deleteIndex:
            continue
        for j in range(correlation_matrix.shape[1]):
            if j in deleteIndex:
                continue
            if i != j and correlation_matrix_abs[i, j] > threshold:
                # 比较相关性较高的两个字段的iv值，保留iv值较大的字段
                feature_i = remainRecord[i]
                feature_j = remainRecord[j]
                iv_i = high_IV[feature_i]
                iv_j = high_IV[feature_j]
                print("%s与%s的相关系数为：%f" %(feature_i,feature_j,correlation_matrix_abs[i, j]))
                if iv_i >= iv_j:
                    deleteIndex.append(j)
                    delCols.append(feature_j)
                    print("变量%s IV:%f >= 变量%s IV:%f,删除变量%s" %(feature_i,iv_i,feature_j,iv_j,feature_j))
                else:
                    deleteIndex.append(i)
                    delCols.append(feature_i)
                    print("变量%s IV:%f < 变量%s IV:%f,删除变量%s" % (feature_i, iv_i, feature_j, iv_j, feature_i))
                    break
    delCols = [x+'_woe' for x in delCols]

    woe_data.drop(delCols,axis=1,inplace=True)
    return woe_data



def vif(woe_data, targetName,threshold=10.0):
    '''
    进行vif相关性分析，返回满足条件的woe编码列
    :param woe_data:
    :param targetName:
    :param threshold:
    :return:
    '''
    target = woe_data[targetName]
    woe_data.drop([targetName],axis=1,inplace=True)
    delCols = []
    cols = list(range(woe_data.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(woe_data.iloc[:, cols].values, ix)
               for ix in range(woe_data.iloc[:, cols].shape[1])]
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > threshold:
            print("删除变量%s,vif=%f" %(woe_data.columns[cols[maxix]],maxvif))
            delCols.append(woe_data.columns[cols[maxix]])
            del cols[maxix]
            dropped = True
    woe_data.drop(delCols, axis=1, inplace=True)
    woe_data = pd.concat([woe_data,target],axis=1)

    return woe_data

def woedata_filter(woeData):
    '''
    对训练用woedata数据进行过滤
    :param woeData: 包含目标列的数据表
    :return: 返回满足条件的列
    '''
    # woe变量筛选
    featureIV = load_obj(FEATURE_IV)
    woeData = iv_coef_filter(featureIV, woeData, threshold=0.7)
    woeData = vif(woeData, targetName=TARGET)
    aicSelectedCols = forward_selection(woeData, TARGET, 'AIC')
    bicSelectedCols = forward_selection(woeData, TARGET, 'BIC')
    # intersection
    intersection = list(set(aicSelectedCols).intersection(set(bicSelectedCols)))
    # union
    union = list(set(aicSelectedCols).union(set(bicSelectedCols)))
    if len(intersection) < 10:
        featureIV = load_obj(FEATURE_IV)
        candidate = list(set(union).difference(set(intersection)))
        candidateRaw = [col[:-4] for col in candidate]  # 去除"_woe"
        candidateIV = {k: v for k, v in featureIV.items() if k in candidateRaw}
        candidateList = sorted(candidateIV.items(), key=lambda x: x[1], reverse=True)
        candidateList = [item[0] + '_woe' for item in candidateList]  # 添加"_woe"
        while len(intersection) < 10:
            appendCol = candidateList.pop(0)
            intersection.append(appendCol)
    selectedCols = intersection
    selectedCols.append(TARGET)
    return woeData[selectedCols]






if __name__=='__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(FEATURE_FILTER_LOG_FILE)
    # 读入woe数据
    woeData = pd.read_csv(WOE_DATA_PATH)
    remainCols = woedata_filter(woeData)
    # 保存入模字段列表
    save_obj(remainCols, path=MODEL_FEATURE)












