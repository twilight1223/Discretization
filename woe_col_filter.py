import numpy as np
import pandas as pd
import os
import sys
from utils.os_util import Logger
from utils.config_file import *
from utils.write_data import save_obj,load_obj
from eda_utils.feature_filter import *
from model_utils.model_train import forward_selection

if __name__=='__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(FEATURE_FILTER_LOG_FILE)
    # 读入woe数据
    woeData = pd.read_csv(TRAIN_WOE_DATA_PATH)
    # woe变量筛选
    featureIV = load_obj(FEATURE_IV)
    woeData = iv_coef_filter(featureIV, woeData,threshold=0.7)
    woeData = vif(woeData,targetName=TARGET)
    aicSelectedCols = forward_selection(woeData, TARGET, 'AIC')
    bicSelectedCols = forward_selection(woeData, TARGET, 'BIC')
    # intersection
    intersection = list(set(aicSelectedCols).intersection(set(bicSelectedCols)))
    # union
    union = list(set(aicSelectedCols).union(set(bicSelectedCols)))
    if len(intersection)<10:
        featureIV = load_obj(FEATURE_IV)
        candidate = list(set(union).difference(set(intersection)))
        candidateRaw = [col[:-4] for col in candidate]#去除"_woe"
        candidateIV = {k: v for k, v in featureIV.items() if k in candidateRaw}
        candidateList = sorted(candidateIV.items(), key=lambda x: x[1],reverse=True)
        candidateList = [item[0]+'_woe' for item in candidateList]#添加"_woe"
        while len(intersection)<10:
            appendCol = candidateList.pop(0)
            intersection.append(appendCol)
    selectedCols = intersection
    # 保存入模字段列表
    save_obj(selectedCols, path=MODEL_FEATURE)