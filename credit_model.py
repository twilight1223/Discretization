import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from model_utils.model_train import *
from model_utils.model_evaluate import *
from model_utils.compute_score import *
from eda_utils.feature_filter import *
from utils.write_data import *
from utils.config_file import *
from feature_dsct_utils.Basic_dsct import *


if __name__=='__main__':

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(MODEL_TRAIN_LOG_FILE)
    # 读入woe数据
    woeData = pd.read_csv(TRAIN_WOE_DATA_PATH)
    # 读入筛选好的变量
    selectedCols = load_obj(MODEL_FEATURE)

    # 数据拆分出训练集，验证集
    X = woeData[selectedCols]
    y = woeData[TARGET]
    trainData,testData,trainTarget,testTarget = train_test_split(X,y,train_size=0.8,random_state=1234)
    # 模型训练
    lr = LogisticRegression()
    clf = lr.fit(trainData,trainTarget)
    pred = lr.predict(trainData)
    predArray = lr.predict_proba(trainData)
    # 保存模型
    save_obj(clf, path=MODEL)

    # 保存训练数据及预测结果
    # trainData[TARGET] = trainTarget
    trainData['pred'] = pred
    trainData.to_csv(MODEL_TRAIN_RESULTS,index='index')


    # 模型评估
    testPreds = lr.predict_proba(testData)[:, 1]
    print("验证集auc:", roc_auc_score(testTarget, testPreds))

    # 调整score分布
    basePoint = 500
    pdo = 20
    flag = True
    print("开始循环调整分数：")
    while flag:
        A, B = alpha_beta(basepoints=basePoint, baseodds=1, pdo=pdo)
        odds = predArray[:,1]/predArray[:,0]
        score = A+B*np.log(odds)
        minScore = min(score)
        maxScore = max(score)
        print("当前循环：A-%f,B-%f,max_score-%f,min_score-%f" % (A, B, maxScore, minScore))
        if (minScore>250 and minScore<400) and (maxScore>700 and maxScore<950):
            #正常范围分值
            flag = False
        elif ((minScore>250 and minScore<400) and maxScore>=950):
            #分值幅度过高
            pdo = pdo - 10
        elif (minScore <= 250 and maxScore >= 950):
            basePoint = basePoint - 50
            pdo = pdo - 10

        elif (minScore >= 400 and maxScore <= 700):
            #分值幅度过低
            pdo = pdo + 10
        elif (minScore>250 and minScore<400) and maxScore <= 700:
            basePoint = basePoint + 50
            pdo = pdo + 10

        elif minScore<=250 and (maxScore>700 and maxScore<950):
            #基础分过低
            basePoint = basePoint + 50
        elif minScore>=400:
            #基础分过高
            basePoint = basePoint - 50


    # 分配评分
    coefs = pd.DataFrame(columns=['feature','coef'])
    coefficients = clf.coef_[0]
    intercept = clf.intercept_[0]
    for col,coef in zip(selectedCols,coefficients):
        coefs = coefs.append(pd.DataFrame({'feature':[col],'coef':[coef]}),ignore_index=True)
    coefs = coefs.append(pd.DataFrame({'feature':['intercept'],'coef':intercept}))
    coefs.to_csv(COEFICENTS_INTERCEPT,index=None)

    binWoeReport = pd.read_csv(BIN_WOE_REPORT_PATH)
    scoreReport = credit_card(coefficients,intercept,selectedCols,binWoeReport,A,B)
    scoreData = pd.DataFrame(columns=[col + '_score' for col in selectedCols])
    # 保存入模变量 woe值到score的映射
    WoeScoreDict = {}
    for col in selectedCols:
        woeScoreMap = {}
        varDf = scoreReport[scoreReport['Var'] == col]
        for row in varDf.values:
            woeScoreMap[row[2]] = row[3]
        # 保存每个变量的映射
        WoeScoreDict[col] = woeScoreMap
        scoreData[col+'_score'] = trainData[col].apply(convert_woe_to_score,args=(woeScoreMap,))
    baseScore = scoreReport.iloc[0,3]

    #保存score结果表
    scoreData['total_score'] = scoreData.apply(lambda x:sum(x.values)+baseScore,axis=1)
    scoreData['p_1'] = predArray[:,1]
    scoreData[TARGET] = trainTarget
    scoreData.to_csv(MODEL_TRAIN_SCORE_RESULTS,index='index')
    # 保存woe_score映射
    save_obj(WoeScoreDict,path=WOE_SCORE)

    # 保存评分卡
    scoreReport.to_csv(SCORE_CARD,index=None)

    # 评分分箱
    count = score_bin_report(scoreData,'total_score',trainTarget,method='equal_distance',bin_num=10)
    count.to_csv(TRAIN_SCORE_BIN_RESULTS,index=None)





















