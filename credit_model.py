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




if __name__=='__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('./log_file/credit_model.txt')
    # 读入woe数据
    woeData = pd.read_csv('./datasource/train_woe_data.csv')
    # woe变量筛选
    featureIV = load_obj('./datasource/obj/featureIV.pkl')
    woeData = iv_coef_filter(featureIV, woeData)
    woeData = vif(woeData)
    TARGET = 'loan_status'
    # aic_selectedCols = forward_selection(woeData, TARGET, 'AIC')
    # bic_selectedCols = forward_selection(woeData,TARGET,'BIC')
    # selectedCols = aic_selectedCols#list(set(aic_selectedCols).intersection(set(bic_selectedCols)))
    # # 保存入模字段列表
    # save_obj(selectedCols,path='./datasource/obj/model_feature.pkl')
    selectedCols = load_obj(path='./datasource/obj/model_feature.pkl')
    # 数据拆分出训练集，验证集
    X = woeData[selectedCols]
    y = woeData[TARGET]
    trainData,testData,trainTarget,testTarget = train_test_split(X,y,train_size=0.8,random_state=1234)
    # 模型训练
    lr = LogisticRegression()
    clf = lr.fit(trainData,trainTarget)
    pred = lr.predict(trainData)
    predArray = lr.predict_proba(trainData)

    # trainData[TARGET] = trainTarget
    trainData['pred'] = pred
    trainData.to_csv('./datasource/model_results.csv',index='index')



    # 保存模型
    save_obj(clf,path='./datasource/model/lr_model.pkl')

    # 模型评估
    testPreds = lr.predict_proba(testData)[:, 1]
    print("auc:", roc_auc_score(testTarget, testPreds))

    # 调整score分布
    basePoint = 500
    pdo = 20
    flag = True
    while flag:
        A, B = alpha_beta(basepoints=basePoint, baseodds=1, pdo=pdo)
        odds = predArray[:,1]/predArray[:,0]
        score = A+B*np.log(odds)
        minScore = min(score)
        maxScore = max(score)
        if (minScore>250 and minScore<400) and (maxScore>700 and maxScore<950):
            flag = False
        elif minScore>400 and maxScore>900:#基础分偏高minScore>400 and
            basePoint = basePoint-50
        elif minScore>400 and maxScore<700:#评分梯度太小
            pdo = pdo + 10
        elif minScore<250 and maxScore<700:# and maxScore<700:#基础分偏低
            basePoint = basePoint + 50
        elif minScore<300 and maxScore>900:
            pdo = pdo - 10
        elif minScore<300 and (maxScore>700 and maxScore<950):
            basePoint = basePoint + 100
        elif (minScore>250 and minScore<400) and maxScore>900:
            basePoint = basePoint - 100
            pdo = pdo + 10







    # 分配评分
    coefficients = clf.coef_[0]
    intercept = clf.intercept_[0]
    binWoeReport = pd.read_csv('./datasource/bin_woe_report.csv')
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

    scoreData.to_csv('./datasource/score_results.csv',index='index')



    # 保存woe_score映射
    save_obj(WoeScoreDict,path='./datasource/obj/woe_score.pkl')

    # 保存评分卡
    scoreReport.to_csv('./datasource/score_card.csv',index=None)


















