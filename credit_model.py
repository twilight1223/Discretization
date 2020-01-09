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
import statsmodels.formula.api as smf
from model_predict import credit_predict

def train():
    # 读入woe数据
    woeData = pd.read_csv(TRAIN_WOE_DATA_PATH)
    # 读入筛选好的变量
    selectedCols = load_obj(MODEL_FEATURE)
    # 数据拆分出训练集，验证集
    X = woeData[selectedCols]
    y = woeData[TARGET]
    trainData, testData, trainTarget, testTarget = train_test_split(X, y, train_size=0.8, random_state=1234)
    # 计算变量相关系数
    correlationMatrix = np.corrcoef(trainData, rowvar=0)  # 相关性分析
    coefMDict = {}
    for index,col in enumerate(selectedCols):
        coefMDict[col] = correlationMatrix[:,index]
    coefMDf = pd.DataFrame(coefMDict,index=selectedCols)
    # 模型训练
    formula = "{}~{}".format(TARGET, "+".join(selectedCols))
    print("final formula is {}".format(formula))
    trainData[TARGET] = trainTarget
    clf = smf.logit(formula=formula, data=trainData).fit()
    print(clf.summary())

    # 模型评估
    testPred = clf.predict(testData)
    print("验证集auc:", roc_auc_score(testTarget, testPred))

    # 保存模型
    save_obj(clf, path=MODEL)

    # 保存相关系数及Pvalue
    coefPvalue = {'coef': clf.params, 'pvalue': clf.pvalues}
    coefPvalue = pd.DataFrame(coefPvalue)

    # 保存训练woe数据及预测结果
    trainData.drop([TARGET], axis=1, inplace=True)
    trainPred = clf.predict(trainData)
    trainData['pred'] = trainPred
    trainData.to_csv(MODEL_TRAIN_RESULTS, index='index')

    # 调整score分布
    odds = trainPred / (1 - trainPred)
    basePoint = 500
    pdo = 20
    A, B = turn_AB(odds, basePoint, pdo)

    # 分配评分
    coefficients = clf.params.iloc[1:]
    intercept = clf.params.iloc[0]
    binWoeReport = pd.read_csv(BIN_WOE_REPORT_PATH)
    scoreReport = credit_card(coefficients, intercept, selectedCols, binWoeReport, A, B)
    scoreData = pd.DataFrame(columns=[col + '_score' for col in selectedCols])
    # 保存入模变量 woe值到score的映射
    WoeScoreDict = {}
    for col in selectedCols:
        woeScoreMap = {}
        varDf = scoreReport[scoreReport['Var'] == col]
        for row in varDf.values:
            woeScoreMap[row[-3]] = row[-1]#{woe:score}
        # 保存每个变量的映射
        WoeScoreDict[col] = woeScoreMap
        scoreData[col + '_score'] = trainData[col].apply(convert_woe_to_score, args=(woeScoreMap,))
    baseScore = scoreReport.iloc[0, -1]

    # 保存score结果表
    scoreData['score'] = scoreData.apply(lambda x: sum(x.values) + baseScore, axis=1)
    scoreData['p_1'] = trainPred
    scoreData[TARGET] = trainTarget
    scoreData.to_csv(MODEL_TRAIN_SCORE_RESULTS, index='index')
    # 保存woe_score映射
    save_obj(WoeScoreDict, path=WOE_SCORE)

    # 保存评分卡
    scoreReport.to_csv(SCORE_CARD, index=None)

    # 评分分箱
    scoreBin = score_bin_report(scoreData, 'score', trainTarget, method='equal_frequency', bin_num=10)

    return coefPvalue,scoreReport,scoreBin,coefMDf



if __name__=='__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(MODEL_TRAIN_LOG_FILE)
    coefPvalue,scoreReport,trainScoreBin,coefMatrixDf = train()
    testScoreBin = credit_predict()
    #保存结果
    writer = pd.ExcelWriter(FINAL_RESULTS_PATH)
    trainScoreBin.to_excel(writer, sheet_name='训练集score分箱',index=False)
    testScoreBin.to_excel(writer, sheet_name='测试集score分箱',index=False)
    scoreReport.to_excel(writer,sheet_name="变量分箱表",index=False)
    coefPvalue.to_excel(writer,sheet_name="回归系数表",index=True)
    coefMatrixDf.to_excel(writer,sheet_name="相关系数表",index=True)
    writer.save()
    writer.close()























