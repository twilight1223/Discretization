from sklearn.model_selection import train_test_split
from model_utils.model_evaluate import *
from model_utils.compute_score import *
from feature_filter import *
from utils.write_data import *
from utils.config_file import *
from feature_dsct_utils.Basic_dsct import *
import statsmodels.formula.api as smf
from model_predict import credit_predict

def train():
    # 读入woe数据
    woeData = pd.read_csv(WOE_DATA_PATH)
    # 数据拆分出训练集，验证集
    trainData, valData = train_test_split(woeData, train_size=0.8, random_state=1234)
    # 变量筛选
    trainData = woedata_filter(trainData)
    selectedCols = list(trainData.columns)
    selectedCols.remove(TARGET)
    save_obj(selectedCols,path=MODEL_FEATURE)

    # 模型训练
    formula = "{}~{}".format(TARGET, "+".join(selectedCols))
    print("final formula is {}".format(formula))
    clf = smf.logit(formula=formula, data=trainData).fit()
    print(clf.summary())

    # 模型评估
    valTarget = valData[TARGET]
    valData.drop([TARGET],axis=1,inplace=True)
    valPred = clf.predict(valData)
    print("验证集auc:", roc_auc_score(valTarget, valPred))

    # 保存模型
    save_obj(clf, path=MODEL)

    '''
    计算模型入模变量统计信息
    '''
    trainTarget = trainData[TARGET]
    trainData.drop([TARGET], axis=1, inplace=True)
    # 计算变量相关系数矩阵
    correlationMatrix = np.corrcoef(trainData, rowvar=0)  # 相关性分析
    coefMDict = {}
    for index,col in enumerate(selectedCols):
        coefMDict[col] = correlationMatrix[:,index]
    coefMDf = pd.DataFrame(coefMDict,index=selectedCols)

    # 保存相关系数/pvalue/vif
    colsIndex = list(range(trainData.shape[1]))
    vif = [variance_inflation_factor(trainData.iloc[:, colsIndex].values, ix)
           for ix in range(trainData.iloc[:, colsIndex].shape[1])]
    vif.insert(0,'')
    coefPvalue = {'coef': clf.params, 'pvalue': clf.pvalues,'vif':vif}
    coefPvalue = pd.DataFrame(coefPvalue)

    # 保存训练woe数据及预测结果
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
    scoreBin,scoreBinRangeMap = score_bin_report(scoreData, 'score', trainTarget, method='equal_distance', bin_num=10)

    save_obj(scoreBinRangeMap,path=SCORE_BIN_RANGE)

    return coefPvalue,scoreReport,scoreBin,coefMDf



if __name__=='__main__':
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(MODEL_TRAIN_LOG_FILE)
    coefPvalue,scoreReport,trainScoreBin,coefMatrixDf = train()
    testScoreBin = credit_predict()
    PSI = pd.DataFrame(data={'PSI':sum((trainScoreBin['bin_rate']-testScoreBin['bin_rate'])*np.log(trainScoreBin['bin_rate']/testScoreBin['bin_rate']))},index=[0])

    #保存结果
    writer = pd.ExcelWriter(FINAL_RESULTS_PATH)
    trainScoreBin.to_excel(writer, sheet_name='训练集score分箱',index=False)
    testScoreBin.to_excel(writer, sheet_name='测试集score分箱',index=False)
    scoreReport.to_excel(writer,sheet_name="变量分箱表",index=False)
    coefPvalue.to_excel(writer,sheet_name="回归系数表",index=True)
    coefMatrixDf.to_excel(writer,sheet_name="相关系数表",index=True)
    PSI.to_excel(writer,sheet_name="PSI",index=False)
    writer.save()
    writer.close()























