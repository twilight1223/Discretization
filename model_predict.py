import pandas as pd
from utils.write_data import *
from eda_utils.eda_tools import *
from utils.config_file import *
from feature_dsct_utils.dsct_tools import *
from model_utils.compute_score import convert_woe_to_score
from feature_dsct_utils.Basic_dsct import score_bin_report
from sklearn.metrics import roc_auc_score


if __name__=='__main__':
    test = pd.read_csv(TEST_DATA_SET)
    target = test[TARGET]
    test.drop(TARGET,axis=1,inplace=True)
    # 数据预处理
    if DATA_TRANSFER:
        perform_data_transfer(test,DATA_TRANSFER)
    # 导入入模变量,带"_woe"
    features = load_obj(MODEL_FEATURE)
    rawFeatures = [col[:-4] for col in features]
    # 进行数据映射
    binRange = load_obj(BIN_RANGE)
    binWoe = load_obj(BIN_WOE)
    woeScore = load_obj(WOE_SCORE)
    for col in rawFeatures:
        feature_type = 1 if is_string_dtype(test[col]) else 0
        binRangeDict = binRange[col]
        binWoeDict = binWoe[col]
        woeScoreDict = woeScore[col+'_woe']
        test[col + '_bin'] = test[col].apply(convert_raw_to_bin, args=(binRangeDict, feature_type,))
        test[col + '_woe'] = test[col + '_bin'].apply(convert_bin_to_woe, args=(binWoeDict,))
        test[col + '_woe_score'] = test[col + '_woe'].apply(convert_woe_to_score, args=(woeScoreDict,))
    # 截取woe数据
    selectWoeCols = [col+'_woe' for col in rawFeatures]
    woeData = test[selectWoeCols]
    # 载入模型
    model = load_obj(MODEL)
    pred = model.predict(woeData)
    predArray = model.predict_proba(woeData)
    # 模型评估
    testPreds = predArray[:, 1]
    print("验证集auc:", roc_auc_score(target, testPreds))
    #保存测试集预测结果
    woeData['pred'] = pred
    woeData.to_csv(MODEL_TEST_RESULTS,index='index')

    # 截取score数据
    selectScoreCols = [col + '_woe_score' for col in rawFeatures]
    scoreData = test[selectScoreCols]
    scoreCard = pd.read_csv(SCORE_CARD)
    baseScore = scoreCard.iloc[0, 3]
    # 保存score结果表
    scoreData['total_score'] = scoreData.apply(lambda x: sum(x.values) + baseScore, axis=1)
    scoreData['p_1'] = predArray[:, 1]
    scoreData[TARGET] = target
    scoreData.to_csv(MODEL_TEST_SCORE_RESULTS, index='index')

    #分数分箱
    count = score_bin_report(scoreData, 'total_score', target, method='equal_distance', bin_num=10)
    count.to_csv(TEST_SCORE_BIN_RESULTS, index=None)

