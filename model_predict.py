import pandas as pd
from utils.write_data import *
from eda_utils.eda_tools import *
from utils.config_file import *
from feature_dsct_utils.dsct_tools import *
from model_utils.compute_score import convert_woe_to_score
from feature_dsct_utils.Basic_dsct import score_bin_report
from sklearn.metrics import roc_auc_score

def credit_predict():
    test = pd.read_csv(TEST_DATA_SET)
    target = test[TARGET]
    test.drop(TARGET, axis=1, inplace=True)
    # 数据转换预处理
    if DEL_COLUMNS:
        test.drop(columns=DEL_COLUMNS, axis=1, inplace=True)
    if DATA_TRANSFER:
        perform_data_transfer(test, DATA_TRANSFER)
    # 缺失值填充处理
    test = perform_data_fillna(test)
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
        woeScoreDict = woeScore[col + '_woe']
        test[col + '_bin'] = test[col].apply(convert_raw_to_bin, args=(binRangeDict, feature_type,))
        test[col + '_woe'] = test[col + '_bin'].apply(convert_bin_to_woe, args=(binWoeDict,))
        test[col + '_woe_score'] = test[col + '_woe'].apply(convert_woe_to_score, args=(woeScoreDict,))
    # 截取woe数据
    selectWoeCols = [col + '_woe' for col in rawFeatures]
    woeData = test[selectWoeCols]
    # 载入模型
    model = load_obj(MODEL)
    pred = model.predict(woeData)
    # 模型评估
    testPreds = model.predict(woeData)
    print("验证集auc:", roc_auc_score(target, testPreds))
    # 保存测试集预测结果
    woeData['pred'] = pred
    woeData.to_csv(MODEL_TEST_RESULTS, index='index')

    # 截取score数据
    selectScoreCols = [col + '_woe_score' for col in rawFeatures]
    scoreData = test[selectScoreCols]
    scoreCard = pd.read_csv(SCORE_CARD)
    baseScore = scoreCard.iloc[0, -1]
    # 保存score结果表
    scoreData['score'] = scoreData.apply(lambda x: sum(x.values) + baseScore, axis=1)
    scoreData['p_1'] = testPreds
    scoreData[TARGET] = target
    scoreData.to_csv(MODEL_TEST_SCORE_RESULTS, index='index')

    # 分数分箱
    binRangeMapDict = load_obj(SCORE_BIN_RANGE)
    scoreBinData = scoreData[['score',TARGET]]
    order = ['bin', 'bin_range', 'bin_num', 'bin_good_num', 'bin_bad_num', 'bin_rate', 'good_bin_rate',
             'bad_bin_rate', \
             'bin_rate_cum', 'good_bin_rate_cum', 'bad_bin_rate_cum', 'good_rate', 'bad_rate']


    scoreBinData['bin'] = scoreBinData['score'].apply(convert_score_to_bin, args=(binRangeMapDict,))
    count = pd.crosstab(scoreBinData['bin'], target)
    count['bin_num'] = count.apply(lambda x: x.sum(), axis=1)
    count['bad_rate'] = count[1] / count['bin_num']
    binDescribeDf = count.rename(columns={0: 'bin_good_num', 1: 'bin_bad_num'})
    goodNum = sum(binDescribeDf['bin_good_num'])
    badNum = sum(binDescribeDf['bin_bad_num'])
    totalNum = sum([goodNum, badNum])
    binDescribeDf['bin'] = [i for i in range(count.shape[0])]
    binDescribeDf['bin_range'] = binDescribeDf['bin'].apply(apply_map,args=(binRangeMapDict,))
    binDescribeDf['bin_rate'] = binDescribeDf['bin_num'] * 1.0 / totalNum
    binDescribeDf['good_bin_rate'] = binDescribeDf['bin_good_num'] * 1.0 / goodNum
    binDescribeDf['bad_bin_rate'] = binDescribeDf['bin_bad_num'] * 1.0 / badNum
    binDescribeDf['bin_rate_cum'] = binDescribeDf['bin_rate'].cumsum()
    binDescribeDf['good_bin_rate_cum'] = binDescribeDf['good_bin_rate'].cumsum()
    binDescribeDf['bad_bin_rate_cum'] = binDescribeDf['bad_bin_rate'].cumsum()
    binDescribeDf['good_rate'] = binDescribeDf['bin_good_num'] / binDescribeDf['bin_num']
    scoreBin = binDescribeDf[order].reset_index(drop=True)
    # scoreBin,_ = score_bin_report(scoreData, 'score', target, method='equal_frequency', bin_num=10)
    scoreBin.to_csv(TEST_SCORE_BIN_RESULTS, index=None)
    return scoreBin

if __name__=='__main__':
    credit_predict()



