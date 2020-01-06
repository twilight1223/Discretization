import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,roc_curve, auc, roc_auc_score,precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from eda_utils.Feature_Coef_Analysis import forward_selection


####################### PlotKS ##########################
def plot_Ks(preds, labels, n,asc=0):
    # preds is score: asc=1
    # preds is prob: asc=0

    pred = preds  # 预测值
    bad = labels  # 取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad

    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))

    # chart
    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
             color='blue', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
             color='red', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.ks, label='ks',
             color='green', linestyle='-', linewidth=2)

    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'], color='red', linestyle='--')
    plt.title('KS=%s ' % np.round(ks_value, 4) +
              'at Pop=%s' % np.round(ks_pop, 4), fontsize=15)
    plt.show()

    return ksds


def plot_Roc_Auc(preds, labels):
    FPR, TPR, threshold = roc_curve(test_target, test_predprob)
    ROC_AUC = auc(FPR, TPR)
    plt.plot(FPR, TPR, 'b', label='AUC = %0.2f' % ROC_AUC)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()



def Prob2Score(prob, basePoint, PDO):
    #将概率转化成分数且为正整数
    y = np.log(prob/(1-prob))
    return (basePoint+PDO/np.log(2)*(-y)).map(lambda x: int(x))

def alpha_beta(basepoints,baseodds,pdo):
    '''
    score = alpha - beta * ln(odds)
    :param basepoints:
    :param baseodds:
    :param pdo:
    :return:
    '''
    beta = pdo/math.log(2)
    alpha = basepoints + beta * math.log(baseodds)
    return alpha,beta

def compute_score(coe, woe, factor):
    scores = []
    for w in woe:
        score = round(coe * w * factor, 0)
        scores.append(score)
    return scores


if __name__=='__main__':
    woe_data = pd.read_csv('./datasource/woe_data_filtered.csv')
    # 1.划分训练集，验证集
    TARGET = 'loan_status'
    train_data,test_data = train_test_split(woe_data,train_size=0.7,random_state=1234)
    selectedCols, model = forward_selection(train_data, TARGET, 'AIC')
    print(selectedCols)
    # 查看所有字段的系数值，除去常数项外，coef
    print(model.summary())

    # 2.模型训练
    lr = LogisticRegression()
    train_target = train_data[TARGET]
    train_data = train_data[selectedCols]
    clf = lr.fit(train_data,train_target)
    pred_array = lr.predict_proba(train_data)

    # 3.模型评估
    test_target = test_data[TARGET]
    test_data = test_data[selectedCols]
    test_predprob = lr.predict_proba(test_data)[:, 1]
    print("auc:", roc_auc_score(test_target, test_predprob))

    # 分配评分
    coefficients = clf.coef_[0]
    intercept = clf.intercept_[0]
    A,B = alpha_beta(500,1,20)

    # 变量集v与对应系数coef组合：v_coef
    feature_coef = {'Var':selectedCols,'coef':coefficients}
    v_coef = pd.DataFrame(feature_coef).reset_index()
    # # 变量集v与对应woe组合：v_woe
    bin_woe_report = pd.read_csv('./datasource/bin_woe_report.csv')
    v_woe = bin_woe_report[['Var', 'bin_range', 'woe']].reset_index(drop=True)
    v_woe['Var'] = v_woe['Var'].apply(lambda x: x + '_woe')
    # v_coef与v_woe组合：v_coef_woe
    v_coef_woe = pd.merge(v_coef, v_woe, how='inner')
    # # 构建新列wf：变量系数coef与变量各分组woe的乘积
    v_coef_woe['wf'] = v_coef_woe['woe'] * v_coef_woe['coef']
    #
    # # 构建新列score_woe:每个分组对应的子评分
    # v_coef_woe['score_woe'] = round(v_coef_woe['wf'] * (-b), 0)  # 变量各分段得分
    #
    # # 合成最终的评分卡：
    # score_0 = round(float(a - b * v_coef.loc[0]['coef']), 0)  # 基准分
    # score_X = v_coef_woe[['Var', 'bin_range', 'woe', 'score_woe']]  # 变量中各分段得分
    # df_score_0 = pd.DataFrame(['init_score', '—', '—', '%.f' % score_0],
    #                           index=['Var', 'bin_range', 'woe', 'score_woe']).T
    # score_card = df_score_0.append(score_X, ignore_index=True)  # 总评分卡
    # score_card['score_woe'] = score_card['score_woe'].astype('float64')
    # score_card  # 输出评分卡









