import numpy as np
import pandas as pd
import math

def Prob2Score(prob, basePoint, PDO):
    #将概率转化成分数且为正整数
    y = np.log(prob/(1-prob))
    return int(basePoint+PDO/np.log(2)*(-y))

def compute_score(coe, woe, factor):
    scores = []
    for w in woe:
        score = round(coe * w * factor, 0)
        scores.append(score)
    return scores

def convert_woe_to_score(x,woeScoreMap):
    for key,value in woeScoreMap.items():
        if round(x,8) == round(key,8):
            return value
    else:
        raise ValueError("data not contained in the woescoremap!!!")

def alpha_beta(basepoints,baseodds,pdo):
    '''
    score = alpha + beta * ln(odds)
    :param basepoints:
    :param baseodds:
    :param pdo:
    :return:
    '''
    beta = pdo/math.log(2)
    alpha = basepoints - beta * math.log(baseodds)
    return alpha,beta

def credit_card(coefficients,intercept,feature,bin_woe_report,A,B):
    # 变量集v与对应系数coef组合：v_coef
    feature_coef = {'Var': feature, 'coef': coefficients}
    v_coef = pd.DataFrame(feature_coef).reset_index(drop=True)
    # 变量集v与对应woe组合：v_woe
    v_woe = bin_woe_report[['Var', 'bin_range', 'woe']].reset_index(drop=True)
    v_woe['Var'] = v_woe['Var'].apply(lambda x: x + '_woe')
    v_coef_woe = pd.merge(v_coef, v_woe, how='inner')

    # 构建新列score_woe:每个分组对应的子评分
    v_coef_woe['score_woe'] = round(v_coef_woe['woe'] * v_coef_woe['coef'] * B, 0)  # 变量各分段得分

    # 合成最终的评分卡：
    score_0 = round(float(A + B * intercept), 0)  # 基准分
    score_X = v_coef_woe[['Var', 'bin_range', 'woe', 'score_woe']]  # 变量中各分段得分

    grouped = score_X['score_woe'].groupby(score_X['Var'])
    groupedMax = grouped.max()
    groupedMin = grouped.min()
    maxScore = score_0 + sum(groupedMax)
    minScore = score_0 + sum(groupedMin)
    print("分值范围：[%d-%d]" %(minScore,maxScore))

    df_score_0 = pd.DataFrame(['init_score', '—', '—', '%.f' % score_0],
                              index=['Var', 'bin_range', 'woe', 'score_woe']).T
    score_card = df_score_0.append(score_X, ignore_index=True)  # 总评分卡
    score_card['score_woe'] = score_card['score_woe'].astype('float64')
    return score_card  # 输出评分卡