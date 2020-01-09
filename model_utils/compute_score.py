import numpy as np
import pandas as pd
import math
from utils.config_file import *

def turn_AB(odds,basePoint=500,pdo=20,):
    flag = True
    print("开始循环调整分数：")
    while flag:
        A, B = alpha_beta(basepoints=basePoint, baseodds=1, pdo=pdo)
        score = A + B * np.log(odds)
        minScore = min(score)
        maxScore = max(score)
        print("当前循环：A-%f,B-%f,max_score-%f,min_score-%f" % (A, B, maxScore, minScore))
        if (minScore >= 300 and minScore <= 350) and (maxScore >= 850 and maxScore <= 900):
            # 正常范围分值
            flag = False
        elif (minScore >= 300 and minScore <= 350) and maxScore > 900:
            # 分值幅度过高
            pdo = pdo - 10
        elif (minScore < 300 and maxScore > 900):
            basePoint = basePoint - 10
            pdo = pdo - 10

        elif (minScore > 350 and maxScore < 850):
            # 分值幅度过低
            pdo = pdo + 10
        elif (minScore >= 300 and minScore <= 350) and maxScore < 850:
            basePoint = basePoint + 20
            pdo = pdo + 10

        elif minScore < 300 and (maxScore >= 850 and maxScore <= 900):
            # 基础分过低
            basePoint = basePoint + 50
        elif minScore > 350:
            # 基础分过高
            basePoint = basePoint - 50
        elif minScore < 300 and maxScore < 850:
            basePoint = basePoint + 50
    return A,B

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
    beta = (-1)*pdo/math.log(2)
    alpha = basepoints - beta * math.log(baseodds)
    return alpha,beta

def credit_card(coefficients,intercept,feature,bin_woe_report,A,B):
    # 变量集v与对应系数coef组合：v_coef
    feature_coef = {'Var': feature, 'coef': coefficients}
    v_coef = pd.DataFrame(feature_coef).reset_index(drop=True)
    # 变量集v与对应woe组合：v_woe
    # v_woe = bin_woe_report[['Var', 'bin_range', 'woe']].reset_index(drop=True)
    v_woe = bin_woe_report.reset_index(drop=True)
    v_woe['Var'] = v_woe['Var'].apply(lambda x: x + '_woe')
    v_coef_woe = pd.merge(v_coef, v_woe, how='inner')

    # 构建新列score_woe:每个分组对应的子评分
    v_coef_woe['score'] = round(v_coef_woe['woe'] * v_coef_woe['coef'] * B, 0)  # 变量各分段得分
    v_coef_woe.drop(['coef'],axis=1,inplace=True)

    # 合成最终的评分卡：
    BIN_WOE_COLS.append('score')
    baseScore = round(float(A + B * intercept), 0)  # 基准分
    # score_X = v_coef_woe[['Var', 'bin_range', 'woe', 'score_woe']]  # 变量中各分段得分
    # grouped = score_X['score_woe'].groupby(score_X['Var'])
    # groupedMax = grouped.max()
    # groupedMin = grouped.min()
    # maxScore = score_0 + sum(groupedMax)
    # minScore = score_0 + sum(groupedMin)
    # print("分值范围：[%d-%d]" %(minScore,maxScore))

    baseScoreRow = ['']*(len(BIN_WOE_COLS)-2)
    baseScoreRow.insert(0,'baseScore')
    baseScoreRow.append(baseScore)
    baseScoreRow = pd.DataFrame(baseScoreRow,
                              index=BIN_WOE_COLS).T
    score_card = baseScoreRow.append(v_coef_woe, ignore_index=True)  # 总评分卡
    score_card['score'] = score_card['score'].astype('float64')
    return score_card  # 输出评分卡