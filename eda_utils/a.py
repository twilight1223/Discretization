
from sklearn.datasets import load_boston
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

def forward_selection(data, target, method):
    '''
    前向逐步回归
    :param data: 数据表，包含标签列
    :param target: 标签列名
    :param method: 'AIC','BIC'
    :return:
    '''
    variate = set(data.columns)
    variate.remove(target)  # 原始自变量集
    selected = []  # 最终自变量集
    current_score, best_new_score = float('inf'), float('inf')  # 设置分数的初始值为无穷大（因为aic/bic越小越好）
    # 循环筛选变量
    while variate:
        score_with_variate = []  # 记录遍历过程的分数
        # 遍历自变量
        for candidate in variate:
            formula = "{}~{}".format(target, "+".join(selected + [candidate]))  # 组合
            if method == 'AIC':
                score = smf.ols(formula=formula, data=data).fit().aic
            else:
                score = smf.ols(formula=formula, data=data).fit().bic
            score_with_variate.append((score, candidate))
        score_with_variate.sort(reverse=True)  # 降序后，取出当前最好模型分数
        best_new_score, best_candidate = score_with_variate.pop()
        if current_score > best_new_score:  # 如果当前最好模型分数 优于 上一次迭代的最好模型分数，则将对于的自变量加入到selected中
            variate.remove(best_candidate)
            selected.append(best_candidate)
            print("选择的列：",selected)
            current_score = best_new_score
            print("score is {},continuing!".format(current_score))
        else:
            print("forward selection over!")
            break
    return selected



def stepwise_selection(data,target,method):
    variate = set(data.columns)
    variate.remove(target)  # 原始自变量集
    selected = []  # 最终自变量集
    current_score, best_new_score = float('inf'), float('inf')  # 设置分数的初始值为无穷大（因为aic/bic越小越好）
    # 循环筛选变量
    while variate:
        score_with_variate = []  # 记录遍历过程的分数
        # forward_step
        for candidate in variate:
            formula = "{}~{}".format(target, "+".join(selected + [candidate]))  # 组合
            if method == 'AIC':
                score = smf.ols(formula=formula, data=data).fit().aic
            else:
                score = smf.ols(formula=formula, data=data).fit().bic
            score_with_variate.append((score, candidate))
        score_with_variate.sort(reverse=True)  # 降序后，取出当前最好模型分数
        best_new_score, best_candidate = score_with_variate.pop()
        if current_score > best_new_score:  # 如果当前最好模型分数 优于 上一次迭代的最好模型分数，则将对于的自变量加入到selected中
            variate.remove(best_candidate)
            selected.append(best_candidate)
            print("forward_step选择的列：", selected)
            current_score = best_new_score
            print("score is {},continuing!".format(current_score))
        else:
            print("forward selection over!")
            break
        if len(selected)<2:
            continue
        # backward_step
        for col in selected:
            selected.remove(col)
            formula = "{}~{}".format(target, "+".join(selected))
            if method == 'AIC':
                score = smf.ols(formula=formula, data=data).fit().aic
            else:
                score = smf.ols(formula=formula, data=data).fit().bic
            if score < current_score:
                current_score = score
                print("backward_step删除的列：",col)
            else:
                selected.append(col)
    return selected










if __name__=="__main__":
    data = pd.read_csv('../datasource/resultdata/woe_data.csv')
    TARGET = 'status'
    selectedCol = stepwise_selection(data,TARGET,'AIC')
    print(selectedCol)

