
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
            print("for selection over!")
            break
    formula = "{}~{}".format(target, "+".join(selected))
    print("final formula is {}".format(formula))
    model = smf.ols(formula=formula, data=data).fit()
    return selected,model



# def stepwise_selection(data,target,method):
#                 # (X, y,initial_list=[],threshold_in=0.01,threshold_out=0.05,verbose=True):
#     """ Perform a forward-backward feature selection
#     based on p-value from statsmodels.api.OLS
#     Arguments:
#         X - pandas.DataFrame with candidate features
#         y - list-like with the target
#         initial_list - list of features to start with (column names of X)
#         threshold_in - include a feature if its p-value < threshold_in
#         threshold_out - exclude a feature if its p-value > threshold_out
#         verbose - whether to print the sequence of inclusions and exclusions
#     Returns: list of selected features
#     Always set threshold_in < threshold_out to avoid infinite looping.
#     See https://en.wikipedia.org/wiki/Stepwise_regression for the details
#     """
#     variate = set(data.columns)
#     variate.remove(target)  # 原始自变量集
#     selected = []  # 最终自变量集
#     current_score, best_new_score = float('inf'), float('inf')  # 设置分数的初始值为无穷大（因为aic/bic越小越好）
#     while True:
#         changed = False
#         # forward step
#         excluded = list(set(variate) - set(selected))
#         new_pval = pd.Series(index=excluded)
#         for new_column in excluded:
#             model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
#             new_pval[new_column] = model.pvalues[new_column]
#         best_pval = new_pval.min()
#         if best_pval < threshold_in:
#             best_feature = new_pval.argmin()
#             included.append(best_feature)
#             changed = True
#             if verbose:
#                 print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
#
#         # backward step
#         model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
#         # use all coefs except intercept
#         pvalues = model.pvalues.iloc[1:]
#         worst_pval = pvalues.max()  # null if pvalues is empty
#         if worst_pval > threshold_out:
#             changed = True
#             worst_feature = pvalues.argmax()
#             included.remove(worst_feature)
#             if verbose:
#                 print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
#         if not changed:
#             break
#     return included
#
#
# result = stepwise_selection(X, y)
#
# print('resulting features:')
# print(result)

if __name__=="__main__":
    data = pd.read_csv('../datasource/woe_data.csv')
    TARGET = 'loan_status'
    selectedCols,model = forward_selection(data,TARGET,'AIC')
    print(selectedCols)

