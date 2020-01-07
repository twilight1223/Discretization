import statsmodels.formula.api as smf


import statsmodels.api as sm

sm.Logit

def forward_selection(data, target, method):
    '''
    前向逐步回归
    :param data: 数据表，包含标签列
    :param target: 标签列名
    :param method: 'AIC','BIC'
    :return:
    '''

    variate = set(data.columns)
    variate.remove(target)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')  # 设置分数的初始值为无穷大（因为aic/bic越小越好）
    while variate:
        score_with_variate = []
        for candidate in variate:
            formula = "{}~{}".format(target, "+".join(selected + [candidate]))  # 组合
            if method == 'AIC':
                score = smf.glm(formula=formula, data=data).fit().aic
            else:
                score = smf.glm(formula=formula, data=data).fit().bic
            score_with_variate.append((score, candidate))
        score_with_variate.sort(reverse=True)
        best_new_score, best_candidate = score_with_variate.pop()
        if current_score > best_new_score:  # 如果当前最好模型分数优于上一次迭代的最好模型分数，则将对于的自变量加入到selected中
            variate.remove(best_candidate)
            selected.append(best_candidate)
            print("选择的列：",selected)
            current_score = best_new_score
            print("score is {},continuing!".format(current_score))
        else:
            print("for selection over!")
            break
    # formula = "{}~{}".format(target, "+".join(selected))
    # print("final formula is {}".format(formula))
    # model = smf.glm(formula=formula, data=data).fit()
    return selected