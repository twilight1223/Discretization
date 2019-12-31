# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# pd.set_option('max_columns', None, 'max_rows', 1000)  #在spyder的输出结果中可以改变显示结果行数和列数，在notebook中也可以改变显示结果行数和列数
#
# credit=pd.read_csv("d:/credit_samples.csv")
# credit.head()
# credit.columns
# credit.describe()
# credit.shape
# credit.dtypes
# credit['loan_status'].value_counts(normalize=True).plot(kind='bar')
# '''
# 0    0.751366
# 1    0.248634
# '''
#
# credit.isnull().sum(axis=0).sort_values(ascending=False)/credit.shape[0]
#
# credit.drop(['id','member_id','url','desc','sec_app_mths_since_last_major_derog','orig_projected_additional_accrued_interest',
# 'payment_plan_start_date','hardship_type','hardship_reason','hardship_status','deferral_term','hardship_amount',
# 'hardship_start_date','hardship_end_date','hardship_length','hardship_dpd','hardship_loan_status','hardship_payoff_balance_amount',
# 'hardship_last_payment_amount','sec_app_revol_util','sec_app_mort_acc','sec_app_earliest_cr_line','revol_bal_joint',
# 'sec_app_inq_last_6mths','sec_app_collections_12_mths_ex_med','sec_app_chargeoff_within_12_mths','sec_app_num_rev_accts',
# 'sec_app_open_act_il','sec_app_open_acc','settlement_percentage','settlement_term','settlement_date','settlement_amount',
# 'settlement_status','debt_settlement_flag_date','dti_joint','verification_status_joint','annual_inc_joint','next_pymnt_d'], axis=1, inplace=True)
#
# credit2 = credit.select_dtypes(include=[np.number])  #筛选出整形和float的
# credit2.dtypes # only numeric columns
# credit2.shape
#
# credit2.fillna(-1)
#
# '''
# 对于连续型变量，处理方式如下：
# 1，利用卡方分箱法将变量分成5个箱
# 2，检查坏样本率的单带性，如果发现单调性不满足，就进行合并，直到满足单调性
# '''
# var_cutoff = {}
# for col in credit2.columns:
#     print("{} is in processing".format(col))
#     col1 = str(col) + '_Bin'
#
#     #(1),用卡方分箱法进行分箱，并且保存每一个分割的端点。例如端点=[10,20,30]表示将变量分为x<10,10<x<20,20<x<30和x>30.
#     #特别地，缺失值-1不参与分箱
#     if -1 in set(credit2[col]):
#         special_attribute = [-1]
#     else:
#         special_attribute = []
#     cutOffPoints = ChiMerge(credit2, col, 'target',special_attribute=special_attribute)
#     var_cutoff[col] = cutOffPoints
#     credit2[col1] = credit2[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
#
#     #(2), check whether the bad rate is monotone
#     BRM = BadRateMonotone(credit2, col1, 'target',special_attribute=special_attribute)
#     if not BRM:
#         if special_attribute == []:
#             bin_merged = Monotone_Merge(credit2, 'target', col1)
#             removed_index = []
#             for bin in bin_merged:
#                 if len(bin)>1:
#                     indices = [int(b.replace('Bin ','')) for b in bin]
#                     removed_index = removed_index+indices[0:-1]
#             removed_point = [cutOffPoints[k] for k in removed_index]
#             for p in removed_point:
#                 cutOffPoints.remove(p)
#             var_cutoff[col] = cutOffPoints
#             credit2[col1] = credit2[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
#         else:
#             cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
#             temp = credit2.loc[~credit2[col].isin(special_attribute)]
#             bin_merged = Monotone_Merge(temp, 'target', col1)
#             removed_index = []
#             for bin in bin_merged:
#                 if len(bin) > 1:
#                     indices = [int(b.replace('Bin ', '')) for b in bin]
#                     removed_index = removed_index + indices[0:-1]
#             removed_point = [cutOffPoints2[k] for k in removed_index]
#             for p in removed_point:
#                 cutOffPoints2.remove(p)
#             cutOffPoints2 = cutOffPoints2 + special_attribute
#             var_cutoff[col] = cutOffPoints2
#             credit2[col1] = credit2[col].map(lambda x: AssignBin(x, cutOffPoints2, special_attribute=special_attribute))
#
#     #(3), 分箱后再次检查是否有单一的值占比超过90%。如果有，删除该变量
#     maxPcnt = MaximumBinPcnt(credit2, col1)
#     if maxPcnt > 0.9:
#         # del credit2[col1]
#         deleted_features.append(col)
#         numerical_var.remove(col)
#         print('we delete {} because the maximum bin occupies more than 90%'.format(col))
#         continue
#
#     WOE_IV = CalcWOE(credit2, col1, 'target')
#     var_IV[col] = WOE_IV['IV']
#     var_WOE[col] = WOE_IV['WOE']
#     #del credit2[col]
#
#
# # CalcWOE(trainData, col1, 'target')
# def CalcWOE(df, col, target):
#     '''
#     :param df: 包含需要计算WOE的变量和目标变量
#     :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
#     :param target: 目标变量，0、1表示好、坏
#     :return: 返回WOE和IV
#     '''
#     total = df.groupby([col])[target].count()
#     total = pd.DataFrame({'total': total})
#     bad = df.groupby([col])[target].sum()
#     bad = pd.DataFrame({'bad': bad})
#     regroup = total.merge(bad, left_index=True, right_index=True, how='left')
#     regroup.reset_index(level=0, inplace=True)
#     N = sum(regroup['total'])
#     B = sum(regroup['bad'])
#     regroup['good'] = regroup['total'] - regroup['bad']
#     G = N - B
#     regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
#     regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
#     regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
#     WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
#     for k, v in WOE_dict.items():
#         WOE_dict[k] = v['WOE']
#     IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
#     IV = sum(IV)
#     return {"WOE": WOE_dict, 'IV':IV}
#
