# # -*- coding: utf-8 -*-
# """
# Created on Sun Dec 29 12:04:49 2019
#
# @author: ubu
# """
#
#
# # ChiMerge(trainData, col, 'target',special_attribute=special_attribute)
# def ChiMerge(df, col, target, max_interval=5,special_attribute=[],minBinPcnt=0):
#     '''
#     :param df: 包含目标变量与分箱属性的数据框
#     :param col: 需要分箱的属性
#     :param target: 目标变量，取值0或1
#     :param max_interval: 最大分箱数。如果原始属性的取值个数低于该参数，不执行这段函数
#     :param special_attribute: 不参与分箱的属性取值
#     :param minBinPcnt：最小箱的占比，默认为0
#     :return: 分箱结果
#     '''
#     colLevels = sorted(list(set(df[col])))
#     N_distinct = len(colLevels)
#     if N_distinct <= max_interval:  #如果原始属性的取值个数低于max_interval，不执行这段函数
#         print("The number of original levels for {} is less than or equal to max intervals".format(col))
#         return colLevels[:-1]
#     else:
#         if len(special_attribute)>=1:
#             df1 = df.loc[df[col].isin(special_attribute)]
#             df2 = df.loc[~df[col].isin(special_attribute)]
#         else:
#             df2 = df.copy()
#         N_distinct = len(list(set(df2[col])))
#
#         # 步骤一: 通过col对数据集进行分组，求出每组的总样本数与坏样本数
#         if N_distinct > 100:
#             split_x = SplitData(df2, col, 100)
#             df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
#         else:
#             df2['temp'] = df2[col]
#         # 总体bad rate将被用来计算expected bad count
#         (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)
#
#         # 首先，每个单独的属性值将被分为单独的一组
#         # 对属性值进行排序，然后两两组别进行合并
#         colLevels = sorted(list(set(df2['temp'])))
#         groupIntervals = [[i] for i in colLevels]
#
#         # 步骤二：建立循环，不断合并最优的相邻两个组别，直到：
#         # 1，最终分裂出来的分箱数<＝预设的最大分箱数
#         # 2，每箱的占比不低于预设值（可选）
#         # 3，每箱同时包含好坏样本
#         # 如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
#         split_intervals = max_interval - len(special_attribute)
#         while (len(groupIntervals) > split_intervals):  # 终止条件: 当前分箱数＝预设的分箱数
#             # 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案
#             chisqList = []
#             for k in range(len(groupIntervals)-1):
#                 temp_group = groupIntervals[k] + groupIntervals[k+1]
#                 df2b = regroup.loc[regroup['temp'].isin(temp_group)]
#                 chisq = Chi2(df2b, 'total', 'bad')
#                 chisqList.append(chisq)
#             best_comnbined = chisqList.index(min(chisqList))
#             groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined+1]
#             # 当将最优的相邻的两个变量合并在一起后，需要从原来的列表中将其移除。例如，将[3,4,5] 与[6,7]合并成[3,4,5,6,7]后，需要将[3,4,5] 与[6,7]移除，保留[3,4,5,6,7]
#             groupIntervals.remove(groupIntervals[best_comnbined+1])
#         groupIntervals = [sorted(i) for i in groupIntervals]
#         cutOffPoints = [max(i) for i in groupIntervals[:-1]]
#
#         # 检查是否有箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
#         groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
#         df2['temp_Bin'] = groupedvalues
#         (binBadRate,regroup) = BinBadRate(df2, 'temp_Bin', target)
#         [minBadRate, maxBadRate] = [min(binBadRate.values()),max(binBadRate.values())]
#         while minBadRate ==0 or maxBadRate == 1:
#             # 找出全部为好／坏样本的箱
#             indexForBad01 = regroup[regroup['bad_rate'].isin([0,1])].temp_Bin.tolist()
#             bin=indexForBad01[0]
#             # 如果是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
#             if bin == max(regroup.temp_Bin):
#                 cutOffPoints = cutOffPoints[:-1]
#             # 如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
#             elif bin == min(regroup.temp_Bin):
#                 cutOffPoints = cutOffPoints[1:]
#             # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
#             else:
#                 # 和前一箱进行合并，并且计算卡方值
#                 currentIndex = list(regroup.temp_Bin).index(bin)
#                 prevIndex = list(regroup.temp_Bin)[currentIndex - 1]
#                 df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
#                 (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
#                 chisq1 = Chi2(df2b, 'total', 'bad')
#                 # 和后一箱进行合并，并且计算卡方值
#                 laterIndex = list(regroup.temp_Bin)[currentIndex + 1]
#                 df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
#                 (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
#                 chisq2 = Chi2(df2b, 'total', 'bad')
#                 if chisq1 < chisq2:
#                     cutOffPoints.remove(cutOffPoints[currentIndex - 1])
#                 else:
#                     cutOffPoints.remove(cutOffPoints[currentIndex])
#             # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
#             groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
#             df2['temp_Bin'] = groupedvalues
#             (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
#             [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
#         # 需要检查分箱后的最小占比
#         if minBinPcnt > 0:
#             groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
#             df2['temp_Bin'] = groupedvalues
#             valueCounts = groupedvalues.value_counts().to_frame()
#             N = sum(valueCounts['temp'])
#             valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / N)
#             valueCounts = valueCounts.sort_index()
#             minPcnt = min(valueCounts['pcnt'])
#             while minPcnt < minBinPcnt and len(cutOffPoints) > 2:
#                 # 找出占比最小的箱
#                 indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0]
#                 # 如果占比最小的箱是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
#                 if indexForMinPcnt == max(valueCounts.index):
#                     cutOffPoints = cutOffPoints[:-1]
#                 # 如果占比最小的箱是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
#                 elif indexForMinPcnt == min(valueCounts.index):
#                     cutOffPoints = cutOffPoints[1:]
#                 # 如果占比最小的箱是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
#                 else:
#                     # 和前一箱进行合并，并且计算卡方值
#                     currentIndex = list(valueCounts.index).index(indexForMinPcnt)
#                     prevIndex = list(valueCounts.index)[currentIndex - 1]
#                     df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]
#                     (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
#                     chisq1 = Chi2(df2b, 'total', 'bad')
#                     # 和后一箱进行合并，并且计算卡方值
#                     laterIndex = list(valueCounts.index)[currentIndex + 1]
#                     df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
#                     (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
#                     chisq2 = Chi2(df2b, 'total', 'bad')
#                     if chisq1 < chisq2:
#                         cutOffPoints.remove(cutOffPoints[currentIndex - 1])
#                     else:
#                         cutOffPoints.remove(cutOffPoints[currentIndex])
#                 groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
#                 df2['temp_Bin'] = groupedvalues
#                 valueCounts = groupedvalues.value_counts().to_frame()
#                 valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / N)
#                 valueCounts = valueCounts.sort_index()
#                 minPcnt = min(valueCounts['pcnt'])
#         cutOffPoints = special_attribute + cutOffPoints
#         return cutOffPoints
#
#
# def AssignBin(x, cutOffPoints,special_attribute=[]):
#     '''
#     :param x: 某个变量的某个取值
#     :param cutOffPoints: 上述变量的分箱结果，用切分点表示
#     :param special_attribute:  不参与分箱的特殊取值
#     :return: 分箱后的对应的第几个箱，从0开始
#     例如, cutOffPoints = [10,20,30], 对于 x = 7, 返回 Bin 0；对于x=23，返回Bin 2； 对于x = 35, return Bin 3。
#     对于特殊值，返回的序列数前加"-"
#     '''
#     cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
#     numBin = len(cutOffPoints2)
#     if x in special_attribute:
#         i = special_attribute.index(x)+1
#         return 'Bin {}'.format(0-i)
#     if x<=cutOffPoints2[0]:
#         return 'Bin 0'
#     elif x > cutOffPoints2[-1]:
#         return 'Bin {}'.format(numBin)
#     else:
#         for i in range(0,numBin):
#             if cutOffPoints2[i] < x <=  cutOffPoints2[i+1]:
#                 return 'Bin {}'.format(i+1)
#
#
# ## 判断某变量的坏样本率是否单调
# # BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
# def BadRateMonotone(df, sortByVar, target,special_attribute = []):
#     '''
#     :param df: 包含检验坏样本率的变量，和目标变量
#     :param sortByVar: 需要检验坏样本率的变量
#     :param target: 目标变量，0、1表示好、坏
#     :param special_attribute: 不参与检验的特殊值
#     :return: 坏样本率单调与否
#     '''
#     df2 = df.loc[~df[sortByVar].isin(special_attribute)]
#     if len(set(df2[sortByVar])) <= 2:
#         return True
#     regroup = BinBadRate(df2, sortByVar, target)[1]
#     combined = zip(regroup['total'],regroup['bad'])
#     badRate = [x[1]*1.0/x[0] for x in combined]
#     badRateNotMonotone = FeatureMonotone(badRate)['count_of_nonmonotone']
#     if badRateNotMonotone > 0:
#         return False
#     else:
#         return True
#
#
# def Monotone_Merge(df, target, col):
#     '''
#     :return:将数据集df中，不满足坏样本率单调性的变量col进行合并，使得合并后的新的变量中，坏样本率单调，输出合并方案。
#     例如，col=[Bin 0, Bin 1, Bin 2, Bin 3, Bin 4]是不满足坏样本率单调性的。合并后的col是：
#     [Bin 0&Bin 1, Bin 2, Bin 3, Bin 4].
#     合并只能在相邻的箱中进行。
#     迭代地寻找最优合并方案。每一步迭代时，都尝试将所有非单调的箱进行合并，每一次尝试的合并都是跟前后箱进行合并再做比较
#     '''
#     def MergeMatrix(m, i,j,k):
#         '''
#         :param m: 需要合并行的矩阵
#         :param i,j: 合并第i和j行
#         :param k: 删除第k行
#         :return: 合并后的矩阵
#         '''
#         m[i, :] = m[i, :] + m[j, :]
#         m = np.delete(m, k, axis=0)
#         return m
#
#     def Merge_adjacent_Rows(i, bad_by_bin_current, bins_list_current, not_monotone_count_current):
#         '''
#         :param i: 需要将第i行与前、后的行分别进行合并，比较哪种合并方案最佳。判断准则是，合并后非单调性程度减轻，且更加均匀
#         :param bad_by_bin_current:合并前的分箱矩阵，包括每一箱的样本个数、坏样本个数和坏样本率
#         :param bins_list_current: 合并前的分箱方案
#         :param not_monotone_count_current:合并前的非单调性元素个数
#         :return:分箱后的分箱矩阵、分箱方案、非单调性元素个数和衡量均匀性的指标balance
#         '''
#         i_prev = i - 1
#         i_next = i + 1
#         bins_list = bins_list_current.copy()
#         bad_by_bin = bad_by_bin_current.copy()
#         not_monotone_count = not_monotone_count_current
#         #合并方案a：将第i箱与前一箱进行合并
#         bad_by_bin2a = MergeMatrix(bad_by_bin.copy(), i_prev, i, i)
#         bad_by_bin2a[i_prev, -1] = bad_by_bin2a[i_prev, -2] / bad_by_bin2a[i_prev, -3]
#         not_monotone_count2a = FeatureMonotone(bad_by_bin2a[:, -1])['count_of_nonmonotone']
#         # 合并方案b：将第i行与后一行进行合并
#         bad_by_bin2b = MergeMatrix(bad_by_bin.copy(), i, i_next, i_next)
#         bad_by_bin2b[i, -1] = bad_by_bin2b[i, -2] / bad_by_bin2b[i, -3]
#         not_monotone_count2b = FeatureMonotone(bad_by_bin2b[:, -1])['count_of_nonmonotone']
#         balance = ((bad_by_bin[:, 1] / N).T * (bad_by_bin[:, 1] / N))[0, 0]
#         balance_a = ((bad_by_bin2a[:, 1] / N).T * (bad_by_bin2a[:, 1] / N))[0, 0]
#         balance_b = ((bad_by_bin2b[:, 1] / N).T * (bad_by_bin2b[:, 1] / N))[0, 0]
#         #满足下述2种情况时返回方案a：（1）方案a能减轻非单调性而方案b不能；（2）方案a和b都能减轻非单调性，但是方案a的样本均匀性优于方案b
#         if not_monotone_count2a < not_monotone_count_current and not_monotone_count2b >= not_monotone_count_current or \
#                                         not_monotone_count2a < not_monotone_count_current and not_monotone_count2b < not_monotone_count_current and balance_a < balance_b:
#             bins_list[i_prev] = bins_list[i_prev] + bins_list[i]
#             bins_list.remove(bins_list[i])
#             bad_by_bin = bad_by_bin2a
#             not_monotone_count = not_monotone_count2a
#             balance = balance_a
#         # 同样地，满足下述2种情况时返回方案b：（1）方案b能减轻非单调性而方案a不能；（2）方案a和b都能减轻非单调性，但是方案b的样本均匀性优于方案a
#         elif not_monotone_count2a >= not_monotone_count_current and not_monotone_count2b < not_monotone_count_current or \
#                                         not_monotone_count2a < not_monotone_count_current and not_monotone_count2b < not_monotone_count_current and balance_a > balance_b:
#             bins_list[i] = bins_list[i] + bins_list[i_next]
#             bins_list.remove(bins_list[i_next])
#             bad_by_bin = bad_by_bin2b
#             not_monotone_count = not_monotone_count2b
#             balance = balance_b
#         #如果方案a和b都不能减轻非单调性，返回均匀性更优的合并方案
#         else:
#             if balance_a< balance_b:
#                 bins_list[i] = bins_list[i] + bins_list[i_next]
#                 bins_list.remove(bins_list[i_next])
#                 bad_by_bin = bad_by_bin2b
#                 not_monotone_count = not_monotone_count2b
#                 balance = balance_b
#             else:
#                 bins_list[i] = bins_list[i] + bins_list[i_next]
#                 bins_list.remove(bins_list[i_next])
#                 bad_by_bin = bad_by_bin2b
#                 not_monotone_count = not_monotone_count2b
#                 balance = balance_b
#         return {'bins_list': bins_list, 'bad_by_bin': bad_by_bin, 'not_monotone_count': not_monotone_count,
#                 'balance': balance}
#
#
#     N = df.shape[0]
#     [badrate_bin, bad_by_bin] = BinBadRate(df, col, target)
#     bins = list(bad_by_bin[col])
#     bins_list = [[i] for i in bins]
#     badRate = sorted(badrate_bin.items(), key=lambda x: x[0])
#     badRate = [i[1] for i in badRate]
#     not_monotone_count, not_monotone_position = FeatureMonotone(badRate)['count_of_nonmonotone'], FeatureMonotone(badRate)['index_of_nonmonotone']
#     #迭代地寻找最优合并方案，终止条件是:当前的坏样本率已经单调，或者当前只有2箱
#     while (not_monotone_count > 0 and len(bins_list)>2):
#         #当非单调的箱的个数超过1个时，每一次迭代中都尝试每一个箱的最优合并方案
#         all_possible_merging = []
#         for i in not_monotone_position:
#             merge_adjacent_rows = Merge_adjacent_Rows(i, np.mat(bad_by_bin), bins_list, not_monotone_count)
#             all_possible_merging.append(merge_adjacent_rows)
#         balance_list = [i['balance'] for i in all_possible_merging]
#         not_monotone_count_new = [i['not_monotone_count'] for i in all_possible_merging]
#         #如果所有的合并方案都不能减轻当前的非单调性，就选择更加均匀的合并方案
#         if min(not_monotone_count_new) >= not_monotone_count:
#             best_merging_position = balance_list.index(min(balance_list))
#         #如果有多个合并方案都能减轻当前的非单调性，也选择更加均匀的合并方案
#         else:
#             better_merging_index = [i for i in range(len(not_monotone_count_new)) if not_monotone_count_new[i] < not_monotone_count]
#             better_balance = [balance_list[i] for i in better_merging_index]
#             best_balance_index = better_balance.index(min(better_balance))
#             best_merging_position = better_merging_index[best_balance_index]
#         bins_list = all_possible_merging[best_merging_position]['bins_list']
#         bad_by_bin = all_possible_merging[best_merging_position]['bad_by_bin']
#         not_monotone_count = all_possible_merging[best_merging_position]['not_monotone_count']
#         not_monotone_position = FeatureMonotone(bad_by_bin[:, 3])['index_of_nonmonotone']
#     return bins_list
#
# def AssignBin(x, cutOffPoints,special_attribute=[]):
#     '''
#     :param x: 某个变量的某个取值
#     :param cutOffPoints: 上述变量的分箱结果，用切分点表示
#     :param special_attribute:  不参与分箱的特殊取值
#     :return: 分箱后的对应的第几个箱，从0开始
#     例如, cutOffPoints = [10,20,30], 对于 x = 7, 返回 Bin 0；对于x=23，返回Bin 2； 对于x = 35, return Bin 3。
#     对于特殊值，返回的序列数前加"-"
#     '''
#     cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
#     numBin = len(cutOffPoints2)
#     if x in special_attribute:
#         i = special_attribute.index(x)+1
#         return 'Bin {}'.format(0-i)
#     if x<=cutOffPoints2[0]:
#         return 'Bin 0'
#     elif x > cutOffPoints2[-1]:
#         return 'Bin {}'.format(numBin)
#     else:
#         for i in range(0,numBin):
#             if cutOffPoints2[i] < x <=  cutOffPoints2[i+1]:
#                 return 'Bin {}'.format(i+1)
#
# def MaximumBinPcnt(df,col):
#     '''
#     :return: 数据集df中，变量col的分布占比
#     '''
#     N = df.shape[0]
#     total = df.groupby([col])[col].count()
#     pcnt = total*1.0/N
#     return max(pcnt)
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
