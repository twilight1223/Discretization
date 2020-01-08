#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
woe分箱限制条件
bin_num 占比不低于5%
不参与分箱的数值单独设置一箱
需满足单调性
0.02<iv<0.3

"""

import numpy as np
import pandas as pd



def data_describe(X,y,feature,feature_type):
    """
    统计各取值的正负样本分布 [累计样本个数，正例样本个数，负例样本个数] 并排序
    :param data: DataFrame 输入数据
    :param var_name_bf: str 待分箱变量
    :param var_name_target: str 标签变量（y)
    :param feature_type: 特征的类型：0（连续） 1（离散）
    :return: DataFrame 排好序的各组中正负样本分布 count
    """
    # 统计待离散化变量的取值类型（string or digits)
    data_type = X[feature].apply(lambda x: type(x)).unique()
    var_type = True if str in data_type else False # 实际取值的类型：false(数字） true(字符）
    
    # 是否需要根据正例样本比重编码，True：需要，False：不需要
    #                   0（连续）    1（离散）
    #     false（数字）    0              0（离散有序）
    #     true（字符）     ×             1（离散无序）
    if feature_type == var_type:
        ratio_indicator = var_type
    elif feature_type == 1:
        ratio_indicator = 0
        # print("特征%s为离散有序数据，按照取值大小排序！" % (feature))
    elif feature_type == 0:
        exit(code="特征%s的类型为连续型，与其实际取值（%s）型不一致，请重新定义特征类型！！！" % (feature, data_type))

    # 统计各分箱（group）内正负样本分布[累计样本个数，正例样本个数，负例样本个数]
    count = pd.crosstab(X[feature], y)
    total = count.sum(axis=1)
    
    # 排序：离散变量按照pos_ratio排序，连续变量按照index排序
    if ratio_indicator:
        count['pos_ratio'] = count[count.columns[count.columns.values>0]].sum(axis=1) * 1.0 / total
        count = count.sort_values('pos_ratio') #离散变量按照pos_ratio排序
        count = count.drop(columns = ['pos_ratio'])
    else:
        count = count.sort_index() # 连续变量按照index排序
    return count, ratio_indicator


def calc_IV(count):
    """
    计算各分组的WOE值以及IV值
    :param count: DataFrame 排好序的各组中正负样本分布
    :return: 各分箱的woe和iv值
    
    计算公式：WOE_i = ln{(sum_i / sum_T) / [(size_i - sum_i) / (size_T - sum_T)]}
    计算公式：IV_i = [sum_i / sum_T - (size_i - sum_i) / (size_T - sum_T)] * WOE_i
    """
    # 计算全体样本中好坏样本的比重
    good = (count[count.columns[count.columns.values>0]].sum(axis=1) / count[count.columns[count.columns.values>0]].values.sum()).values # ？？？
    # good = (count[1] / count[1].sum()).values
    bad = (count[0] / count[0].sum()).values
    
    woe = np.log(good / bad)
    if 0 in bad:
        ind = np.where(bad == 0)[0][0]
        woe[ind] = 0
        print('第%s类负例样本个数为0！！！' % ind)
    if 0 in good:
        ind = np.where(good == 0)[0][0]
        woe[ind] = 0
        print('第%s类正例样本个数为0！！！' % ind)
    iv = (good - bad) * woe
    return woe, iv

def convert_singlevalue_to_bin(x,group,label):
    # 针对特殊值属性进行分箱转换
    for key,value in zip(group,label):
        if x in key:
            return value
    else:
        raise ValueError("`data` should be contained in a group.")



def convert_str_to_bin(x,group):
    '''
    将离散型数据利用apply方法转换为bin编码
    :param x:
    :param group:
    :return:
    '''
    for i in range(len(group)):
        if x in group[i]:
            return i
    else:
        raise ValueError("`data` should be contained in a group.")


def convert_numraw_to_bin(x,binRangeMap):
    '''
    将原始值映射到分箱编码
    :param x:
    :param binRangeMap:
    :return:
    '''
    for key,value in binRangeMap.items():
        if (len(value)>1) and (x > value[0] and x <= value[1]):
            return key
        elif (len(value)==1) and (x==value[0]):
            return key

    else:
        raise ValueError("data not contained in the grouplist!!!")

def convert_strraw_to_bin(x,binRangeMap):
    '''
    将原始值映射到分箱编码
    :param x:
    :param binRangeMap:
    :return:
    '''
    for key,value in binRangeMap.items():
        if x in value:
            return key
    else:
        raise ValueError("data not contained in the grouplist!!!")


def convert_raw_to_bin(x,binRangeMap,feature_type):
    if not feature_type:#连续型
        return convert_numraw_to_bin(x,binRangeMap)
    else:
        return convert_strraw_to_bin(x,binRangeMap)




def convert_bin_to_woe(x,binWoeMap):
    '''
    将bin编码映射为woe值
    :param x:
    :param binMap:
    :return:
    '''
    for key,value in binWoeMap.items():
        if x==key:
            return value
    else:
        raise ValueError("bin woe value do not match!!!")


def bin_describe(X,y,feature,group,labels, feature_type=0):
    """
    分箱基础统计结果
    :param bins:
    :param data:
    :param var_name_bf: 特征名
    :param var_name_target: 标签名
    :return: dataframe

    'bin_code|0|1|bin_num|badRate'
    """
    featureBin = feature + "_bin"
    if not feature_type:
        # 单值截点
        if len(group)==1 and len(labels)==1:
            # 针对连续型数据的特殊分箱做处理
            group = group.reshape((-1, 1))  # 转为二维
            X[featureBin] = X[feature].apply(convert_singlevalue_to_bin, args=(group, labels,))
        else:
            X[featureBin] = pd.cut(X[feature],group,True,labels=labels)
    else:
        X[featureBin] = X[feature].apply(convert_str_to_bin,args=(group,))

    count,var_type = data_describe(X, y, featureBin, feature_type=feature_type)
    count['bin_num'] = count.apply(lambda x: x.sum(),axis=1)
    count['bad_rate'] = count[1]/count['bin_num']
    X.drop(featureBin,axis=1,inplace=True)
    return count

def monotonic_detection(count,temp=True):
    bins = count.shape[0]
    badRate = count['bad_rate'].tolist()
    binRate = (count['bin_num']/sum(count['bin_num'])).tolist()
    if not temp:
        if bins == 2:
            return -1
    for i in range(bins):
        if i==0 or i==bins-1:
            if badRate[i] == 0.0 or badRate[i]==1.0 or binRate[i]<0.05:
                return i
        elif (badRate[i] > badRate[i - 1] and badRate[i] > badRate[i + 1])\
            or (badRate[i] < badRate[i - 1] and badRate[i] < badRate[i + 1])\
            or (badRate[i] == 0.0)\
            or (badRate[i] == 1.0)\
            or (binRate[i]<0.05):
            return i
    else:
        return -1





def woeConsMerge(group,X,y,feature, max_interval=5, feature_type=0):
    '''
    根据woe计算规则条件调整分箱
    :param group: 初始卡方分箱结果
    :param X:
    :param y:
    :param feature:
    :param max_interval:
    :param feature_type:
    :return:
    '''

    # 根据feature_type修改返回的group样式(feature_type=0: 返回分割点列表；feature_type=1：返回分箱成员列表）
    if not feature_type:
        group = [ele[-1] for ele in group] if len(group[0]) == 1 else [group[0][0]] + [ele[-1] for ele in group]
        group[0] = -float("inf") #group[0] - 0.001 if group[0] == 0 else group[0] * (1 - 0.001)  # 包含最小值
        group[-1] = float("inf") #group[-1] + 0.001 if group[-1] == 0 else group[-1] * (1 + 0.001)  # 包含最大值

    LOOP_FLAG = True

    while LOOP_FLAG:
        if not feature_type:
            labels = range(len(group)-1)
        else:
            labels = range(len(group))
        binDescribeDf = bin_describe(X,y,feature,group,labels,feature_type)
        # 获取不满足条件的分箱索引
        index = monotonic_detection(binDescribeDf,temp=False)
        if not feature_type:
            # 连续型变量合箱操作
            if index != -1:
                if index == 0:
                    group.pop(index+1)
                elif index == binDescribeDf.shape[0]-1:
                    group.pop(index)
                else:
                    tempGroup = tuple(group)
                    group.pop(index)
                    labels = range(len(group)-1)

                    tempBinDescribeDf = bin_describe(X,y,feature,group,labels,feature_type)
                    tempIndex = monotonic_detection(tempBinDescribeDf,temp=True)
                    if tempIndex == -1:
                        #返回向左合并分箱的结果
                        group = group
                    else:
                        #返回向右合并分箱的结果
                        tempGroup = list(tempGroup)
                        tempGroup.pop(index+1)
                        group = tempGroup
            else:
                LOOP_FLAG = False
        else:
            # 进行离散型变量合箱操作
            if index != -1:
                if index == 0:
                    group[index] = group[index] + group[index + 1]
                    group.remove(group[index + 1])
                elif index == binDescribeDf.shape[0] - 1:
                    group[index-1] = group[index-1] + group[index]
                    group.remove(group[index])
                else:
                    tempGroup = tuple(group)
                    #向左合箱
                    group[index - 1] = group[index - 1] + group[index]
                    group.remove(group[index])
                    labels = range(len(group))
                    tempBinDescribeDf = bin_describe(X,y,feature, group, labels, feature_type)
                    tempIndex = monotonic_detection(tempBinDescribeDf)
                    if tempIndex == -1:
                        # 返回向左合并分箱的结果
                        group = group
                    else:
                        # 返回向右合并分箱的结果
                        tempGroup = list(tempGroup)
                        tempGroup[index] = tempGroup[index] + tempGroup[index + 1]
                        tempGroup.remove(tempGroup[index + 1])
                        group = tempGroup
            else:
                LOOP_FLAG = False

    if len(labels)>max_interval:
        print("warning: 分箱后，%s的箱体个数（%s）与您输入的分箱数量（%s）不符" % (
            feature, len(group), max_interval))
    print("%s分箱调整结束!!!"%feature)
    return group,labels






def woe_summary_report(special,special_y,group,labels,X,y,feature,feature_type=0):
    '''
    Var :变量名
    bin:分箱编码
    bin_range：分箱截点

    bin_num:箱内样本数
    bin_good_num:箱内负样本数
    bin_bad_num:箱内正样本数

    bin_rate:箱内样本数/总样本数****
    good_bin_rate:箱内负样本数/负样本总数
    bad_bin_rate:箱内正样本数/正样本总数

    bin_rate_cum:累计sum(箱内样本数/总样本数)
    good_bin_rate_cum:累计sum(箱内负样本数/负样本总数)
    bad_bin_rate_cum:累计sum(箱内正样本数/正样本总数)


    good_rate:箱内负样本数/该箱内样本总数
    bad_rate:箱内正样本数/该箱内样本总数****

    woe:ln(bad_bin_rate/good_bin_rate)
    iv:(bad_bin_rate-good_bin_rate)*woe
    :param group:
    :param data:
    :param var_name_bf:
    :param var_name_target:
    :param feature_type:
    :return:
    '''

    order = ['Var','bin','bin_range','bin_num','bin_good_num','bin_bad_num','bin_rate','good_bin_rate','bad_bin_rate',\
             'bin_rate_cum','good_bin_rate_cum','bad_bin_rate_cum','good_rate','bad_rate','woe','iv']
    if not feature_type:
        binGroup = [i for i in range(len(group) - 1)]
        binRangeGroup = [[group[i], group[i + 1]] for i in range(len(group) - 1)]
    else:
        binGroup = [i for i in range(len(group))]
        binRangeGroup = [i for i in group]
    binDescribeDf = bin_describe(X, y, feature, group, labels, feature_type)
    if special.shape[0] != 0:
        special_labels = []
        for item in special[feature].unique():
            special_labels.append(len(binGroup))
            binGroup.append(len(binGroup))#在数值分箱后放置特殊值分箱
            binRangeGroup.append([item])
        # 不参与分箱属性值的统计结果
        specialDf = bin_describe(special, special_y,feature,special[feature].unique(),special_labels,feature_type=feature_type)
        binDescribeDf = pd.concat([binDescribeDf, specialDf], ignore_index=True)
        binDescribeDf.fillna(0,inplace=True)# specialDf因label全1或全0 crosstab不包含0或1列值
    binDescribeDf = binDescribeDf.rename(columns={0:'bin_good_num',1:'bin_bad_num'})
    goodNum = sum(binDescribeDf['bin_good_num'])
    badNum = sum(binDescribeDf['bin_bad_num'])
    totalNum = sum([goodNum,badNum])
    binDescribeDf['bin'] = binGroup
    binDescribeDf['Var'] = [feature for i in range(len(binGroup))]
    binDescribeDf['bin_range'] = binRangeGroup
    binDescribeDf['bin_rate'] = binDescribeDf['bin_num']*1.0/totalNum
    binDescribeDf['good_bin_rate'] = binDescribeDf['bin_good_num']*1.0/goodNum
    binDescribeDf['bad_bin_rate'] = binDescribeDf['bin_bad_num']*1.0/badNum
    binDescribeDf['bin_rate_cum'] = binDescribeDf['bin_rate'].cumsum()
    binDescribeDf['good_bin_rate_cum'] = binDescribeDf['good_bin_rate'].cumsum()
    binDescribeDf['bad_bin_rate_cum'] = binDescribeDf['bad_bin_rate'].cumsum()
    binDescribeDf['good_rate'] = binDescribeDf['bin_good_num']/binDescribeDf['bin_num']
    binDescribeDf['woe'] = np.log(binDescribeDf['bad_bin_rate']/binDescribeDf['good_bin_rate'])
    '''
    添加特殊值分箱内标签特殊分布woe值替换处理
    如果bin_good_num=0,woe为inf,用woe最大值替代
    如果bin_bad_num=0,woe为-inf,用woe最小值替代
    '''
    binDescribeDf['woe'].replace([np.inf, -np.inf], np.nan,inplace=True)
    row_index = binDescribeDf.index[np.where(np.isnan(binDescribeDf['woe']))]
    if len(row_index)>0:
        for idx in row_index:
            if binDescribeDf.loc[idx,'good_bin_rate'] == 0:
                binDescribeDf['woe'].fillna(max(binDescribeDf['woe']),inplace=True)
            if binDescribeDf.loc[idx,'good_bin_rate'] == 1:
                binDescribeDf['woe'].fillna(min(binDescribeDf['woe']),inplace=True)

    binDescribeDf['iv'] = (binDescribeDf['bad_bin_rate']-binDescribeDf['good_bin_rate'])*binDescribeDf['woe']
    binDescribeDf = binDescribeDf[order]
    binRangeMap = {key:value for key,value in zip(binGroup,binRangeGroup)}
    binWoeMap = {key:value for key,value in zip(binGroup,list(binDescribeDf['woe']))}

    featureIv = sum(binDescribeDf['iv'])
    return binDescribeDf,binRangeMap,binWoeMap,featureIv





































