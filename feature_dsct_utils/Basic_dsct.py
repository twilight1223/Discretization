import numpy as np
import pandas as pd

#####################等频分箱#################################################
def equal_frequency(data,input_col,target,bin_num=10):
    colBin = input_col+"_bin"
    data[colBin] = pd.qcut(data[input_col],bin_num)
    count = pd.crosstab(data[colBin], target)
    count = count.sort_index()
    return count



#####################等宽分箱###################################################
def equal_distance(data,input_col,target,bin_num=10):
    colBin = input_col + "_bin"
    data[colBin] = pd.cut(data[input_col],bin_num)
    count = pd.crosstab(data[colBin], target)
    count = count.sort_index()
    return count

def score_bin_report(data,input_col,target,method,bin_num=10):
    order = ['bin', 'bin_range', 'bin_num', 'bin_good_num', 'bin_bad_num', 'bin_rate', 'good_bin_rate',
             'bad_bin_rate', \
             'bin_rate_cum', 'good_bin_rate_cum', 'bad_bin_rate_cum', 'good_rate', 'bad_rate']
    if method=='equal_distance':
        count = equal_distance(data,input_col,target,bin_num)
    else:
        count = equal_frequency(data,input_col,target,bin_num)
    count['bin_num'] = count.apply(lambda x: x.sum(), axis=1)
    count['bad_rate'] = count[1] / count['bin_num']
    binDescribeDf = count.rename(columns={0: 'bin_good_num', 1: 'bin_bad_num'})
    goodNum = sum(binDescribeDf['bin_good_num'])
    badNum = sum(binDescribeDf['bin_bad_num'])
    totalNum = sum([goodNum, badNum])
    binDescribeDf['bin'] = [i for i in range(count.shape[0])]
    binDescribeDf['bin_range'] = count.index
    binDescribeDf['bin_rate'] = binDescribeDf['bin_num'] * 1.0 / totalNum
    binDescribeDf['good_bin_rate'] = binDescribeDf['bin_good_num'] * 1.0 / goodNum
    binDescribeDf['bad_bin_rate'] = binDescribeDf['bin_bad_num'] * 1.0 / badNum
    binDescribeDf['bin_rate_cum'] = binDescribeDf['bin_rate'].cumsum()
    binDescribeDf['good_bin_rate_cum'] = binDescribeDf['good_bin_rate'].cumsum()
    binDescribeDf['bad_bin_rate_cum'] = binDescribeDf['bad_bin_rate'].cumsum()
    binDescribeDf['good_rate'] = binDescribeDf['bin_good_num'] / binDescribeDf['bin_num']

    binDescribeDf = binDescribeDf[order].reset_index(drop=True)

    binRangeMap = {key: value for key, value in zip(binDescribeDf['bin'], binDescribeDf['bin_range'])}
    return binDescribeDf,binRangeMap