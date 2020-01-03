import numpy as np
import pandas as pd

def coef_filter(woe_data,feature_iv_data):
    remainCols = []
    correlation_matrix = np.corrcoef(woe_data, rowvar=0)  # 相关性分析
    # return correlation_matrix
    print(correlation_matrix.round(2))  # 打印输出相关性结果
    # 查找相关系数大于0.5的字段
    correlation_matrix_abs = np.abs(correlation_matrix)
    for i in range(len(correlation_matrix_abs.shape[0])):
        for j in range(len(correlation_matrix.shape[1])):
            if i!=j and correlation_matrix_abs.iloc[i:j]>0.5:
                print(i,j)
                # 比较相关性较高的两个字段的iv值，保留iv值较大的字段
                if feature_iv_data.iloc[i,1]>feature_iv_data.iloc[j,1]:
                    remainCols.append(feature_iv_data.iloc[i,0])
                    print("保留字段：",feature_iv_data.iloc[i,0])
    return remainCols


def iv_filter(data,threshold=0.02):
    '''
    过滤出iv值大于0.02的字段
    :param data:
    :param threshold:
    :return: 结果表 feature|iv
    '''
    data = data[data['iv'] > threshold]
    return data

if __name__=='__main__':
    woe_data = pd.read_csv('../datasource/woe_data.csv')
    feature_iv = pd.read_csv('../datasource/feature_iv.csv')
    data = iv_filter(feature_iv)
    remainCols = data['feature']
    print(remainCols)




