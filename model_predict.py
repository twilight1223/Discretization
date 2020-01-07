import pandas as pd
from utils.write_data import *
from eda_utils.eda_tools import *


if __name__=='__main__':
    CUSTOMIZE_NUM_VALUE = -9999
    CUSTOMIZE_STR_VALUE = 'unknown'
    testData = pd.read_csv('./datasource/test_data.csv')
    # 导入入模字段
    modelFeature = load_obj(path=)

    #数据预处理
    # 需要转换的数据列 'int_rate','emp_length','revol_util'
    inputCols = ['int_rate', 'revol_util']
    testData = change_percent_to_num(testData, inputCols)
    empLengthDict = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6,
                     '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
    testData['emp_length'] = testData['emp_length'].apply(apply_map, args=(empLengthDict,))
    # 缺失值填充
    nanCols = count_nan(testData)  # 缺失值统计
    fillnaMap = {}
    numColWithNan, strColWithNan = num_str_split(testData, nanCols)
    for col in numColWithNan:
        fillnaMap[col] = CUSTOMIZE_NUM_VALUE
    for col in strColWithNan:
        fillnaMap[col] = CUSTOMIZE_STR_VALUE
    print("进行缺失值填充的字段-值映射字典：\n", fillnaMap.items())
    testData.fillna(value=fillnaMap, inplace=True)

    # 对测试数据打分


        test_data[col + '_score'] = test_data[col].apply(convert_woe_to_score, args=(woe_score_dict,))
    scoreCols = [col + '_score' for col in selectedCols]
    test_data['score'] = sum(test_data[scoreCols])
    print(test_data.shape[1])