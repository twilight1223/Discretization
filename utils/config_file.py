
# 配置保存文件路径
FINAL_RESULTS_PATH = './datasource/finalresult/result.xlsx'# 保存excel结果
BIN_WOE_REPORT_PATH = './datasource/resultdata/bin_woe_report.csv'#保存分箱报告
DATA_BIN_WOE_PATH = './datasource/resultdata/data_bin_woe.csv'#保存bin_woe编码数据
TRAIN_WOE_DATA_PATH = './datasource/resultdata/train_woe_data.csv'#单独保存训练woe数据
TEST_WOE_DATA_PATH = './datasource/resultdata/test_woe_data.csv'#单独保存测试woe数据

FEATURE_IV = './datasource/obj/feature_iv.pkl' #保存feature:iv映射字典
BIN_RANGE = './datasource/obj/bin_range.pkl' #保存bin:range映射字典
BIN_WOE = './datasource/obj/bin_woe.pkl' #保存bin:woe映射字典


WOE_SCORE = './datasource/model/woe_score.pkl' #保存woe:score映射字典
MODEL = './datasource/model/lr_model.pkl'#保存逻辑回归模型
MODEL_FEATURE = './datasource/model/model_feature.pkl' #保存模型特征字段
MODEL_TRAIN_RESULTS = './datasource/model/train_results.csv'#保存模型训练数据集及预测结果
MODEL_TEST_RESULTS = './datasource/model/test_results.csv'#保存模型测试数据集及预测结果
MODEL_TRAIN_SCORE_RESULTS = './datasource/model/train_score_results.csv'#保存模型评分映射结果
MODEL_TEST_SCORE_RESULTS = './datasource/model/test_score_results.csv'#保存模型评分映射结果
SCORE_CARD = './datasource/model/score_card.csv'
TEST_SCORE_BIN_RESULTS = './datasource/model/test_score_bin_results.csv'



BIN_ENCODE_LOG_FILE = './log_file/bindata_encode.txt'
FEATURE_FILTER_LOG_FILE = './log_file/woe_col_filter.txt'
MODEL_TRAIN_LOG_FILE = './log_file/credit_model.txt'
MODEL_PREDICT_LOG_FILE = './log_file/model_predict.txt'

###############################################################
#数据集设置
DATA_SET = './datasource/data_due.csv'
TEST_DATA_SET = './datasource/test_data.csv'
TARGET = 'status'
# 删除变量
DEL_COLUMNS = ['first_transaction_time',TARGET]
# DEL_COLUMNS = ['issue_d','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d']
#数据转换
DATA_TRANSFER = {'change_percent_to_num':['int_rate','revol_util'],
                 'apply_map':{'emp_length':{'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5, '6 years': 6,
                     '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
                              }
                 }

DATA_TRANSFER = {}

BIN_WOE_COLS = ['Var', 'bin', 'bin_range', 'bin_num', 'bin_good_num', 'bin_bad_num', 'bin_rate', 'good_bin_rate', 'bad_bin_rate', \
    'bin_rate_cum', 'good_bin_rate_cum', 'bad_bin_rate_cum', 'good_rate', 'bad_rate', 'woe', 'iv']
###############################################################


#数据分析
# 自定义缺失值
CUSTOMIZE_NUM_VALUE = -9999  # 自定义缺失值填充
CUSTOMIZE_STR_VALUE = 'unknown'





