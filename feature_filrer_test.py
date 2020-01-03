'''
   3.对变量进行筛选
   ① 以iv>0.02进行筛选
   ② 以相关性进行筛选
   ③ 按照vif条件进行筛选
   '''
woeCols = [col + '_woe' for col in RAW_COLS]
woeData = data[woeCols].copy(deep=True)
woeData = iv_filter(woeData, objs)
remainCols = coef_filter(data)

print(result.shape[0])