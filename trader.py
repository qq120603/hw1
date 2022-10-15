import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import datetime
if __name__ == '__main__':
    # You should not modify this part.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    # The following part is an example.
    # You can modify it at will.
    # 整理訓練資料
    training_data = pd.read_csv(args.training, sep=',', names=[
                                'Open', 'High', 'Low', 'Close'])
# 設定index
# datelist = pd.date_range(
#    datetime.datetime.today(), periods=len(training_data), freq="D").tolist()
# training_data.index = datelist
# training_data.index.name = 'Date'
training_data['week_trend'] = np.where(
    training_data.Close.shift(-1) > training_data.Close, 1, 0)

# 檢查資料有無缺值
training_data.isnull().sum()
# 有缺值的資料整列拿掉
training_data = training_data.dropna()
# print(training_data)
# 整理測試資料
testing_data = pd.read_csv(args.testing, sep=',', names=[
    'Open', 'High', 'Low', 'Close'])
testing_data['week_trend'] = '0'
# 去除最後一列
testing_data = testing_data.iloc[:-1].copy()
# print(testing_data)
# 訓練樣本再分成目標序列 y 以及因子矩陣 X
train_X = training_data.drop('week_trend', axis=1)
train_y = training_data.week_trend
# print(train_y)
# print(testing_data)
test_X = testing_data.drop('week_trend', axis=1)
test_y = testing_data.week_trend

# 叫出一棵決策樹
model = DecisionTreeClassifier()
model.fit(train_X, train_y)
# print(train_X.loc[[1, 5, 7]])
prediction = []
action = [0] * len(testing_data)
# print(action)
for i in range(0, len(testing_data)):
    rowdata = test_X.loc[[i]]
    rowpredit = model.predict(rowdata)
    prediction.append(rowpredit.item(0))
i = 0
for i in range(len(prediction)):
    # 第一天預測漲就買 否則賣
    if i == 0 and prediction[i] == 1:
        action[i] = 1
    elif i == 0 and prediction[i] == 0:
        action[i] = -1
    # 前一天跌 當天漲 還能買就買
    else:
        if i != 0 and prediction[i-1] == 0 and prediction[i] == 1 and sum(action) <= 1 and sum(action) > -1:
            action[i] = 1
        else:
            action[i] = 0
    # 前一天漲 當天跌 還能賣就賣
        if i != 0 and prediction[i-1] == 1 and prediction[i] == 1 and sum(action) <= 1 and sum(action) > -1:
            action[i] = -1
        else:
            action[i] = 0
    # 連續漲或跌都不動

# 寫檔案
with open(args.output, 'w') as f:
    for a in action:
        f.write(str(a)+'\n')
