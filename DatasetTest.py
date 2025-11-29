#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import openpyxl

# 读取数据
data = pd.read_csv('D:/Dataset Test.csv')
print(data.head(), data.tail())
print(data.shape)
data.info()

# 数据清洗：去掉 $ 和 ,，转换为数字，删除缺失行
data_clean = data.copy()
for col in data_clean.columns:
    data_clean[col] = data_clean[col].replace(r'[\$\,]', '', regex=True)
    data_clean[col] = pd.to_numeric(data_clean[col], errors='ignore')
data_clean = data_clean.dropna()
print(data_clean.head())

# 基本统计
print(data_clean.describe())
print(data_clean.isna().sum(axis=0))

# 选择月份列
selected_columns = data_clean.loc[:, 'CurrYrActualsSep':'CurrYrActualsAug']
print(selected_columns.tail(7))

# 创建 DataFrame
df = pd.DataFrame(selected_columns)
X_train = df.iloc[:, :9]
y_train = df.iloc[:, 9:12]
print("X shape:", X_train.shape, "y shape:", y_train.shape)

# 初始化回归模型
model = LinearRegression()
predictions = []

# 循环每行预测6-8月
for _, row in df.iterrows():
    X = np.arange(1, 10).reshape(-1, 1)  # 9个月特征
    y = row.values[:9]
    model.fit(X, y)
    x_predict = np.array([10, 11, 12]).reshape(-1, 1)
    predictions.append(model.predict(x_predict))

# 添加预测结果
df_predictions = pd.DataFrame(predictions, columns=['Predicted_June', 'Predicted_July', 'Predicted_August'])
df_final = pd.concat([df, df_predictions], axis=1)
print(df_final.tail(10))

# 绘图：最后10行
for i, row in df.tail(10).iterrows():
    actual = row.values[:9]
    predicted = predictions[i]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), actual, label=f"Actual Row {i+879}", marker='o')
    plt.plot([10, 11, 12], predicted, label=f"Predicted Row {i+879}", linestyle='--', marker='x')
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.title(f'Actual vs Predicted Row {i+879}')
    plt.xticks(range(1, 13), ['Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug'])
    plt.legend()
    plt.grid(True)
    plt.show()