#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import openpyxl

# ----------------- 读取数据 -----------------
def load_data(file_path):
    """读取 CSV 数据"""
    df = pd.read_csv(file_path)
    print(df.head(), df.tail())
    print(df.shape)
    df.info()
    return df

# ----------------- 数据清洗 -----------------
def clean_data(df):
    """去掉 $ 和 ,，转换为数字，删除缺失行"""
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(r'[\$\,]', '', regex=True)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
    df_clean = df_clean.dropna()
    print(df_clean.head())
    print(df_clean.describe())
    print(df_clean.isna().sum(axis=0))
    return df_clean

# ----------------- 预测未来三个月 -----------------
def predict_future(df, start_col='CurrYrActualsSep', end_col='CurrYrActualsAug'):
    """用线性回归预测6-8月"""
    selected_columns = df.loc[:, start_col:end_col]
    df_model = pd.DataFrame(selected_columns)
    model = LinearRegression()
    predictions = []

    for _, row in df_model.iterrows():
        X = np.arange(1, 10).reshape(-1, 1)  # 9个月特征
        y = row.values[:9]
        model.fit(X, y)
        x_predict = np.array([10, 11, 12]).reshape(-1, 1)
        predictions.append(model.predict(x_predict))

    df_predictions = pd.DataFrame(predictions, columns=['Predicted_June', 'Predicted_July', 'Predicted_August'])
    df_final = pd.concat([df_model, df_predictions], axis=1)
    print(df_final.tail(10))
    return df_final, predictions

# ----------------- 绘图 -----------------
def plot_predictions(df_final, predictions, last_n=10):
    """绘制实际值和预测值对比图"""
    for i, row in df_final.tail(last_n).iterrows():
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

# ----------------- 主函数 -----------------
def main():
    file_path = 'D:/Dataset Test.csv'
    df = load_data(file_path)
    df_clean = clean_data(df)
    df_final, predictions = predict_future(df_clean)
    plot_predictions(df_final, predictions, last_n=10)

if __name__ == "__main__":
    main()