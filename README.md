# Monthly Forecast Demo

This repo shows a simple Python workflow for forecasting monthly values using Linear Regression.

## Features
- Load and clean CSV data (remove `$`, `,`)
- Select monthly columns (Sep → Aug)
- Fit Linear Regression per row
- Predict next 3 months (June, July, August)
- Plot actual vs predicted values

## Requirements
pip install pandas numpy matplotlib scikit-learn openpyxl

## Run
Place your dataset path in the script (default: `D:/Dataset Test.csv`) and run:

## Output
- DataFrame including:
  - `Predicted_June`
  - `Predicted_July`
  - `Predicted_August`
- Plots for the last 10 rows
