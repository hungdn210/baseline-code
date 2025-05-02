# save_output_scaler.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv('../data/dataset/Water_Discharge_STA_Normalized.csv')
OUT_DIM = 7 * 26
IN_DIM = 32 * 26

y = df.iloc[:, IN_DIM:].values  # Only the outputs

scaler_y = MinMaxScaler()
scaler_y.fit(y)

joblib.dump(scaler_y, '../data/output/output_scaler.pkl')
print("Scaler saved to ../data/output/output_scaler.pkl")
