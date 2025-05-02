import pandas as pd
import numpy as np

input_len = 32
pred_len = 7

df = pd.read_csv("../data/dataset/Water_Discharge_Data.csv")
data = df.iloc[:, 1:].values  # Exclude timestamp

samples = []

for i in range(len(data) - input_len - pred_len + 1):
    x = data[i:i+input_len].flatten()
    y = data[i+input_len:i+input_len+pred_len].flatten()
    sample = np.concatenate([x, y])
    samples.append(sample)

samples = np.array(samples)
output_df = pd.DataFrame(samples)
output_df.to_csv("../data/dataset/Water_Discharge_STA_Normalized.csv", index=False)
