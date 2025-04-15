import numpy as np
from scipy.sparse import load_npz
import pandas as pd


def get_adj_matrix():
    adj = load_npz('adj_directed.npz').toarray()
    pd.DataFrame(adj).to_csv('mekong_adj.csv', index=False, header=False)

def get_dataset():
    # Load your file
    df = pd.read_csv('Water_Discharge_Data.csv')

    # Drop the timestamp
    df = df.drop(columns=df.columns[0])  # Assumes first column is timestamp

    # Replace column names with fake numeric node IDs
    df.columns = list(range(len(df.columns)))

    # Save as T-GCN-compatible CSV
    df.to_csv('mekong_data.csv', index=False, header=True)

get_adj_matrix()