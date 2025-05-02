import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import scipy.sparse

def convert_combined_csv_to_astgcn_npz(input_dir, output_file, feature='Discharge.Daily'):
    """
    Convert multi-station CSV data to ASTGCN (T, N, F) format and save as .npz

    Parameters:
        input_dir (str): path to folder containing all station CSVs
        output_file (str): path to save the output .npz file
        feature (str): the feature column to use (e.g. 'Discharge.Daily')
    """
    all_stations = []
    station_names = sorted(os.listdir(input_dir))  # Alphabetical order

    for file in tqdm(station_names, desc="Processing stations"):
        path = os.path.join(input_dir, file)
        df = pd.read_csv(path)

        # Parse and set timestamp index
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.set_index('Timestamp').sort_index()

        # Keep only the selected feature
        series = df[[feature]].copy()

        all_stations.append(series)

    # Align all stations to the same date index
    aligned_data = pd.concat(all_stations, axis=1, join='inner')
    aligned_data.columns = station_names  # station name as columns

    # Convert to shape (T, N, F)
    data_tensor = aligned_data.values  # shape (T, N)
    data_tensor = np.expand_dims(data_tensor, axis=-1)  # shape (T, N, 1)

    print(f"Final data shape: {data_tensor.shape}")  # (T, N, F)
    np.savez_compressed(output_file, data=data_tensor)


def convert_npz_adj_to_csv(npz_path, output_csv_path):
    """
    Convert sparse adj.npz file to dense CSV adjacency matrix.
    
    Parameters:
        npz_path (str): path to the adj.npz file
        output_csv_path (str): path to save the output CSV file
    """
    loader = np.load(npz_path, allow_pickle=True)
    sparse_matrix = scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    dense_matrix = sparse_matrix.toarray()

    # Save as CSV
    pd.DataFrame(dense_matrix).to_csv(output_csv_path, index=False, header=False)
    print(f"Saved adjacency matrix as CSV to: {output_csv_path}")

def fix_disconnected_nodes(input_csv, output_csv):
    adj = pd.read_csv(input_csv, header=None).values
    for i in range(adj.shape[0]):
        if np.sum(adj[i]) == 0:
            adj[i, i] = 1  # add self-loop
    pd.DataFrame(adj).to_csv(output_csv, index=False, header=False)
    print(f"Fixed adjacency saved to: {output_csv}")

#convert_combined_csv_to_astgcn_npz(input_dir='combined_station_data',output_file='METR_LA.npz',feature='Discharge.Daily')

#convert_npz_adj_to_csv("adj.npz", "distance_LA.csv")


# Usage
fix_disconnected_nodes("METR_LA/distance_LA.csv", "METR_LA/distance_LA_fixed.csv")
