import pandas as pd
import glob
import numpy as np
import pickle
import os

def change_dataset_to_meet_requirements(output_path='data/mekong.h5'):
    files = glob.glob('combined_station_data/*.csv')
    merged = None

    for file in files:
        df = pd.read_csv(file)
        df = df[['Timestamp', 'Discharge.Daily']]
        station_name = os.path.basename(file).replace('.xlsx', '')
        df = df.rename(columns={'Discharge.Daily': station_name})

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on='Timestamp', how='inner')

    # Convert to datetime and sort
    merged['Timestamp'] = pd.to_datetime(merged['Timestamp'])
    merged = merged.sort_values('Timestamp')
    merged = merged.set_index('Timestamp')

    # Save to HDF5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_hdf(output_path, key='df', mode='w')

def change_from_npz_to_pkl():
    # Load .npz file
    data = np.load('adj_directed.npz')

    # See what's inside
    print("Keys inside .npz file:", data.files)

    # If your adj matrix is under 'adj', 'matrix', or similar key, access it
    adj = data['adj'] if 'adj' in data.files else data[data.files[0]]  # fallback to first key

    # Optionally set up a second value (like distance matrix), else use None
    adj_mx_tuple = (adj, None)

    # Save to .pkl
    with open('adj_mx.pkl', 'wb') as f:
        pickle.dump(adj_mx_tuple, f)

#change_from_npz_to_pkl()
change_dataset_to_meet_requirements()