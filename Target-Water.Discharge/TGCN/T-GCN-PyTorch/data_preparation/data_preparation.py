import numpy as np
from scipy.sparse import load_npz
import pandas as pd
import glob
import os

def get_adj_matrix():
    adj = load_npz('adj_directed.npz').toarray()
    pd.DataFrame(adj).to_csv('mekong_adj.csv', index=False, header=False)

def get_dataset(category, start_date, end_date):
    csv_files = glob.glob(os.path.join(os.path.dirname(__file__), 'combined_station_data', '*.csv'))
    start_duration = pd.to_datetime(start_date).tz_localize('UTC')
    end_duration = pd.to_datetime(end_date).tz_localize('UTC')
    
    merged_df = None

    for file in csv_files:
        df = pd.read_csv(file)
        df['Timestamp'] = pd.to_datetime(
            df['Timestamp'], 
            errors='coerce', 
            infer_datetime_format=True
        )
        df = df[(df['Timestamp'] >= start_duration) & (df['Timestamp'] <= end_duration)]
        # Extract station name from filename
        station_name = os.path.splitext(os.path.basename(file))[0]

        # Ensure Timestamp is datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # Select only Timestamp and Discharge.Daily
        if category not in df.columns:
            print(f"[WARNING] Skipping {station_name} – no {category} column")
            continue

        df = df[['Timestamp', 'Discharge.Daily']]
        df = df.rename(columns={'Discharge.Daily': station_name})

        # Merge
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Timestamp', how='outer')

    # Sort by timestamp
    merged_df = merged_df.sort_values(by='Timestamp')

    data_only = merged_df.drop(columns=['Timestamp'])
    data_only = data_only[sorted(data_only.columns)]
    data_only.columns = list(range(len(data_only.columns)))

    data_only.to_csv('mekong_data.csv', index=False)

#get_adj_matrix()
get_dataset('Discharge.Daily', '1988-08-01', '2002-12-31')