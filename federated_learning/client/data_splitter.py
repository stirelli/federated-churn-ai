# data_splitter.py - Partitions the central dataset for each client

import pandas as pd
import numpy as np
import os
import argparse

def split_and_save_data(filepath, num_clients, output_dir='federated_learning/client/data'):
    """
    Splits the central dataset into separate files for each client.
    """
    df = pd.read_csv(filepath)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into partitions
    partitions = np.array_split(df, num_clients)

    # Save each partition
    os.makedirs(output_dir, exist_ok=True)
    for i, partition in enumerate(partitions):
        partition.to_csv(os.path.join(output_dir, f"client_{i}_data.csv"), index=False)
        print(f"Saved client {i} data to {output_dir}/client_{i}_data.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into client partitions")
    parser.add_argument("--filepath", type=str, default="data/Telco-Customer-Churn.csv", help="Path to the central dataset CSV")
    parser.add_argument("--num_clients", type=int, default="2", help="Number of clients to split the data into")

    args = parser.parse_args()

    split_and_save_data(filepath=args.filepath, num_clients=args.num_clients)