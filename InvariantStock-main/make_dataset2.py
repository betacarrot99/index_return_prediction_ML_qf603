import os
import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
import multiprocessing

# Define argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process the dataset for training, validation, and testing.")
    parser.add_argument('--data_dir', type=str, default='/Users/kevinmwongso/Documents/SMU MQF/qf603_Quantitative_Analysis_of_Financial_Market/PROJECT/InvariantStock-main/data', help='directory containing the data')
    parser.add_argument('--train_date', type=str, default='2023-06-30', help='end date for training data')
    parser.add_argument('--valid_date', type=str, default='2023-09-30', help='end date for validation data')
    parser.add_argument('--test_date', type=str, default='2023-12-31', help='end date for test data')
    parser.add_argument('--seq_len', type=int, default=20, help='sequence length for processing')
    return parser.parse_args()

def norm(df_tuple):
    df = df_tuple[1]
    mean = df.mean()
    std = df.std()
    df = (df - mean) / std 
    return df

def multi_normalize(df_list):
    with multiprocessing.Pool() as pool:
        results = pool.map(norm, df_list)
    df = pd.concat(results)
    return df

def get_index(index, date_list, dataset):
    sequence_length = 20
    date, stock = dataset.index[index]
    if date > date_list[-sequence_length]:
        return None
    date_seq = range(date_list.index(date), date_list.index(date) + sequence_length)
    idx_list = [(date_list[i], stock) for i in date_seq]
    if not all(i in dataset.index for i in idx_list):
        return None

    return np.stack([dataset.index.get_indexer(idx_list)])

def multi_get_index(index_list, date_list, dataset):
    with multiprocessing.Pool() as pool:
        results = pool.starmap(get_index, [(i, date_list, dataset) for i in index_list])
    results = [i for i in results if i is not None]
    return np.stack(results)

def main():
    
    args = parse_arguments()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    dataset = pd.read_pickle(os.path.join(args.data_dir, "usdataset.pkl"))
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])

    dataset.set_index(["datetime", "instrument"], inplace=True)
    
    # Normalize the dataset
    dataset[dataset.columns.drop("label")] = multi_normalize([*dataset[dataset.columns.drop("label")].groupby("datetime")])

    # Convert dates to Timestamps
    train_end_date = pd.to_datetime(args.train_date)
    valid_end_date = pd.to_datetime(args.valid_date)
    test_end_date = pd.to_datetime(args.test_date)

    # Define ranges for training, validation, and testing
    # train_range = range(0, len(dataset.loc[dataset.index.get_level_values("datetime") <= train_end_date]))
    # valid_range = range(len(dataset.loc[dataset.index.get_level_values("datetime") <= train_end_date]),
    #                     len(dataset.loc[dataset.index.get_level_values("datetime") <= valid_end_date]))
    # test_range = range(len(dataset.loc[dataset.index.get_level_values("datetime") <= valid_end_date]),
    #                    len(dataset))
    
    train_range = range(0, len(dataset.loc[dataset.index.get_level_values("datetime") <= train_end_date]))
    valid_range = range(len(dataset.loc[dataset.index.get_level_values("datetime") <= train_end_date]),
                        len(dataset.loc[dataset.index.get_level_values("datetime") <= valid_end_date]))
    test_range = range(len(dataset.loc[dataset.index.get_level_values("datetime") <= valid_end_date]),
                       len(dataset))

    print('checkpoint0')

    dataset.to_pickle(os.path.join(args.data_dir, "usdataset_norm.pkl"))

    # Remaining code...
    print('checkpoint1')

    date_list = list(dataset.index.get_level_values("datetime").unique())

    # def get_index(index):
    #     sequence_length = 20
    #     date, stock = dataset.index[index]
    #     if date > date_list[-sequence_length]:
    #         return None
    #     date_seq = range(date_list.index(date), date_list.index(date) + sequence_length)
    #     idx_list = [(date_list[i], stock) for i in date_seq]
    #     if not all(i in dataset.index for i in idx_list):
    #         return None

    #     return np.stack([dataset.index.get_indexer(idx_list)])

    # def multi_get_index(index_list):
    #     with multiprocessing.Pool() as pool:
    #         results = pool.map(get_index, index_list)
    #     results = [i for i in results if i is not None]
    #     return np.stack(results)
    print('checkpoint2')
    train_index = multi_get_index([i for i in train_range], date_list, dataset)
    np.save(os.path.join(args.data_dir, "train_index.npy"), np.squeeze(train_index))
    valid_index = multi_get_index([i for i in valid_range], date_list, dataset)
    np.save(os.path.join(args.data_dir, "valid_index.npy"), np.squeeze(valid_index))
    test_index = multi_get_index([i for i in test_range], date_list, dataset)
    np.save(os.path.join(args.data_dir, "test_index.npy"), np.squeeze(test_index))

    print("Success!")

if __name__ == '__main__':
    main()
