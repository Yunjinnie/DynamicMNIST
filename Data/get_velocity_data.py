import argparse
import pandas as pd
import pickle
from utils import get_velocity_by_time, get_file_list
from os.path import isfile, join, split, splitext, exists
from os import listdir, mkdir
from tqdm import tqdm

def create_dataset(dir_vector_data, dir_velocity_data, dt):
    paths_vector_img = get_file_list(dir_vector_data)

    if not exists(dir_velocity_data):
        mkdir(dir_velocity_data)

    print('Convert character samples ...')
    for path_vector_img in tqdm(paths_vector_img):
        # Convert position data into velocity data.
        df = pd.read_csv(path_vector_img)
        df_velocity = get_velocity_by_time(df, dt)

        # Naming a new file containing the velocity data.
        csv_name_old = splitext(split(path_vector_img)[1])[0] # CSV name w/o extension.
        csv_full_name = '{}_{}ms.csv'.format(csv_name_old, dt)
        csv_path = join(dir_velocity_data, csv_full_name)

        # Save the velocity data.
        df_velocity.to_csv(csv_path, index=False)

def save_dataset(dataset_dict, dir_datasets):
    if not exists(dir_datasets):
        mkdir(dir_datasets)
    save_path = join(dir_datasets, '{}.pkl'.format(dataset_dict['dataset_name']))
    with open(save_path, 'wb') as f:
        pickle.dump(dataset_dict, f)
    print('- Data name : {}'.format(dataset_dict['dataset_name']))
    print('- Saved path : {}'.format(save_path))

def load_dataset(dataset_name, dir_datasets):
    '''This function helps you know how to load a dataset.'''
    with open(join(dir_datasets, '{}.pkl'.format(dataset_name)), 'rb') as f:
        loaded_dataset = pickle.load(f)
    return loaded_dataset


if __name__ == '__main__':
    '''
    [Example command]
    python get_velocity_data.py
        --dir_vector_data "dir_vector_data"
        --dir_velocity_data "dir_velocity_data"
        --dt 20
    '''
    parser = argparse.ArgumentParser(
        description='Convert positional vector images into velocity data.')
    parser.add_argument('--dir_vector_data', type=str, required=True,
        help='Directory where vector data of character trajectories are saved')
    parser.add_argument('--dir_velocity_data', type=str, required=True,
        help='Directory where velocity data will be saved')
    parser.add_argument('--dt', type=int, default=20,
        help='Time frame length (ms)')
    args = parser.parse_args()

    create_dataset(args.dir_vector_data, args.dir_velocity_data, args.dt)
