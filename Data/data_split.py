import argparse
import random
import pandas as pd
import os
from utils import get_character_from_path, get_file_list

def main():
    dir_dataset_path = args.data_path
    save_path = args.save_path
    seed = args.seed
    ratio_train = args.train_ratio
    ratio_val = args.val_ratio
    ratio_test = 1 - ratio_train - ratio_val

    csv_paths = get_file_list(dir_dataset_path)
    csv_paths = sorted(csv_paths)

    dict_paths = dict()
    for i_sample, csv_path in enumerate(csv_paths):
        character = get_character_from_path(csv_path)
        if character not in dict_paths.keys():
            dict_paths[character] = list()
        dict_paths[character].append(csv_path)

    # columns: path, split, character
    df_list = list()
    for character, csv_paths in dict_paths.items():
        n_samples = len(csv_paths)
        n_train = round(n_samples * ratio_train)
        n_val = round(n_samples * (ratio_train + ratio_val)) - n_train
        n_test = n_samples - n_train - n_val
        split_tags = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
        random.Random(seed).shuffle(split_tags)
        df = pd.DataFrame({'csv_path':csv_paths, 'character':character, 'split':split_tags})
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    df.to_csv(save_path, index=False)
    print('Splitted samples are saved in {}.'.format(save_path))
    print(df.groupby(by=['split']).count())
    print(df.groupby(by=['character', 'split']).count())

if __name__ == '__main__':
    '''
    [Example command]
    python data_split.py
        --data_path "data_velocity_digit_1000"
        --save_path "dir_velocity_data"
        --train_ratio 0.8
        --val_ratio 0.1
    '''
    parser = argparse.ArgumentParser(
        description='Split a velocity dataset into training, validation, and test sets.')
    parser.add_argument('-d', '--data_path', type=str, required=True,
        help='A directory path storing CSV files of velocity handwriting data')
    parser.add_argument('-s', '--save_path', type=str, required=True,
        help='A CSV file path storing the splitted samples of velocity handwriting data')
    parser.add_argument('--train_ratio', type=float, required=True,
        help='The ratio of a training set in the whole set')
    parser.add_argument('--val_ratio', type=float, required=True,
        help='The ratio of a validation set in the whole set')
    parser.add_argument('--seed', type=int, default=97531,
        help='A random seed for shuffling before spltting')

    args = parser.parse_args()

    dir_path, _ = os.path.split(args.save_path)
    os.makedirs(dir_path, exist_ok=True)

    main()
