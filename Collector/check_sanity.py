import numpy as np
import pandas as pd
import os
import argparse

#TODO: DTW classifier >> Check incorrectly classified samples. 

def list_csv_files(dir_path):
    '''Get lists of the name and paths of csv files in a directory.'''
    csv_files = list()
    csv_paths = list()
    for csv_file in os.listdir(dir_path):
        csv_path = os.path.join(dir_path, csv_file)
        if os.path.isfile(csv_path) and os.path.splitext(csv_path)[1] == '.csv':
            csv_files.append(csv_file)
            csv_paths.append(csv_path)

    return csv_files, csv_paths

def check_num_strokes(dir_path, size, print_out=True):
    '''Get the number of samples with respect to characters and the number of strokes.'''
    csv_files, csv_paths = list_csv_files(dir_path)
    dict_count = dict()
    for csv_file, csv_path in zip(csv_files, csv_paths):
        char = get_character(csv_file)
        n_strokes = count_strokes(csv_path)
        if char not in dict_count.keys():
            dict_count[char] = dict()
        if n_strokes in dict_count[char].keys():
            dict_count[char][n_strokes] += 1
        else:
            dict_count[char][n_strokes] = 1

    chars = sorted(list(dict_count.keys()))

    dict_n_strokes_max = dict()
    for char in sorted(dict_count.keys()):
        max_n_samples = -1
        n_strokes_max = -1
        if print_out:
            print('# {} = {}'.format(char, np.sum([cnt for cnt in dict_count[char].values()])))
        for n_strokes in sorted(dict_count[char].keys()):
            n_samples = dict_count[char][n_strokes]
            if print_out:
                print(' - {} strokes: # = {}'.format(n_strokes, n_samples))
            if n_samples > max_n_samples:
                max_n_samples = n_samples
                n_strokes_max = n_strokes
        dict_n_strokes_max[char] = n_strokes_max

    if print_out:
        print('Characters: {}'.format(chars))
        print('#Characters: {}'.format(len(chars)))
        print('#Samples: {}'.format(np.sum([cnt for cnt in dict_count[char].values() for char in chars])))

    for char in sorted(dict_count.keys()):
        if dict_count[char][dict_n_strokes_max[char]] < size:
            if print_out:
                print('Failed to pass the sanity test!')
            return None
    if print_out:
        print('Succeeded in passing the sanity test!')

    return dict_n_strokes_max

def get_character(f_name):
    char = f_name.split('_')[-3]
    return char

def count_strokes(csv_path):
    df = pd.read_csv(csv_path)
    n_strokes = len(pd.unique(df.i_stroke))
    return n_strokes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the integrity of collected data.')
    parser.add_argument('--dir_path', type=str,
        help='Path of a directory where handwriting files are saved.')
    parser.add_argument('--size', type=int, default=100,
        help='The number of samples needed per character')
    args = parser.parse_args()

    check_num_strokes(args.dir_path, args.size)