import numpy as np
import pandas as pd
import os
import argparse
from shutil import copyfile
from check_sanity import list_csv_files, check_num_strokes, get_character, count_strokes

def get_sanity(target_dir, destination_dir, size):
    dict_n_strokes_max = check_num_strokes(target_dir, size, False)
    chars = list(dict_n_strokes_max.keys())
    csv_files, csv_paths = list_csv_files(target_dir)

    csv_files_sanity = {char:list() for char in chars}
    csv_paths_sanity = {char:list() for char in chars}

    for csv_file, csv_path in zip(csv_files, csv_paths):
        char = get_character(csv_file)
        n_strokes = count_strokes(csv_path)

        if dict_n_strokes_max[char] == n_strokes:
            csv_files_sanity[char].append(csv_file)
            csv_paths_sanity[char].append(csv_path)

    for char in chars:
        csv_files_sanity[char] = csv_files_sanity[char][-size:]
        csv_paths_sanity[char] = csv_paths_sanity[char][-size:]

    os.makedirs(destination_dir, exist_ok=True)
    for char in chars:
        for f, p in zip(csv_files_sanity[char], csv_paths_sanity[char]):
            dst_path = os.path.join(destination_dir, f)
            copyfile(p, dst_path)

    print('Sanitized samples have been saved at {} !'.format(destination_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the integrity of collected data.')
    parser.add_argument('--target_dir', '-t', type=str,
        help='Path of a directory where handwriting files are saved.')
    parser.add_argument('--destination_dir', '-d', type=str,
        help='Path of a directory where sanitized handwriting files are to be saved.')
    parser.add_argument('--size', type=int, default=100,
        help='The number of samples needed per character')
    args = parser.parse_args()

    get_sanity(args.target_dir, args.destination_dir, args.size)