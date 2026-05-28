import argparse
import pandas as pd
import numpy as np
from utils import get_file_list, get_character_from_path
from os.path import split, splitext

def count_hover(hover_column):
    '''
    Count how many hoverings occur in a drawing.
    #hoverings = #strokes - 1
    '''
    n_hover = 0
    prev = hover_column[0]
    for i in range(1,len(hover_column)):
        now = hover_column[i]
        if (prev, now) == (0, 1):
            # 0: Not hovering, 1: hovering.
            n_hover += 1
        prev = hover_column[i]

    return n_hover

def compute_stat_position(dir_data):
    pass

def compute_stat_velocity(dir_data):
    file_list = get_file_list(dir_data)
    csv_paths = [f for f in file_list if splitext(split(f)[1])[1].lower() == '.csv']

    df_list_by_character = dict()
    for character in range(10):
        df_list_by_character[character] = list()

    for csv_path in csv_paths:
        character = get_character_from_path(csv_path)
        df_velocity = pd.read_csv(csv_path)
        df_list_by_character[character].append(df_velocity)

    n_samples_tot = 0
    time_lengths_tot = list()
    hover_lengths_tot = list()

    for digit, df_list in df_list_by_digit.items():
        print('\n====== Statistics for digit {} ======'.format(digit))

        time_lengths = list()
        hover_lengths = list()
        hover_times = list()
        strokes = list()

        for df in df_list:
            n_hover = count_hover(df.hover)
            n_strokes = n_hover + 1
            time_lengths.append(len(df))
            hover_lengths.append(len(df[df.hover == 1]))
            hover_times.append(n_hover)
            strokes.append(n_strokes)

        n_samples_tot += len(df_list)
        time_lengths_tot += time_lengths
        hover_lengths_tot += hover_lengths

        print('- #Samples : {}'.format(len(df_list)))
        print('- Time length mean: {}'.format(np.mean(time_lengths)))
        print('- Time length max: {}'.format(np.max(time_lengths)))
        print('- Time length min: {}'.format(np.min(time_lengths)))
        print('- Hovering length mean: {}'.format(np.mean(hover_lengths)))
        print('- Hovering length max: {}'.format(np.max(hover_lengths)))
        print('- Hovering length min: {}'.format(np.min(hover_lengths)))
        print('- Types of hovering times: {}'.format(sorted(set(hover_times))))
        print('- The possible number of strokes: {}'.format(sorted(set(strokes))))

    print('\n====== Statistics for total characters ======'.format(character))

    print('- #Samples : {}'.format(n_samples_tot))
    print('- Time length mean: {}'.format(np.mean(time_lengths_tot)))
    print('- Time length max: {}'.format(np.max(time_lengths_tot)))
    print('- Time length min: {}'.format(np.min(time_lengths_tot)))
    print('- Hovering length mean: {}'.format(np.mean(hover_lengths_tot)))
    print('- Hovering length max: {}'.format(np.max(hover_lengths_tot)))
    print('- Hovering length min: {}'.format(np.min(hover_lengths_tot)))

if __name__ == '__main__':
    '''
    [Example command]
    python compute_metadata.py
        --dir_data "dir_data"
    '''
    parser = argparse.ArgumentParser(
        description='Compute the metadata of a dataset.')
    parser.add_argument('--dir_data', type=str, required=True,
        help='Directory where data are saved')
    parser.add_argument('--position', action='store_true',
        help='Directory where data are saved')
    parser.add_argument('--velocity', action='store_true',
        help='Directory where data are saved')
    args = parser.parse_args()

    if args.position:
        compute_stat_position(args.dir_data)

    if args.velocity:
        compute_stat_velocity(args.dir_data)
