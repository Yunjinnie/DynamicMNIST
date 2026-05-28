import numpy as np
import pandas as pd

i_stroke_hover = -1

def sample_times(record_times, sampling_length):
    '''
    PARAMS
    ------
    record_times: time points of contagious intervals
    sampling_length: time length (ms) of sampling.

    RETURNS
    -----
    sampling_indexes: list. A list of sampled indexes of record_times
        aligned with sampled_times.
    sampled_times: list. A list of sampling times.
    '''
    record_times = sorted(record_times)

    # Make evenly spaced time moments
    t_last = record_times[-1]
    n_steps = int(t_last // sampling_length) + 1
    if t_last % sampling_length != 0:
        n_steps += 1
    sampled_times = sampling_length * np.array(range(n_steps))

    # Sample
    i = len(record_times) - 1
    sampling_indexes = list()
    sampling_indexes.append(i)
    reversed_sampled_times = sampled_times[::-1]
    for t_sampling in reversed_sampled_times[1:]:
        while i >= 0:
            t_recent = record_times[i]
            if t_recent <= t_sampling:
                sampling_indexes.append(i)
                break
            else:
                i -= 1

    # Reverse the collected indexes
    sampling_indexes = sampling_indexes[::-1]

    return sampling_indexes, sampled_times

def sample_dots(df, dt):
    '''
    Sampling dots using linear interpolation of two points.

    PARAMS
    ------
    df : pandas.DataFrame. Imported from a CSV file of raw recorded dots.
    dt : int. ms. Sampling time length.

    RETURNS
    ------
    df_new : pandas.DataFrame. The same format as `df` with sampled dots.
    '''
    i = 0
    sampled_times = np.arange(0, int(df.time_ms.max())+1, dt)
    x_list = list(); y_list = list(); i_stroke_list = list()
    for t_sampling in sampled_times:
        while i < len(df.time_ms):
            t_start, t_end = df.time_ms[i], df.time_ms[i+1]
            x_start, x_end = df.x[i], df.x[i+1]
            y_start, y_end = df.y[i], df.y[i+1]
            i_stroke_start, i_stroke_end = df.i_stroke[i], df.i_stroke[i+1]
            if (t_start <= t_sampling) and (t_sampling <= t_end):
                x = (t_sampling - t_start) * x_end + (t_end - t_sampling) * x_start
                x /= (t_end - t_start)
                #x = round(x)
                y = (t_sampling - t_start) * y_end + (t_end - t_sampling) * y_start
                y /= (t_end - t_start)
                #y = round(y)

                x_list.append(x); y_list.append(y)
                if i_stroke_start == i_stroke_end:
                    i_stroke_list.append(i_stroke_start)
                else:
                    i_stroke_list.append(i_stroke_hover)
                break
            else:
                i += 1
    df_new = pd.DataFrame.from_dict({
        'i_stroke':i_stroke_list,
        'time_ms':sampled_times,
        'x':x_list,
        'y':y_list
    })

    return df_new


def reset_hovering_time(df_vector, hovering_time=20):
    '''
    Reset hovering time (ms) in dataframe containing trajectories of strokes.

    PARAMS
    ------
    df_vector: pandas.dataframe. Dataframe contains trajectories of strokes.
    hovering_time: float. Hovering time (ms) between strokes.

    RETURNS
    -----
    df_vector: pandas.dataframe. Dataframe contains trajectories of strokes,
               hovering time of which is all sampling length.

    '''
    n_strokes = len(set(df_vector.i_stroke.unique()).difference({i_stroke_hover}))
    n_hovering = n_strokes - 1

    for i_hovering in range(n_hovering):
        i = df_vector[df_vector.i_stroke == i_hovering].index.max()
        if df_vector.loc[i].i_stroke != df_vector.loc[i+1].i_stroke:
            time_move = df_vector.loc[i].time_ms - df_vector.loc[i+1].time_ms + hovering_time
            df_vector.loc[i+1:, 'time_ms'] = df_vector.loc[i+1:, 'time_ms'] + time_move

    return df_vector


def get_polar_dot(x, y):
    '''Cartesian dot to polar dot'''
    theta = np.rad2deg(np.arctan2(y,x))
    magnitude = np.sqrt(x**2 + y**2)
    return theta, magnitude


def get_velocity_by_time(df, dt=20):
    '''Get direction and speed of velocity by time'''
    df = sample_dots(df, dt)
    sampled_times = df.time_ms.tolist()
    #df = reset_hovering_time(df, dt)

    list_t = list()
    list_hover_indexes = list()
    list_theta = list()
    list_magnitude = list()

    for i in range(len(df)-1):
        t = (sampled_times[i] + sampled_times[i+1]) / 2
        v_x = (df.iloc[i+1].x - df.iloc[i].x) / dt
        v_y = (df.iloc[i+1].y - df.iloc[i].y) / dt
        v_theta, v_magnitude = get_polar_dot(v_x, v_y)
        if (df.iloc[i].i_stroke == i_stroke_hover) or (df.iloc[i+1].i_stroke == i_stroke_hover):
            list_hover_indexes.append(i)
            #v_magnitude = 0 # speed = 0 at hovering moments.
        list_t.append(t); list_theta.append(v_theta); list_magnitude.append(v_magnitude)

    return (list_t, list_theta, list_magnitude), list_hover_indexes, sampled_times
