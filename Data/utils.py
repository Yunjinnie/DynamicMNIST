import numpy as np
import pandas as pd
import cv2
from os.path import join, isfile, split, splitext
from os import listdir

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
    for t_sample in sampled_times:
        while i < len(df.time_ms):
            t_start, t_end = df.time_ms[i], df.time_ms[i+1]
            x_start, x_end = df.x[i], df.x[i+1]
            y_start, y_end = df.y[i], df.y[i+1]
            i_stroke_start, i_stroke_end = df.i_stroke[i], df.i_stroke[i+1]
            if t_start <= t_sample <= t_end:
                if i_stroke_start == i_stroke_end:
                    # Drawing mode.
                    i_stroke_list.append(i_stroke_start)
                    # Sampling a dot.
                    x, y = sample_dot(t_sample, t_start, t_end, x_start, x_end, y_start, y_end, 'linear')
                else:
                    # Hovering mode.
                    i_stroke_list.append(i_stroke_hover)
                    # Sampling a dot.
                    x, y = sample_dot(t_sample, t_start, t_end, x_start, x_end, y_start, y_end, 'linear')
                    #x, y = sample_dot(t_sample, t_start, t_end, x_start, x_end, y_start, y_end, 'quadratic')

                x_list.append(x); y_list.append(y)
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

def sample_dot(t_sample, t_start, t_end, x_start, x_end, y_start, y_end, method='linear'):
    if method == 'linear':
        x, y = sample_dot_linear(t_sample, t_start, t_end, x_start, x_end, y_start, y_end)
    elif method == 'quadratic':
        x, y = sample_dot_quadratic(t_sample, t_start, t_end, x_start, x_end, y_start, y_end)

    return x, y

def sample_dot_linear(t_sample, t_start, t_end, x_start, x_end, y_start, y_end):
    x = (t_sample - t_start) * x_end + (t_end - t_sample) * x_start
    x /= (t_end - t_start)
    y = (t_sample - t_start) * y_end + (t_end - t_sample) * y_start
    y /= (t_end - t_start)

    return x, y

def sample_dot_quadratic(t_sample, t_start, t_end, x_start, x_end, y_start, y_end):
    t_hov = t_end - t_start
    x_hov = x_end - x_start
    y_hov = y_end - y_start

    if t_sample <= (t_start + t_end) / 2:
        # First half.
        x = 2 * x_hov / (t_hov**2) * (t_sample - t_start)**2 + x_start
        y = 2 * y_hov / (t_hov**2) * (t_sample - t_start)**2 + y_start
    else:
        # Second half.
        x = - 2 * x_hov / (t_hov**2) * (t_sample - t_start - t_hov)**2 + x_hov + x_start
        y = - 2 * y_hov / (t_hov**2) * (t_sample - t_start - t_hov)**2 + y_hov + y_start

    return x, y


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
    theta = np.rad2deg(np.arctan2(y,x)) + 180 # theta in [0, 360]
    theta = 0 if theta == 360 else theta # # theta in [0, 360)
    magnitude = np.sqrt(x**2 + y**2)
    return theta, magnitude


def get_velocity_by_time(df, dt=20):
    '''Get direction and speed of velocity by time'''
    set_avg_speed_at_hover(df)
    df = sample_dots(df, dt)
    sampled_times = df.time_ms.tolist()
    #df = reset_hovering_time(df, dt)

    list_t = list()
    list_hover = list()
    list_dx = list()
    list_dy = list()
    list_theta = list()
    list_magnitude = list()

    for i in range(len(df)-1):
        t = (sampled_times[i] + sampled_times[i+1]) / 2
        #dx = (df.iloc[i+1].x - df.iloc[i].x) / dt
        dx = (df.iloc[i+1].x - df.iloc[i].x)
        #dy = (df.iloc[i+1].y - df.iloc[i].y) / dt
        dy = (df.iloc[i+1].y - df.iloc[i].y)
        v_theta, v_magnitude = get_polar_dot(dx, dy)
        if (df.iloc[i].i_stroke == i_stroke_hover) or (df.iloc[i+1].i_stroke == i_stroke_hover):
            list_hover.append(1) # 1 means hovering; otherwise, 0 means drawing.
            #v_magnitude = 0 # speed = 0 at hovering moments.
        else:
            list_hover.append(0) # 0 means drawing.
        list_t.append(t)
        list_dx.append(dx); list_dy.append(dy)
        list_theta.append(v_theta); list_magnitude.append(v_magnitude)

    df_velocity = pd.DataFrame({
        'time_ms':list_t,
        'dx':list_dx,
        'dy':list_dy,
        'direction':list_theta,
        'speed':list_magnitude,
        'hover':list_hover
    })

    return df_velocity

def start_on_center(df_vector, h_target, w_target):
    x, y = df_vector.x, df_vector.y
    x_center = w_target // 2
    y_center = h_target // 2

    # Make the character half.
    x = x // 2
    y = y // 2

    # Move the starting point to the center of the image.
    x = x - x[0] + x_center
    y = y - y[0] + y_center

    # Inplace update.
    df_vector.x = x
    df_vector.y = y

def resize_vector_img(df_vector, h_target=28, w_target=28, pad=4):
    x, y = df_vector.x, df_vector.y

    # 1. Boundary cropping
    ## Collect parameters needed for cropping.
    bound_L = x.min()
    bound_R = x.max()
    bound_T = y.min()
    bound_B = y.max()
    crop_w = (bound_R - bound_L + 1)
    crop_h = (bound_B - bound_T + 1)
    ## Cropping
    x_new = x - bound_L
    y_new = y - bound_T

    # 2. Make the image a square and place at the center
    if crop_w > crop_h:
        edge_len = crop_w
        gap = crop_w - crop_h
        pad_top = gap // 2
        y_new += pad_top
    else:
        edge_len = crop_h
        gap = crop_h - crop_w
        pad_left = gap // 2
        x_new += pad_left

    # 3. Resize the image.
    w_ratio =  (w_target - 2*pad - 1) / (edge_len - 1)
    h_ratio =  (h_target - 2*pad - 1) / (edge_len - 1)
    ## Resizing
    x_new *= w_ratio
    y_new *= h_ratio
    x_new = x_new.round().astype(int)
    y_new = y_new.round().astype(int)

    # 4. Add padding as MNIST images.
    x_new += pad
    y_new += pad

    # 5. Inplace update
    df_vector.x = x_new
    df_vector.y = y_new

def get_pad_len(array, from_back=False):
    # Count padding pixels.
    pad_val = 0
    pad_len = 0
    if from_back:
        array = np.flip(array)
    for val in array:
        if val != pad_val:
            break
        pad_len += 1
    return pad_len

def get_pad_params(d_img):
    '''
    PARAMS
    -----
    d_img :  Dynamic image like GIF. A sequence of images.
    - Type : numpy.ndarray. np.float.
    - Shape : (time_steps, w, h)

    RETURNS
    -----
    pad_left, pad_right, pad_top, pad_bottom : int.
    - This function returns the pad length of the four directions.
    '''
    img_last = d_img[-1,:,:]
    h_img = img_last.shape[0]
    w_img = img_last.shape[1]
    sum_along_h = img_last.sum(axis=0) # summation across height
    sum_along_w = img_last.sum(axis=1) # summation across width

    pad_left = get_pad_len(sum_along_h)
    pad_right = get_pad_len(sum_along_h, from_back=True)
    pad_top = get_pad_len(sum_along_w)
    pad_bottom = get_pad_len(sum_along_w, from_back=True)

    return pad_left, pad_right, pad_top, pad_bottom

def crop_square(d_img):
    time_steps = d_img.shape[0]
    h_img = d_img.shape[1]
    w_img = d_img.shape[2]
    pl, pr, pt, pb = get_pad_params(d_img)

    d_img_cropped = d_img[:,pt:h_img-pb,pl:w_img-pr]

    h_img_no_pad = h_img - pt - pb
    w_img_no_pad = w_img - pl - pr

    if h_img_no_pad > w_img_no_pad:
        d_img_new = np.zeros((time_steps, h_img_no_pad, h_img_no_pad))
        pad = h_img_no_pad - w_img_no_pad
        pad_L = pad // 2
        pad_R = pad_L + (pad % 2)
        for t in range(time_steps):
            d_img_new[t,:,:] = np.pad(d_img_cropped[t,:,:],
                ((0,0),(pad_L,pad_R)), mode='constant')
    else:
        d_img_new = np.zeros((time_steps, w_img_no_pad, w_img_no_pad))
        pad = w_img_no_pad - h_img_no_pad
        pad_T = pad // 2
        pad_B = pad_T + (pad % 2)
        for t in range(time_steps):
            d_img_new[t,:,:] = np.pad(d_img_cropped[t,:,:],
                ((pad_T,pad_B),(0,0)), mode='constant')

    return d_img_new

def resize_d_img(d_img, h, w):
    time_steps = d_img.shape[0]
    d_img_new = np.zeros((time_steps, h, w))
    for t in range(time_steps):
        d_img_new[t,:,:] = cv2.resize(
            d_img[t,:,:],
            (h, w), interpolation=cv2.INTER_AREA)

    #_, d_img_new = cv2.threshold(d_img_new,1,255, cv2.THRESH_TOZERO)

    return d_img_new

def get_diff_to_center_of_mass(img):
    h_cm = ((img.sum(axis=1) / img.sum() * np.arange(img.shape[0]))).sum().round().astype(int)
    w_cm = ((img.sum(axis=0) / img.sum() * np.arange(img.shape[1]))).sum().round().astype(int)
    h_c = (img.shape[0] - 1) / 2
    w_c = (img.shape[1] - 1) / 2
    h_diff = int(round(h_c - h_cm))
    w_diff = int(round(w_c - w_cm))

    return h_diff, w_diff

def resize_and_pad_raster_img(d_img, h, w, pad_width):
    # Depending on the last image, crop the previous images.
    d_img_cropped = crop_square(d_img)
    # Resize sequence images as the model takes.
    d_img_cropped_resized = resize_d_img(d_img_cropped, h-2*pad_width, w-2*pad_width)
    # Add the same pixels of padding around dynamic images.
    d_img_cropped_resized_padded = np.pad(d_img_cropped_resized,
        ((0,0),(pad_width, pad_width),(pad_width, pad_width)),
        mode='constant')

    return d_img_cropped_resized_padded

def get_file_list(dir_path, ext='.csv'):
    '''
    ext should include the dot in front, e.g., '.csv'
    '''
    if ext is None:
        file_list = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    else:
        file_list = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and splitext(f)[1] == ext]
    return file_list

def get_character_from_path(csv_path):
    character = splitext(split(csv_path)[1])[0].split('_')[2]
    return character

def get_average_drawing_speed(df_vector_img):
    distance_sum = 0
    time_sum = 0
    for i_stroke in df_vector_img.i_stroke.unique():
        df = df_vector_img[df_vector_img.i_stroke == i_stroke]
        for i in range(len(df)-1):
            dx = df.iloc[i+1].x - df.iloc[i].x
            dy = df.iloc[i+1].y - df.iloc[i].y
            distance = np.sqrt(dx**2 + dy**2)
            dt = df.iloc[i+1].time_ms - df.iloc[i].time_ms

            distance_sum += distance
            time_sum += dt

    average_speed = distance_sum / time_sum

    return average_speed

def set_avg_speed_at_hover(df_vector_img):
    df = df_vector_img
    average_speed = get_average_drawing_speed(df)

    for i in range(len(df)-1):
        if (df.iloc[i+1].i_stroke - df.iloc[i].i_stroke) == 1:
            # Two points between hovering moment.
            dx = df.iloc[i+1].x - df.iloc[i].x
            dy = df.iloc[i+1].y - df.iloc[i].y
            distance = np.sqrt(dx**2 + dy**2)
            dt = df.iloc[i+1].time_ms - df.iloc[i].time_ms
            dt_new = distance / average_speed

            df.loc[i+1:,'time_ms'] -= dt
            df.loc[i+1:,'time_ms'] += dt_new
