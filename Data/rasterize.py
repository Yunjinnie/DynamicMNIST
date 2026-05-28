import csv
import os
import numpy as np
import pandas as pd
import argparse
import PIL.Image
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
from wand.display import display
from tqdm import tqdm

from utils import sample_times, sample_dots, i_stroke_hover, \
                  reset_hovering_time, resize_vector_img, \
                  resize_and_pad_raster_img, start_on_center

def read_csv(csv_path):
    '''
    Read a CSV files containing trajectories of dots.
    Each row is in the format of (stroke_id, time_ms, x, y).
    '''
    with open(csv_path, newline='\n') as csvfile:
        csv_lines = csv.reader(csvfile, delimiter=',', quotechar='|')
        csv_lines = list(csv_lines)
        header = csv_lines[0]
        rows = csv_lines[1:]
    return header, rows

def RGB2Grey(np_image):
    '''A widely used RGB2Grey conversion equation'''
    R = np_image[:,:,0]
    G = np_image[:,:,1]
    B = np_image[:,:,2]
    Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return Y

def shift(X, dy, dx):
    '''Shifting an image X as much as (dy, dx).'''
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def get_diff_to_center_of_mass(img):
    h_cm = ((img.sum(axis=1) / img.sum() * np.arange(img.shape[0]))).sum().round().astype(int)
    w_cm = ((img.sum(axis=0) / img.sum() * np.arange(img.shape[1]))).sum().round().astype(int)
    h_c = (img.shape[0] - 1) / 2
    w_c = (img.shape[1] - 1) / 2
    h_diff = int(round(h_c - h_cm))
    w_diff = int(round(w_c - w_cm))

    return h_diff, w_diff

def rasterize(csv_path, save_dir, sampling_length, stroke_width,
        remove_hover, raw_movement, mnist, is_starting_on_center=False):
    '''
    Rasterize a vector image contained in a CSV file into the formats of
    NPZ and GIF.

    PARAMS
    -----
    csv_path: str. Path of a CSV file conatining stroke trajectorie of dots.
    save_dir: str. Path of a directory where NPZ and GIF images are saved.
    sampling_length: int. How long each image appears. Sample length. Millisecond.
    stroke_width: int. Width (pixel) of each stroke.
    '''
    dir_data, csv_name = os.path.split(csv_path)
    csv_name = os.path.splitext(csv_name)[0]
    st, wh, target_character, user, hand = csv_name.split('_')
    w, h = wh.split('x')
    w = int(w); h = int(h)

    # Read trajectories of dots from a given CSV file.
    # df_v == DataFrame of a vector image
    df_v = pd.read_csv(csv_path)

    # Remove hovering moments. Set hovering time as sampling length.
    if remove_hover:
        df_v = reset_hovering_time(df_v, sampling_length)

    # Sample dot according to sampling length.
    if not raw_movement:
        df_v = sample_dots(df_v, sampling_length)

    # Start drawing on the center of the image, the size of which is half.
    if is_starting_on_center:
        start_on_center(df_v, h_target=h, w_target=w)

    # Resize the vector image to be used togther with the MNIST data.
    if mnist:
        w = h = 28
        resize_vector_img(df_v, h_target=h, w_target=w, pad=4) # inplace operation.
        csv_name = csv_name.replace(wh, '{}x{}'.format(w, h))

    # Color settings
    color_background = Color('black')
    color_stroke = Color('white')

    # Data collectors
    np_images = list() # images in numpy.array
    pil_images = list() # images in PIL.Image
    times = df_v.time_ms.tolist()

    # First image is all black.
    np_image_grey = np.zeros((w, h))
    np_images.append(np_image_grey)
    pil_images.append(PIL.Image.fromarray(np_image_grey))

    # Rasterize each moment.
    for j in range(1, len(df_v)):
        rows = df_v.loc[:j]

        with Drawing() as draw:
            # Drawing preset
            draw.stroke_width = stroke_width
            draw.stroke_color = color_stroke
            draw.fill_color = color_background

            # Drawing
            draw.path_start()
            prev_s, prev_t, prev_x, prev_y = rows.loc[0]
            prev_s, prev_x, prev_y = int(prev_s), int(prev_x), int(prev_y)
            draw.path_move(to=(prev_x, prev_y))
            for i in range(1, len(rows)):
                s, t, x, y = rows.loc[i]
                s, x, y = int(s), int(x), int(y)
                if raw_movement:
                    if prev_s != s:
                        draw.path_move(to=(x, y))
                    else:
                        draw.path_line(to=(x, y))
                else:
                    if (prev_s == i_stroke_hover) and (prev_s != s):
                        # Start a new sub-path at the last hovering step.
                        draw.path_move(to=(x, y))
                    elif (prev_s == i_stroke_hover) or (s == i_stroke_hover):
                        # Draw nothing at hovering steps except the last.
                        pass
                    else:
                        draw.path_line(to=(x, y))
                prev_s, prev_x, prev_y = s, x, y
            draw.path_move(to=(prev_x, prev_y))
            draw.path_finish()

            # Save drawing as a raster image.
            with Image(width=w, height=h, background=color_background) as image:
                draw(image)
                # np_image.shape == (w, h, 4)
                # 4: RGBA channel. RGBA vector = (R, G, B, A)
                # Each value is an integer and ranges from 0 to 255.
                np_image_RGBA = np.array(image)
                np_image_grey = RGB2Grey(np_image_RGBA)
                np_images.append(np_image_grey)

    if not raw_movement:
        # Get the most recent images for all time moments
        i_image_list, t_moments  = sample_times(times, sampling_length)
        np_images = np.asarray([np_images[i] for i in i_image_list])

    if mnist:
        np_images = resize_and_pad_raster_img(np_images, w=28, h=28, pad_width=4)

    if not is_starting_on_center:
        # Move the center of mass to be at the center of the last image.
        h_diff, w_diff = get_diff_to_center_of_mass(np_images[-1])
        np_images = np.asarray([shift(np_image, h_diff, w_diff) for np_image in np_images])
    pil_images = [PIL.Image.fromarray(np_image) for np_image in np_images]

    # String to specify either raw or sampled movement to the file name.
    str_mov_type = 'rmv' if raw_movement else 'smv'

    # Save one array stacked from numpy.array images of all time moments.
    np_images = np.stack(np_images)
    npz_path = os.path.join(save_dir, f'{csv_name}_{str_mov_type}_{stroke_width}px_{sampling_length}ms.npz')
    with open(npz_path, 'wb') as f:
        np.savez_compressed(f, np_images)
        #print('Saved the NPZ format of the raster image sequence in', npz_path)

    # Save PIL.Image images into a GIF file.
    gif_path = os.path.join(save_dir, f'{csv_name}_{str_mov_type}_{stroke_width}px_{sampling_length}ms.gif')
    with PIL.Image.new('L', (w, h)) as im:
        im.save(gif_path, save_all=True, append_images=pil_images,
            duration=sampling_length, loop=0)
        #print('Saved the GIF format of the raster image sequence in', gif_path)

    # Save PIL.Image image into a PNG file.
    png_path = os.path.join(save_dir, f'{csv_name}_{str_mov_type}_{stroke_width}px_{sampling_length}ms.png')
    pil_images[-1].convert('L').save(png_path)
    #print('Saved the PNG format of the raster image sequence in', png_path)

def rasterize_dev(df_dots, w, h, sampling_length, stroke_width, gif_path, png_path, eod_stop=False):
    '''
    Rasterize a vector image contained in a CSV file into the formats of
    NPZ and GIF.
    This makes use of EOD in df_dots.

    PARAMS
    -----
    csv_path: str. Path of a CSV file conatining stroke trajectorie of dots.
    save_dir: str. Path of a directory where NPZ and GIF images are saved.
    sampling_length: int. How long each image appears. Sample length. Millisecond.
    stroke_width: int. Width (pixel) of each stroke.
    '''
    w = int(w); h = int(h)

    # Color settings
    color_background = Color('black')
    color_stroke = Color('white')
    color_fill = Color('none')

    # Data collectors
    np_images = list() # images in numpy.array
    pil_images = list() # images in PIL.Image

    # First image is all black.
    np_image_grey = np.zeros((w, h))
    #np_images.append(np_image_grey)
    #pil_images.append(PIL.Image.fromarray(np_image_grey))
    #np_image_RGBA = np.zeros((w, h, 4))
    #np_images.append(np_image_RGBA)
    pil_image = PIL.Image.fromarray(np_image_grey).convert('RGBA')
    pil_images.append(pil_image)
    np_images.append(np.asarray(pil_image))

    # Rasterize each moment.
    for j in range(1, len(df_dots)):
        rows = df_dots.loc[:j]

        with Drawing() as draw:
            # Drawing preset
            draw.stroke_width = stroke_width
            draw.stroke_color = color_stroke
            draw.fill_color = color_fill
            draw.stroke_line_join = 'round'

            # Drawing
            draw.path_start()
            prev_x, prev_y, prev_eos, prev_eod = rows.loc[0]
            prev_x, prev_y = int(prev_x), int(prev_y)
            #prev_s, prev_t, prev_x, prev_y = rows.loc[0]
            #prev_s, prev_x, prev_y = int(prev_s), int(prev_x), int(prev_y)
            draw.path_move(to=(prev_x, prev_y))
            for i in range(1, len(rows)):
                x, y, eos, eod = rows.loc[i]
                x, y = int(x), int(y)
                #s, t, x, y = rows.loc[i]
                #s, x, y = int(s), int(x), int(y)

                if (prev_eos, eos) == (1,0):
                #if (prev_s == i_stroke_hover) and (prev_s != s):
                    # Start a new sub-path right after the last hovering step.
                    # (old) Start a new sub-path at the last hovering step.
                    draw.path_move(to=(x, y))
                elif eos == 1:
                #elif (prev_s == i_stroke_hover) or (s == i_stroke_hover):
                    # Draw nothing at hovering steps
                    # (old) Draw nothing at hovering steps except the last.
                    pass
                else:
                    draw.path_line(to=(x, y))
                prev_x, prev_y, prev_eos = x, y, eos
                #prev_s, prev_x, prev_y = s, x, y

                if eod == 1 and eod_stop:
                    break

            draw.path_move(to=(prev_x, prev_y))
            draw.path_finish()

            # Drawing 2nd ######################################################
            # Drawing preset

            point_width = stroke_width / 5
            draw.stroke_width = point_width / 4
            draw.stroke_color = Color('red')
            draw.fill_color = Color('red')

            # Drawing
            #draw.path_start()
            prev_x, prev_y, prev_eos, prev_eod = rows.loc[0]
            prev_x, prev_y = int(prev_x), int(prev_y)
            #prev_s, prev_t, prev_x, prev_y = rows.loc[0]
            #prev_s, prev_x, prev_y = int(prev_s), int(prev_x), int(prev_y)
            #draw.path_move(to=(prev_x, prev_y))
            for i in range(1, len(rows)):
                x, y, eos, eod = rows.loc[i]
                x, y = int(x), int(y)
                #s, t, x, y = rows.loc[i]
                #s, x, y = int(s), int(x), int(y)

                '''if (prev_eos, eos) == (1,0):
                #if (prev_s == i_stroke_hover) and (prev_s != s):
                    # Start a new sub-path right after the last hovering step.
                    # (old) Start a new sub-path at the last hovering step.
                    draw.path_move(to=(x, y))
                elif eos == 1:
                #elif (prev_s == i_stroke_hover) or (s == i_stroke_hover):
                    # Draw nothing at hovering steps
                    # (old) Draw nothing at hovering steps except the last.
                    pass
                else:
                    draw.path_line(to=(x, y))'''
                if eos == 1:
                #elif (prev_s == i_stroke_hover) or (s == i_stroke_hover):
                    # Draw nothing at hovering steps
                    # (old) Draw nothing at hovering steps except the last.
                    draw.stroke_width = point_width / 4 #* 8
                    draw.stroke_color = Color('cyan')
                    draw.fill_color = Color('cyan')
                    draw.circle(origin=(x, y), perimeter=(x+point_width/4,y+point_width/4))
                else:
                    draw.stroke_width = point_width / 4
                    draw.stroke_color = Color('red')
                    draw.fill_color = Color('red')
                #draw.path_line(to=(x, y))
                #draw.point(x, y)
                draw.circle(origin=(x, y), perimeter=(x+point_width/4,y+point_width/4))
                #prev_x, prev_y, prev_eos = x, y, eos
                #prev_s, prev_x, prev_y = s, x, y

                if eod == 1 and eod_stop:
                    break

            #draw.path_move(to=(prev_x, prev_y))
            #draw.path_finish()

            # Save drawing as a raster image.
            with Image(width=w, height=h, background=color_background) as image:
                draw(image)
                # np_image.shape == (w, h, 4)
                # 4: RGBA channel. RGBA vector = (R, G, B, A)
                # Each value is an integer and ranges from 0 to 255.
                np_image_RGBA = np.array(image)
                #np_image_grey = RGB2Grey(np_image_RGBA)
                #np_images.append(np_image_grey)
                np_images.append(np_image_RGBA)

                #pil_images.append(PIL.Image.fromarray(np_image_grey))
                pil_images.append(PIL.Image.fromarray(np_image_RGBA))

        if eod == 1 and eod_stop:
            break

    # Save PIL.Image images into a GIF file.
    #with PIL.Image.new('L', (w, h)) as im:
    with PIL.Image.new('RGBA', (w, h)) as im:
        im.save(gif_path, save_all=True, append_images=pil_images,
            duration=sampling_length, loop=0)
        #print('Saved the GIF format of the raster image sequence in', gif_path)

    # Save PIL.Image image into a PNG file.
    #pil_images[-1].convert('L').save(png_path)
    pil_images[-1].convert('RGBA').save(png_path)
    #print('Saved the PNG format of the raster image sequence in', png_path)

    # Return a sequence of images during handwriting.
    np_images = np.stack(np_images)

    return np_images

def get_dot_seq_from_dx_dy(df_v, w_img, h_img, dt):
    seq_len = df_v.shape[0]
    x, y = 0, 0
    x_seq = np.zeros(seq_len)
    y_seq = np.zeros(seq_len)
    eos_seq = df_v.eos_seq
    eod_seq = df_v.eod_seq
    for t, (dx, dy, eos, eod) in df_v.iterrows():
        #dx, dy = get_cartesian_dot(d, s)
        x += dx; y += dy
        #x += dx*dt; y += dy*dt
        x_seq[t] = x; y_seq[t] = y
    # Move the trajectory to be on the center.
    x_center = (x_seq.max() + x_seq.min()) // 2
    y_center = (y_seq.max() + y_seq.min()) // 2
    x_mv = w_img // 2 - x_center
    y_mv = h_img // 2 - y_center
    x_seq += x_mv
    y_seq += y_mv

    df_dots = pd.DataFrame({'x_seq':x_seq, 'y_seq':y_seq, 'eos_seq':eos_seq, 'eod_seq':eod_seq})

    return df_dots

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert dot trajectories to raster images.')
    parser.add_argument('--data_path', type=str, required=True,
        help='Data directory where dot trajectories are stored.')
    parser.add_argument('--save_dir', type=str, required=True,
        help='Data directory where raster images are stored.')
    parser.add_argument('--sampling_length', type=int, default=20,
        help='Time length of a sample. millisecond.')
    parser.add_argument('--stroke_width', type=int, default=10,
        help='Stroke width')
    parser.add_argument('--remove_hover', action='store_true',
        help='Remove hovering moments.')
    parser.add_argument('--raw_movement', action='store_true',
        help='Use raw movment; otherwise, use sampled movemt.')
    parser.add_argument('--mnist', action='store_true',
        help='Resize images of the MNIST size.')
    parser.add_argument('--start_on_center', action='store_true',
        help='Start drawing on the center of the image.')
    args = parser.parse_args()

    if os.path.isdir(args.data_path):
        csv_paths = [os.path.join(args.data_path, f)
            for f in os.listdir(args.data_path)
            if os.path.isfile(os.path.join(args.data_path, f))]
    else:
        csv_paths = [args.data_path]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for csv_path in tqdm(csv_paths):
        rasterize(csv_path, args.save_dir, args.sampling_length,
            args.stroke_width, args.remove_hover, args.raw_movement,
            args.mnist, args.start_on_center)
