import numpy as np
import pandas as pd
import PIL.Image
import os
import tensorflow as tf
import yaml
import cv2
from copy import deepcopy
from sklearn.utils import shuffle
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
from scipy.stats import norm
from tqdm import tqdm
from utils import get_batch_iterations, sample_gmm_seqs_tf, parse_inputs

class DataLoader():
    def __init__(self, hparams):
        # Get holistic hyperparamters.
        self.hp = hparams
        self.path_velocity_dataset = self.hp['path_velocity_dataset']
        self.path_home_dataset = self.hp['path_home_dataset']
        self.character_set = list(self.hp['character_set'])
        self.delayed_steps = self.hp['delayed_steps']
        self.delayed_steps_ctrl = self.hp['delayed_steps_ctrl']
        self.target = self.hp['target']
        self.input_kine = self.hp['input_kine']
        self.w_img = self.hp['w_img']
        self.h_img = self.hp['h_img']

        self.splits = ['train', 'val', 'test']

        # Get inferred parameters.
        means, stds = self.compute_norm_stat()
        self.dx_mean, self.dy_mean, self.d2x_mean, self.d2y_mean = means
        self.dx_std, self.dy_std, self.d2x_std, self.d2y_std = stds
        self.dx_norm, self.dy_norm, self.d2x_norm, self.d2y_norm = self.get_norms()

        self.datasets = self.load_datasets()

    def get_norms(self):
        s = self.hp['scale']
        if not self.hp['dxdy_normalization']:
            return 1, 1, 1, 1
        elif self.hp['std_scale']:
            return s * self.dx_std, s * self.dy_std, s * self.d2x_std, s * self.d2y_std
        else:
            return s, s, s, s

    def get_character_index(self, character):
        i = self.character_set.index(character)
        return i

    def normalize_position(self, x, y):
        w_half = self.w_img / 2
        h_half = self.h_img / 2
        x = (x - x[0]) / w_half
        y = (y - y[0]) / h_half

        return x, y

    def compute_norm_stat(self):
        # Compute normalization statistics of kinematics
        df_csv = pd.read_csv(self.path_velocity_dataset)
        df_csv = df_csv[df_csv.split == 'train']

        dx_seqs = list(); dy_seqs = list()
        d2x_seqs = list(); d2y_seqs = list()

        for i, row in df_csv.iterrows():
            csv_path = os.path.join(self.path_home_dataset, row.csv_path)
            df = pd.read_csv(csv_path)
            sample_len = len(df) - 1 # All zeros are at the first step.

            dx_seq = np.asarray(df.dx)[1:];   dx_seqs.append(dx_seq)
            dy_seq = np.asarray(df.dy)[1:];   dy_seqs.append(dy_seq)
            d2x_seq = np.asarray(df.d2x)[1:]; d2x_seqs.append(d2x_seq)
            d2y_seq = np.asarray(df.d2y)[1:]; d2y_seqs.append(d2y_seq)

        dx_seqs = np.concatenate(dx_seqs); dy_seqs = np.concatenate(dy_seqs)
        d2x_seqs = np.concatenate(d2x_seqs); d2y_seqs = np.concatenate(d2y_seqs)

        means = (dx_seqs.mean(), dy_seqs.mean(), d2x_seqs.mean(), d2y_seqs.mean())
        stds = (dx_seqs.std(), dy_seqs.std(), d2x_seqs.std(), d2y_seqs.std())

        return means, stds

    def load_datasets(self):
        datasets = dict()
        for split in self.splits:
            df = pd.read_csv(self.path_velocity_dataset)
            df.character = df.character.astype('string')
            df = df[df.split == split]
            
            stack_df_char = list()
            for character in sorted(list(self.hp['character_set'])):
                df_char = df[df.character == character].reset_index()
                indexes = df_char.index.values
                if split == 'train':
                    np.random.seed(self.hp['seed'])
                    np.random.shuffle(indexes)
                if isinstance(self.hp['{}_used_ratio'.format(split)], int):
                    indexes = indexes[:self.hp['{}_used_ratio'.format(split)]]
                else:
                    indexes = indexes[:int(len(indexes)*self.hp['{}_used_ratio'.format(split)])]
                df_char = df_char.iloc[indexes]
                stack_df_char.append(df_char)
            df = pd.concat(stack_df_char, ignore_index=True)

            str_verb = {'train':'training', 'val':'validating', 'test':'testing'}
            print('Generating the Cartesian velocity dataset for {} neural networks ...'.format(str_verb[split]))
            inputs, targets, seq_lengths, characters = self.generate_dataset_cartesian(df)

            inputs = np.asarray(inputs, dtype=object)
            targets = np.asarray(targets, dtype=object)
            seq_lengths = np.asarray(seq_lengths, dtype=object)
            characters = np.asarray(characters, dtype=object)

            datasets[split] = (inputs, targets, seq_lengths, characters)

        return datasets

    def load_dataset(self, split):
        return self.datasets[split]

    def generate_dataset_cartesian(self, df):
        csv_paths = df.csv_path
        input_dim = self.hp['input_dim']
        v_character_dim = self.hp['v_character_dim']

        inputs = list()
        targets = list()
        seq_lengths = list()
        characters = list()


        for i_sample, csv_path in tqdm(enumerate(csv_paths), total=len(csv_paths)):
            csv_path = os.path.join(self.path_home_dataset, csv_path)
            df = pd.read_csv(csv_path)
            sample_len = len(df) - 1 # All zeros are at the first step.

            # Make the target sequence of each sample.
            character = get_character_from_csv_path(csv_path)
            if character not in self.character_set:
                continue
            character_vector = get_character_vector(character, self.character_set)
            x_seq, y_seq = self.normalize_position(np.asarray(df.x), np.asarray(df.y))
            x_seq = x_seq[1:]; y_seq = y_seq[1:]
            dx_seq = np.asarray(df.dx)[1:] / self.dx_norm
            dy_seq = np.asarray(df.dy)[1:] / self.dy_norm
            d2x_seq = np.asarray(df.d2x)[1:] / self.d2x_norm
            d2y_seq = np.asarray(df.d2y)[1:] / self.d2y_norm
            h_seq = np.asarray(df.hover)[1:]
            eod_seq = np.zeros(sample_len)
            eod_seq[-1] = 1 # eod: 1 at the End Of Drawing. Otherwise, 0.

            if self.target == 'v':
                target_seq = np.stack([dx_seq, dy_seq, h_seq, eod_seq]).T
            elif self.target == 'a':
                target_seq = np.stack([d2x_seq, d2y_seq, h_seq, eod_seq]).T
            else:
                exit()

            # Make the input sequence of each sample.
            input_seq = np.zeros((sample_len, input_dim))
            first_in_vector = np.zeros(input_dim)
            first_in_vector = np.concatenate(
                (np.zeros(input_dim-v_character_dim),
                 get_character_vector(character, self.character_set)))

            input_seq[0,:] = first_in_vector
            for i in range(sample_len-1):
                # Delaying inputs == Filling in initial inputs with the null vector
                if i < self.delayed_steps_ctrl:
                    input_seq[i+1,:] = first_in_vector
                    continue
                j = i - self.delayed_steps_ctrl

                in_stack = list()
                if 'p' in self.input_kine:
                    in_stack += [x_seq[j:j+1], y_seq[j:j+1]]
                if 'v' in self.input_kine:
                    in_stack += [dx_seq[j:j+1], dy_seq[j:j+1]]
                if 'a' in self.input_kine:
                    in_stack += [d2x_seq[j:j+1], d2y_seq[j:j+1]]
                in_stack += [h_seq[j:j+1], eod_seq[j:j+1], character_vector]
                in_vector = np.concatenate(in_stack)
                input_seq[i+1,:] = in_vector


            # Stack target and input sequences.
            inputs.append(input_seq)
            targets.append(target_seq)
            seq_lengths.append(sample_len)
            characters.append(character)

        return inputs, targets, seq_lengths, characters

    def generate_dataset_for_corrector(self, preds_controller, states_controller, h_seqs_controller, inputs_controller, targets_controller, mask_controller):
        '''
        Generate a dataset for the corrector from controller's outputs and dataset.

        PARAMS
        -----
        preds_controller: The GMM and Bernoulli distributions' parameters of the controller.
          - type: tf.Tensor.
          - shape: [batch_size, max_seq_len, n_gmms*6+2].

        targets_controller: The targets of the controller.
          - type: np.ndarray.
          - shape: [batch_size, max_seq_len, 4].

        seq_lengths_controller: The sequence lengths of the controller.
          - type: np.ndarray.
          - shape: [batch_size].

        characters_controller: The sequences of character vectors.
          - type: np.ndarray.
          - shape: [batch_size, max_seq_len, vector_dim]

        RETURNS
        -----
        inputs: The inputs of the corrector with delayed feedback.
          - type: np.ndarray.

        targets: The targets of the corrector with delayed feedback.
          - type: np.ndrarray.

        seq_lengths: The sequence lengths of the corrector with delayed feedback.
          - type: np.ndarray.
          - Corrector receives inputs from the moment that delayed feedback appears.
          - (seqeunce_length_corrector) = (sequence_length_controller) - (delayed_steps)
        '''
        preds_controller, states_controller, inputs_controller, targets_controller, mask_controller
        #preds_controller = tf.stop_gradient(preds_controller)
        dx, dy, hov, eod = sample_gmm_seqs_tf(preds_controller, self.hp)
        preds_controller_sampled = np.stack([dx, dy, hov], axis=-1)
        dim_target = 2
        characters_controller = parse_inputs(inputs_controller)[-1]
        ((h1, c1),(h2, c2)) = states_controller
        ##a_controller = preds_controller # The hidden activity of Controller.
        a_controller = h_seqs_controller
        l = preds_controller.shape[1]

        # Reflect feedback delay.
        last_del_steps = 1 + (self.delayed_steps - self.delayed_steps_ctrl)
        dtype =  tf.float32
        if self.hp['correction_input']:
            if self.hp['use_corrector_input_gmm_params']:
                p_delayed = preds_controller_sampled[:,:l-last_del_steps,:]
                if last_del_steps == 1:
                    preds_controller_sampled_delayed = p_delayed
                else:
                    p_silent = np.zeros_like(preds_controller_sampled[:,:last_del_steps-1,:])
                    preds_controller_sampled_delayed = np.concatenate([p_silent, p_delayed], axis=-2)
                characters = characters_controller[:,1:,:]
                a_controller = a_controller[:,1:,:]
                added_terms = preds_controller_sampled_delayed[...,:2]
                #inputs = tf.concat([preds_controller_sampled_delayed, characters, tf.cast(a_controller, dtype)], axis=-1)
                inputs = tf.concat([added_terms, preds_controller_sampled_delayed, characters, tf.cast(a_controller, dtype)], axis=-1)
                targets = inputs_controller[:,1:,:dim_target]
                mask = mask_controller[:,1:]
            else:
                preds_controller_sampled_delayed = preds_controller_sampled[:,:l-last_del_steps,:]
                characters = characters_controller[:,:l-last_del_steps,:]
                added_terms = preds_controller_sampled_delayed[...,:2]
                #inputs = tf.concat([preds_controller_sampled_delayed, characters], axis=-1)
                inputs = tf.concat([added_terms, preds_controller_sampled_delayed, characters], axis=-1)
                targets = inputs_controller[:,last_del_steps:,:dim_target]
                mask = mask_controller[:,last_del_steps:]

        elif self.hp['correction_output']:
            added_terms = preds_controller_sampled[...,:2]
            if self.hp['use_corrector_input_gmm_params']:
                #p_silent = np.zeros_like(preds_controller_sampled[:,:last_del_steps,:])
                #p_delayed = preds_controller_sampled[:,:l-last_del_steps,:]
                #preds_controller_sampled_delayed = np.concatenate([p_silent, p_delayed], axis=-2)
                targets_fb_silent = np.zeros_like(targets_controller[:,:last_del_steps,:3])
                targets_fb_delayed = targets_controller[:,:l-last_del_steps,:3]
                fb_delayed = np.concatenate([targets_fb_silent, targets_fb_delayed], axis=-2)
                characters = characters_controller
                #inputs = tf.concat([preds_controller_sampled_delayed, characters, tf.cast(a_controller, dtype)], axis=-1)
                #inputs = tf.concat([added_terms, preds_controller_sampled_delayed, characters, tf.cast(a_controller, dtype)], axis=-1)
                inputs = tf.concat([added_terms, fb_delayed, characters, tf.cast(a_controller, dtype)], axis=-1)
                #inputs = tf.concat([fb_delayed[...,:2], fb_delayed, characters, tf.cast(a_controller, dtype)], axis=-1)
                #inputs = tf.concat([preds_controller_sampled_delayed[...,:2], preds_controller_sampled_delayed, characters, tf.cast(a_controller, dtype)], axis=-1)
                #inputs = tf.concat([added_terms, preds_controller_sampled, characters, tf.cast(a_controller, dtype)], axis=-1)
                #targets = targets_controller[:,:,:dim_target]
                targets = targets_controller[:,:,:dim_target] - preds_controller_sampled[:,:,:dim_target]
                mask = mask_controller
            else:
                preds_controller_sampled_delayed = preds_controller_sampled[:,:l-last_del_steps,:]
                characters = characters_controller[:,:l-last_del_steps,:]
                mask = mask_controller[:,last_del_steps:]
                inputs = tf.concat([preds_controller_sampled_delayed, characters], axis=-1)
                #inputs = tf.concat([added_terms, preds_controller_sampled_delayed, characters], axis=-1)
                targets = targets_controller[:,last_del_steps:,:dim_target]

        else:
            inputs = inputs_controller
            targets = targets_controller
            mask = mask_controller

        inputs = tf.cast(inputs, tf.float32)
        targets = tf.cast(targets, tf.float32)

        return inputs, targets, mask

    def transform_set_a2v(self, dataset):
        inputs, targets, seq_lengths, characters = dataset
        targets = self.transform_a2v(targets)
        return inputs, targets, seq_lengths, characters

    def transform_a2v(self, targets):
        if isinstance(targets, tf.Tensor):
            targets = self.transform_a2v_tf(targets)
        elif isinstance(targets, list):
            targets_new = list()
            for i in range(targets.shape[0]):
                targets_new.append(self.transform_a2v_np(targets[i]))
            targets = targets_new
        elif (isinstance(targets, np.ndarray) and targets.ndim == 1):
            targets_new = targets.copy()
            for i in range(targets.shape[0]):
                targets_new[i] = self.transform_a2v_np(targets[i])
            targets = targets_new
        elif isinstance(targets, np.ndarray):
                targets = self.transform_a2v_np(targets)
        else:
            exit()
        return targets

    def transform_a2v_tf(self, targets):
        '''
        targets
        - shape: [batch_size, max_seq_len, dim] or [max_seq_len, dim]
        - dim: 4
        '''
        if self.target == 'a':
            dx = tf.cumsum(targets[...,0:1] * self.d2x_norm, axis=-2) / self.dx_norm
            dy = tf.cumsum(targets[...,1:2] * self.d2y_norm, axis=-2) / self.dy_norm
            targets = tf.concat([dx, dy, targets[...,2:]], axis=-1)
        return targets

    def transform_a2v_np(self, targets):
        '''
        targets
        - shape: [batch_size, max_seq_len, dim] or [max_seq_len, dim]
        - dim: 4
        '''
        if self.target == 'a':
            dx = np.cumsum(targets[...,0:1] * self.d2x_norm, axis=-2) / self.dx_norm
            dy = np.cumsum(targets[...,1:2] * self.d2y_norm, axis=-2) / self.dy_norm
            targets = np.concatenate([dx, dy, targets[...,2:]], axis=-1)
        return targets

class BatchGenerator():
    def __init__(self, hparams):
        self.batch_size = hparams['batch_size']
        self.mask_value = hparams['mask_value']
        self.degree_random_rotation = hparams['degree_random_rotation']
        self.p_mask_input_dxdy = hparams['p_mask_input_dxdy_train']

    def __call__(self, x, y):
        batch_size = self.batch_size
        mask_value = self.mask_value

        seq_lengths = np.asarray([len(i) for i in x])
        x, y, seq_lengths = shuffle(x, y, seq_lengths)

        n_samples = len(x)
        iterations = get_batch_iterations(n_samples, batch_size)

        for i in range(iterations):
            if (i+1) * batch_size > n_samples:
                x_batch, y_batch, sl_batch = x[i*batch_size:], y[i*batch_size:], seq_lengths[i*batch_size:]
            else:
                x_batch, y_batch, sl_batch = x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size], seq_lengths[i*batch_size:(i+1)*batch_size]

            # Data augmentation
            x_batch_aug, y_batch_aug = rotate_batch(self.degree_random_rotation, x_batch, y_batch)

            # Mask dxdy in the inputs
            x_batch_aug = mask_dxdy_batch(x_batch_aug, self.p_mask_input_dxdy)

            x_batch_padded = tf.keras.preprocessing.sequence.pad_sequences(
                x_batch_aug, padding="post", dtype='float32', value=mask_value)
            y_batch_padded = tf.keras.preprocessing.sequence.pad_sequences(
                y_batch_aug, padding="post", dtype='float32', value=mask_value)
            mask_batch = tf.sequence_mask(sl_batch, dtype=tf.float32)

            yield x_batch_padded, y_batch_padded, sl_batch, mask_batch

def import_hparams(yml_path, v_type='cartesian'):
    with open(yml_path) as f:
        hparams = yaml.load(f,  Loader=yaml.FullLoader)
    # Store inferred hyperparamters
    hparams['v_character_dim'] = len(hparams['character_set'])
    if v_type == 'cartesian':
        hparams['input_dim'] = (hparams['v_dx_dim'] + hparams['v_dy_dim']) * len(hparams['input_kine'])
        hparams['input_dim'] += hparams['v_hover_dim'] + hparams['v_eod_dim'] + hparams['v_character_dim']
    elif v_type == 'polar':
        hparams['input_dim'] = hparams['v_direction_dim'] + hparams['v_speed_dim'] + hparams['v_hover_dim'] + hparams['v_eod_dim'] + hparams['v_character_dim']
    hparams['g_mixtures_dim'] = 6 * hparams['n_g_mixtures'] # These 2D Gaussians predict dx and dy at each step.
    hparams['output_dim'] = hparams['g_mixtures_dim'] + hparams['v_hover_dim'] + hparams['v_eod_dim']

    return hparams

def get_hovering_sequence(sample):
    hovering_sequence = np.zeros(len(sample['time'])) # Default. None hovering moments.
    hovering_sequence[sample['hover_indexes']] = 1 # Hovering moment.
    return hovering_sequence

def get_direction_vector(degree, min_theta=-180, max_theta=180, d_theta=15, std=9):
    sensing_thetas = sorted(np.arange(max_theta, min_theta, -d_theta))
    theta_vector = [norm_circular_fn(degree, min_theta, max_theta, sensing_theta, std) for sensing_theta in sensing_thetas]
    theta_vector = np.asarray(theta_vector)
    theta_vector /= norm.pdf(0, 0, std)
    return theta_vector, sensing_thetas

def get_speed_vector(speed, min_s=0, max_s=3, n_sensors=24, std=0.09):
    speed = np.clip(speed, min_s, max_s)
    d_speed = (max_s - min_s) / n_sensors
    sensing_speeds = sorted(np.arange(max_s, min_s, -d_speed))
    speed_vector = [norm.pdf(speed, sensing_speed, std) for sensing_speed in sensing_speeds]
    speed_vector = np.asarray(speed_vector)
    speed_vector /= norm.pdf(0, 0, std)
    return speed_vector, sensing_speeds

def get_character_vector(character, character_set):
    '''Assume we only use characters from 0 to 9 and character vectors are one-hot.'''
    character_vector = np.zeros(len(character_set))
    character_vector[character_set.index(character)] = 1.0
    return character_vector

def get_character_from_vector(character_vector, character_set):
    character = character_set[np.argmax(character_vector)]
    return character

def get_character_from_csv_path(csv_path):
    character = os.path.splitext(os.path.split(csv_path)[1])[0].split('_')[2]
    return character

def norm_circular_fn(x, x_min, x_max, loc, scale):
    if (x_max - loc) == (loc - x_min):
        y = norm.pdf(x, loc, scale)
    elif (x_max - loc) > (loc - x_min):
        if x <= loc + 180:
            y = norm.pdf(x, loc, scale)
        else:
            y = norm.pdf(x - 360, loc, scale)
    else: # (x_max - loc) < (loc - x_min):
        if x > loc - 180:
            y = norm.pdf(x, loc, scale)
        else:
            y = norm.pdf(x + 360, loc, scale)
    return y

def norm_circular(x, x_min, x_max, loc, scale):
    y = np.array([norm_circular_fn(x_i, x_min, x_max, loc, scale) for x_i in x])
    return y

def RGB2Grey(np_image):
    '''A widely used RGB2Grey conversion equation'''
    R = np_image[:,:,0]
    G = np_image[:,:,1]
    B = np_image[:,:,2]
    Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return Y

def rasterize(df_dots, w, h, sampling_length, stroke_width, gif_path=None, png_path=None, mnist=False):
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
    w = int(w); h = int(h)

    # Resize the vector image to be used togther with the MNIST data.
    if mnist:
        stroke_width = 2
        w = h = 28
        resize_vector_img(df_dots, h_target=h, w_target=w, pad=4) # inplace operation.

    # Color settings
    color_background = Color('black')
    color_stroke = Color('white')
    color_fill = Color('none')

    # Data collectors
    np_images = list() # images in numpy.array
    pil_images = list() # images in PIL.Image

    # First image is all black.
    np_image_grey = np.zeros((w, h))
    np_images.append(np_image_grey)
    pil_images.append(PIL.Image.fromarray(np_image_grey))

    # Rasterize each moment.
    for j in range(1, len(df_dots)):
        rows = df_dots.loc[:j]

        with Drawing() as draw:
            # Drawing preset
            draw.stroke_width = stroke_width
            draw.stroke_color = color_stroke
            draw.fill_color = color_fill

            # Drawing
            draw.path_start()
            prev_x, prev_y, prev_hov, prev_eod = rows.loc[0]
            prev_x, prev_y = int(prev_x), int(prev_y)
            #prev_s, prev_t, prev_x, prev_y = rows.loc[0]
            #prev_s, prev_x, prev_y = int(prev_s), int(prev_x), int(prev_y)
            draw.path_move(to=(prev_x, prev_y))
            for i in range(1, len(rows)):
                x, y, hov, eod = rows.loc[i]
                x, y = int(x), int(y)
                #s, t, x, y = rows.loc[i]
                #s, x, y = int(s), int(x), int(y)

                if (prev_hov, hov) == (1,0):
                #if (prev_s == i_stroke_hover) and (prev_s != s):
                    # Start a new sub-path right after the last hovering step.
                    # (old) Start a new sub-path at the last hovering step.
                    draw.path_move(to=(x, y))
                elif hov == 1:
                #elif (prev_s == i_stroke_hover) or (s == i_stroke_hover):
                    # Draw nothing at hovering steps
                    # (old) Draw nothing at hovering steps except the last.
                    pass
                else:
                    draw.path_line(to=(x, y))
                prev_x, prev_y, prev_hov = x, y, hov
                #prev_s, prev_x, prev_y = s, x, y

                if eod == 1:
                    break
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
                pil_images.append(PIL.Image.fromarray(np_image_grey))

    np_images = np.stack(np_images)

    if mnist:
        np_images = resize_and_pad_raster_img(np_images, w=28, h=28, pad_width=4)

        # Move the center of mass to be at the center of the last image.
        h_diff, w_diff = get_diff_to_center_of_mass(np_images[-1])
        np_images = np.asarray([shift(np_image, h_diff, w_diff) for np_image in np_images])
        pil_images = [PIL.Image.fromarray(np_image) for np_image in np_images]

    # Save PIL.Image images into a GIF file.
    if gif_path is not None:
        with PIL.Image.new('L', (w, h)) as im:
            im.save(gif_path, save_all=True, append_images=pil_images,
                duration=sampling_length, loop=0)
            #print('Saved the GIF format of the raster image sequence in', gif_path)

    # Save PIL.Image image into a PNG file.
    if png_path is not None:
        pil_images[-1].convert('L').save(png_path)
        #print('Saved the PNG format of the raster image sequence in', png_path)

    # Return a sequence of images during handwriting.
    #np_images = np.stack(np_images)

    return np_images

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
            prev_x, prev_y, prev_hov, prev_eod = rows.loc[0]
            prev_x, prev_y = int(prev_x), int(prev_y)
            #prev_s, prev_t, prev_x, prev_y = rows.loc[0]
            #prev_s, prev_x, prev_y = int(prev_s), int(prev_x), int(prev_y)
            draw.path_move(to=(prev_x, prev_y))
            for i in range(1, len(rows)):
                x, y, hov, eod = rows.loc[i]
                x, y = int(x), int(y)
                #s, t, x, y = rows.loc[i]
                #s, x, y = int(s), int(x), int(y)

                if (prev_hov, hov) == (1,0):
                #if (prev_s == i_stroke_hover) and (prev_s != s):
                    # Start a new sub-path right after the last hovering step.
                    # (old) Start a new sub-path at the last hovering step.
                    draw.path_move(to=(x, y))
                elif hov == 1:
                #elif (prev_s == i_stroke_hover) or (s == i_stroke_hover):
                    # Draw nothing at hovering steps
                    # (old) Draw nothing at hovering steps except the last.
                    pass
                else:
                    draw.path_line(to=(x, y))
                prev_x, prev_y, prev_hov = x, y, hov
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
            prev_x, prev_y, prev_hov, prev_eod = rows.loc[0]
            prev_x, prev_y = int(prev_x), int(prev_y)
            #prev_s, prev_t, prev_x, prev_y = rows.loc[0]
            #prev_s, prev_x, prev_y = int(prev_s), int(prev_x), int(prev_y)
            #draw.path_move(to=(prev_x, prev_y))
            for i in range(1, len(rows)):
                x, y, hov, eod = rows.loc[i]
                x, y = int(x), int(y)
                #s, t, x, y = rows.loc[i]
                #s, x, y = int(s), int(x), int(y)

                '''if (prev_hov, hov) == (1,0):
                #if (prev_s == i_stroke_hover) and (prev_s != s):
                    # Start a new sub-path right after the last hovering step.
                    # (old) Start a new sub-path at the last hovering step.
                    draw.path_move(to=(x, y))
                elif hov == 1:
                #elif (prev_s == i_stroke_hover) or (s == i_stroke_hover):
                    # Draw nothing at hovering steps
                    # (old) Draw nothing at hovering steps except the last.
                    pass
                else:
                    draw.path_line(to=(x, y))'''
                if hov == 1:
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
                #prev_x, prev_y, prev_hov = x, y, hov
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

def rasterize_dev_v2(df_dots, w, h, sampling_length, stroke_width, gif_path, png_path, eod_stop=False,
                     df_corrections=None, corrected_dots=True):
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
            prev_x, prev_y, prev_hov, prev_eod = rows.loc[0]
            prev_x, prev_y = int(prev_x), int(prev_y)
            #prev_s, prev_t, prev_x, prev_y = rows.loc[0]
            #prev_s, prev_x, prev_y = int(prev_s), int(prev_x), int(prev_y)
            draw.path_move(to=(prev_x, prev_y))
            for i in range(1, len(rows)):
                x, y, hov, eod = rows.loc[i]
                x, y = int(x), int(y)
                #s, t, x, y = rows.loc[i]
                #s, x, y = int(s), int(x), int(y)

                if (prev_hov, hov) == (1,0):
                #if (prev_s == i_stroke_hover) and (prev_s != s):
                    # Start a new sub-path right after the last hovering step.
                    # (old) Start a new sub-path at the last hovering step.
                    draw.path_move(to=(x, y))
                elif hov == 1:
                #elif (prev_s == i_stroke_hover) or (s == i_stroke_hover):
                    # Draw nothing at hovering steps
                    # (old) Draw nothing at hovering steps except the last.
                    pass
                else:
                    draw.path_line(to=(x, y))
                prev_x, prev_y, prev_hov = x, y, hov
                #prev_s, prev_x, prev_y = s, x, y

                if eod == 1 and eod_stop:
                    break

            draw.path_move(to=(prev_x, prev_y))
            draw.path_finish()

            # Drawing 2nd ######################################################

            point_width = stroke_width / 5
            draw.stroke_width = point_width

            for i in range(len(rows)):
                x, y, hov, eod = rows.loc[i]
                x, y = int(x), int(y)

                if hov == 1:
                    # Draw nothing at hovering steps
                    draw.stroke_color = Color('cyan')
                    draw.fill_color = Color('cyan')
                    draw.circle(origin=(x, y), perimeter=(x+point_width/4,y+point_width/4))
                else:
                    draw.stroke_color = Color('black')
                    draw.fill_color = Color('black')
                    draw.circle(origin=(x, y), perimeter=(x+point_width/4,y+point_width/4))

            if (df_corrections is not None) and corrected_dots:

                for i in range(1, len(rows)):
                    prev_x, prev_y, prev_hov, prev_eod = rows.loc[i-1]
                    prev_x, prev_y = int(prev_x), int(prev_y)
                    x, y, hov, eod = rows.loc[i]
                    x, y = int(x), int(y)

                    dx = x - prev_x; dy = y - prev_y
                    dx_cr = df_corrections.loc[i-1].dx; dy_cr = df_corrections.loc[i-1].dy
                    dx_rnn = dx - dx_cr; dy_rnn = dy - dy_cr

                    draw.stroke_color = Color('lime')
                    draw.line((prev_x,prev_y), (prev_x+dx_rnn, prev_y+dy_rnn))

                for i in range(1, len(rows)):
                    prev_x, prev_y, prev_hov, prev_eod = rows.loc[i-1]
                    prev_x, prev_y = int(prev_x), int(prev_y)
                    x, y, hov, eod = rows.loc[i]
                    x, y = int(x), int(y)

                    dx = x - prev_x; dy = y - prev_y
                    dx_cr = df_corrections.loc[i-1].dx; dy_cr = df_corrections.loc[i-1].dy
                    dx_rnn = dx - dx_cr; dy_rnn = dy - dy_cr

                    draw.stroke_color = Color('red')
                    draw.line((prev_x+dx_rnn, prev_y+dy_rnn), (prev_x+dx_rnn+dx_cr, prev_y+dy_rnn+dy_cr))

            if (df_corrections is not None) and (not corrected_dots):

                for i in range(1, len(rows)):
                    x, y, hov, eod = rows.loc[i]
                    x, y = int(x), int(y)

                    dx_cr = df_corrections.loc[i-1].dx; dy_cr = df_corrections.loc[i-1].dy

                    draw.stroke_color = Color('red')
                    draw.line((x, y), (x+dx_cr, y+dy_cr))

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

def rotate_batch(degree, x_batch, y_batch):
    if degree == 0:
        return x_batch, y_batch
    angle = np.deg2rad(degree)
    angle_random = np.random.uniform(-angle, angle)
    x_batch_rot = deepcopy(x_batch)
    y_batch_rot = deepcopy(y_batch)
    for i in range(len(x_batch)):
        # dx == x[:,0]; dy == x[:,1]
        # dx == y[:,0]; dy == y[:,1]
        # dx = dx * cos(angle) - dy * sin(angle)
        x_batch_rot[i][:,0] = x_batch[i][:,0] * tf.math.cos(angle_random) - x_batch[i][:,1] * tf.math.sin(angle_random)
        y_batch_rot[i][:,0] = y_batch[i][:,0] * tf.math.cos(angle_random) - y_batch[i][:,1] * tf.math.sin(angle_random)
        # dy = dx * sin(angle) + dy * cos(angle)
        x_batch_rot[i][:,1] = x_batch[i][:,0] * tf.math.sin(angle_random) + x_batch[i][:,1] * tf.math.cos(angle_random)
        y_batch_rot[i][:,1] = y_batch[i][:,0] * tf.math.sin(angle_random) + y_batch[i][:,1] * tf.math.cos(angle_random)

    return x_batch_rot, y_batch_rot

def mask_dxdy_batch(x_batch, p_mask_dxdy):
    x_batch_masked = deepcopy(x_batch)
    for i in range(len(x_batch)):
        # dx == x[:,0]; dy == x[:,1]
        # dx == y[:,0]; dy == y[:,1]
        length = x_batch_masked[i].shape[0]
        indexes_masked = np.where(np.random.binomial(1, p_mask_dxdy, length) == 1)[0]
        x_batch_masked[i][indexes_masked,0] = 0
        x_batch_masked[i][indexes_masked,1] = 0

    return x_batch_masked

