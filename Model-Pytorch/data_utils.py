import numpy as np
import pandas as pd
import PIL.Image
import os
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import yaml
import cv2
from copy import deepcopy
from sklearn.utils import shuffle
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
from scipy.stats import norm
from tqdm import tqdm

from utils import get_batch_iterations, sample_gmm_seqs_pt, parse_inputs_pt

class DataLoader():
    def __init__(self, hparams):
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
        return self.character_set.index(character)

    def normalize_position(self, x, y):
        w_half = self.w_img / 2
        h_half = self.h_img / 2
        x = (x - x[0]) / w_half
        y = (y - y[0]) / h_half
        return x, y

    def compute_norm_stat(self):
        df_csv = pd.read_csv(self.path_velocity_dataset)
        df_csv = df_csv[df_csv.split == 'train']

        dx_seqs, dy_seqs = [], []
        d2x_seqs, d2y_seqs = [], []

        for i, row in df_csv.iterrows():
            csv_path = os.path.join(self.path_home_dataset, row.csv_path)
            df = pd.read_csv(csv_path)

            dx_seqs.append(np.asarray(df.dx)[1:])
            dy_seqs.append(np.asarray(df.dy)[1:])
            d2x_seqs.append(np.asarray(df.d2x)[1:])
            d2y_seqs.append(np.asarray(df.d2y)[1:])

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
                if isinstance(self.hp[f'{split}_used_ratio'], int):
                    indexes = indexes[:self.hp[f'{split}_used_ratio']]
                else:
                    indexes = indexes[:int(len(indexes)*self.hp[f'{split}_used_ratio'])]
                df_char = df_char.iloc[indexes]
                stack_df_char.append(df_char)
            df = pd.concat(stack_df_char, ignore_index=True)

            str_verb = {'train':'training', 'val':'validating', 'test':'testing'}
            print(f'Generating the Cartesian velocity dataset for {str_verb[split]} neural networks ...')
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

        inputs, targets, seq_lengths, characters = [], [], [], []

        for i_sample, csv_path in tqdm(enumerate(csv_paths), total=len(csv_paths)):
            csv_path = os.path.join(self.path_home_dataset, csv_path)
            df = pd.read_csv(csv_path)
            sample_len = len(df) - 1 

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
            eod_seq[-1] = 1 

            if self.target == 'v':
                target_seq = np.stack([dx_seq, dy_seq, h_seq, eod_seq]).T
            elif self.target == 'a':
                target_seq = np.stack([d2x_seq, d2y_seq, h_seq, eod_seq]).T

            input_seq = np.zeros((sample_len, input_dim))
            first_in_vector = np.concatenate((np.zeros(input_dim-v_character_dim), character_vector))
            input_seq[0,:] = first_in_vector
            
            for i in range(sample_len-1):
                if i < self.delayed_steps_ctrl:
                    input_seq[i+1,:] = first_in_vector
                    continue
                j = i - self.delayed_steps_ctrl

                in_stack = list()
                if 'p' in self.input_kine: in_stack += [x_seq[j:j+1], y_seq[j:j+1]]
                if 'v' in self.input_kine: in_stack += [dx_seq[j:j+1], dy_seq[j:j+1]]
                if 'a' in self.input_kine: in_stack += [d2x_seq[j:j+1], d2y_seq[j:j+1]]
                in_stack += [h_seq[j:j+1], eod_seq[j:j+1], character_vector]
                input_seq[i+1,:] = np.concatenate(in_stack)

            inputs.append(input_seq)
            targets.append(target_seq)
            seq_lengths.append(sample_len)
            characters.append(character)

        return inputs, targets, seq_lengths, characters

    def generate_dataset_for_corrector(self, preds_controller, states_controller, h_seqs_controller, inputs_controller, targets_controller, mask_controller):
        # PyTorch 기반 리팩토링 (tf 의존성 제거)
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
        dx, dy, hov, eod = sample_gmm_seqs_pt(preds_controller, self.hp)
        preds_controller_sampled = torch.stack([dx, dy, hov], dim=-1)
        dim_target = 2
        
        characters_controller = parse_inputs_pt(inputs_controller)[-1]
        a_controller = h_seqs_controller
        l = preds_controller.shape[1]

        last_del_steps = 1 + (self.delayed_steps - self.delayed_steps_ctrl)
        
        if self.hp['correction_input']:
            if self.hp['use_corrector_input_gmm_params']:
                p_delayed = preds_controller_sampled[:, :l-last_del_steps, :]
                if last_del_steps == 1:
                    preds_controller_sampled_delayed = p_delayed
                else:
                    p_silent = torch.zeros_like(preds_controller_sampled[:, :last_del_steps-1, :])
                    preds_controller_sampled_delayed = torch.cat([p_silent, p_delayed], dim=-2)
                    
                characters = characters_controller[:, 1:, :]
                a_controller = a_controller[:, 1:, :]
                added_terms = preds_controller_sampled_delayed[..., :2]
                
                inputs = torch.cat([added_terms, preds_controller_sampled_delayed, characters, a_controller.to(torch.float32)], dim=-1)
                targets = inputs_controller[:, 1:, :dim_target]
                mask = mask_controller[:, 1:]
            else:
                preds_controller_sampled_delayed = preds_controller_sampled[:, :l-last_del_steps, :]
                characters = characters_controller[:, :l-last_del_steps, :]
                added_terms = preds_controller_sampled_delayed[..., :2]
                
                inputs = torch.cat([added_terms, preds_controller_sampled_delayed, characters], dim=-1)
                targets = inputs_controller[:, last_del_steps:, :dim_target]
                mask = mask_controller[:, last_del_steps:]

        elif self.hp['correction_output']:
            added_terms = preds_controller_sampled[..., :2]
            if self.hp['use_corrector_input_gmm_params']:
                targets_fb_silent = torch.zeros_like(targets_controller[:, :last_del_steps, :3])
                targets_fb_delayed = targets_controller[:, :l-last_del_steps, :3]
                fb_delayed = torch.cat([targets_fb_silent, targets_fb_delayed], dim=-2)
                characters = characters_controller
                
                ############# 
                inputs = torch.cat([added_terms, fb_delayed, characters, a_controller.to(torch.float32)], dim=-1)
                targets = targets_controller[:, :, :dim_target] - preds_controller_sampled[:, :, :dim_target]
                
                mask = mask_controller
            else:
                preds_controller_sampled_delayed = preds_controller_sampled[:, :l-last_del_steps, :]
                characters = characters_controller[:, :l-last_del_steps, :]
                mask = mask_controller[:, last_del_steps:]
                
                inputs = torch.cat([preds_controller_sampled_delayed, characters], dim=-1)
                targets = targets_controller[:, last_del_steps:, :dim_target]
        else:
            inputs = inputs_controller
            targets = targets_controller
            mask = mask_controller

        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)

        return inputs, targets, mask

    def transform_set_a2v(self, dataset):
        inputs, targets, seq_lengths, characters = dataset
        targets = self.transform_a2v(targets) ## error
        return inputs, targets, seq_lengths, characters

    def transform_a2v(self, targets):
        if torch.is_tensor(targets):
            targets = self.transform_a2v_pt(targets)
        elif isinstance(targets, list):
            targets = [self.transform_a2v_np(t) for t in targets]
        elif isinstance(targets, np.ndarray) and targets.ndim == 1:
            #print(targets.shape) # (1000,)]
            #temp_list = [self.transform_a2v_np(t) for t in targets]
            #print("length of first 3 data:", [len(x) for x in temp_list[:3]]) # length of first 3 data: [28, 22, 22]
            #targets = np.array([self.transform_a2v_np(t) for t in targets]) ####### dimension error
            targets = np.array([self.transform_a2v_np(t) for t in targets], dtype=object)
        elif isinstance(targets, np.ndarray):
            targets = self.transform_a2v_np(targets)
        return targets

    def transform_a2v_pt(self, targets):
        '''
        targets
        - shape: [batch_size, max_seq_len, dim] or [max_seq_len, dim]
        - dim: 4
        '''
        if self.target == 'a':
            dx = torch.cumsum(targets[..., 0:1] * self.d2x_norm, dim=-2) / self.dx_norm
            dy = torch.cumsum(targets[..., 1:2] * self.d2y_norm, dim=-2) / self.dy_norm
            targets = torch.cat([dx, dy, targets[..., 2:]], dim=-1)
        return targets

    def transform_a2v_np(self, targets):
        '''
        targets
        - shape: [batch_size, max_seq_len, dim] or [max_seq_len, dim]
        - dim: 4
        '''
        if self.target == 'a':
            dx = np.cumsum(targets[..., 0:1] * self.d2x_norm, axis=-2) / self.dx_norm
            dy = np.cumsum(targets[..., 1:2] * self.d2y_norm, axis=-2) / self.dy_norm
            targets = np.concatenate([dx, dy, targets[..., 2:]], axis=-1)
        return targets

class BatchGenerator():
    def __init__(self, hparams):
        self.batch_size = hparams['batch_size']
        self.mask_value = hparams['mask_value']
        self.degree_random_rotation = hparams['degree_random_rotation']
        self.p_mask_input_dxdy = hparams['p_mask_input_dxdy_train']

    def __call__(self, x, y):
        seq_lengths = np.asarray([len(i) for i in x])
        x, y, seq_lengths = shuffle(x, y, seq_lengths)

        n_samples = len(x)
        iterations = get_batch_iterations(n_samples, self.batch_size)

        for i in range(iterations):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            sl_batch = seq_lengths[start_idx:end_idx]

            x_batch_aug, y_batch_aug = rotate_batch(self.degree_random_rotation, x_batch, y_batch)
            x_batch_aug = mask_dxdy_batch(x_batch_aug, self.p_mask_input_dxdy)

            # numpy arrays를 PyTorch Tensor로 변환
            x_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in x_batch_aug]
            y_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in y_batch_aug]

            # PyTorch 내장 pad_sequence 사용 (batch_first=True 적용)
            x_batch_padded = pad_sequence(x_tensors, batch_first=True, padding_value=self.mask_value)
            y_batch_padded = pad_sequence(y_tensors, batch_first=True, padding_value=self.mask_value)

            # Sequence Mask 생성
            max_len = x_batch_padded.size(1)
            sl_tensor = torch.tensor(sl_batch, dtype=torch.long)
            mask_batch = (torch.arange(max_len).expand(len(sl_batch), max_len) < sl_tensor.unsqueeze(1)).to(torch.float32)

            yield x_batch_padded, y_batch_padded, sl_batch, mask_batch

def import_hparams(yml_path, v_type='cartesian'):
    with open(yml_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    hparams['v_character_dim'] = len(hparams['character_set'])
    if v_type == 'cartesian':
        hparams['input_dim'] = (hparams['v_dx_dim'] + hparams['v_dy_dim']) * len(hparams['input_kine'])
        hparams['input_dim'] += hparams['v_hover_dim'] + hparams['v_eod_dim'] + hparams['v_character_dim']
    elif v_type == 'polar':
        hparams['input_dim'] = hparams['v_direction_dim'] + hparams['v_speed_dim'] + hparams['v_hover_dim'] + hparams['v_eod_dim'] + hparams['v_character_dim']
    hparams['g_mixtures_dim'] = 6 * hparams['n_g_mixtures']
    hparams['output_dim'] = hparams['g_mixtures_dim'] + hparams['v_hover_dim'] + hparams['v_eod_dim']
    return hparams

def get_hovering_sequence(sample):
    hovering_sequence = np.zeros(len(sample['time'])) 
    hovering_sequence[sample['hover_indexes']] = 1 
    return hovering_sequence

def get_direction_vector(degree, min_theta=-180, max_theta=180, d_theta=15, std=9):
    sensing_thetas = sorted(np.arange(max_theta, min_theta, -d_theta))
    theta_vector = [norm_circular_fn(degree, min_theta, max_theta, sensing_theta, std) for sensing_theta in sensing_thetas]
    theta_vector = np.asarray(theta_vector) / norm.pdf(0, 0, std)
    return theta_vector, sensing_thetas

def get_speed_vector(speed, min_s=0, max_s=3, n_sensors=24, std=0.09):
    speed = np.clip(speed, min_s, max_s)
    d_speed = (max_s - min_s) / n_sensors
    sensing_speeds = sorted(np.arange(max_s, min_s, -d_speed))
    speed_vector = [norm.pdf(speed, sensing_speed, std) for sensing_speed in sensing_speeds]
    speed_vector = np.asarray(speed_vector) / norm.pdf(0, 0, std)
    return speed_vector, sensing_speeds

def get_character_vector(character, character_set):
    character_vector = np.zeros(len(character_set))
    character_vector[character_set.index(character)] = 1.0
    return character_vector

def get_character_from_vector(character_vector, character_set):
    return character_set[np.argmax(character_vector)]

def get_character_from_csv_path(csv_path):
    return os.path.splitext(os.path.split(csv_path)[1])[0].split('_')[2]

def norm_circular_fn(x, x_min, x_max, loc, scale):
    if (x_max - loc) == (loc - x_min):
        y = norm.pdf(x, loc, scale)
    elif (x_max - loc) > (loc - x_min):
        if x <= loc + 180: y = norm.pdf(x, loc, scale)
        else: y = norm.pdf(x - 360, loc, scale)
    else: 
        if x > loc - 180: y = norm.pdf(x, loc, scale)
        else: y = norm.pdf(x + 360, loc, scale)
    return y

def norm_circular(x, x_min, x_max, loc, scale):
    return np.array([norm_circular_fn(x_i, x_min, x_max, loc, scale) for x_i in x])

def RGB2Grey(np_image):
    R, G, B = np_image[:,:,0], np_image[:,:,1], np_image[:,:,2]
    return 0.2989 * R + 0.5870 * G + 0.1140 * B

# rasterize 및 rasterize_dev는 의존성 이슈가 없는 순수 그리기 로직이므로 그대로 유지
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
    # (원본 로직 생략 없이 그대로 사용)
    w = int(w); h = int(h)
    if mnist:
        stroke_width = 2
        w = h = 28
        resize_vector_img(df_dots, h_target=h, w_target=w, pad=4) 

    color_background = Color('black')
    color_stroke = Color('white')
    color_fill = Color('none')

    np_images, pil_images = list(), list()
    np_image_grey = np.zeros((w, h))
    np_images.append(np_image_grey)
    pil_images.append(PIL.Image.fromarray(np_image_grey))

    for j in range(1, len(df_dots)):
        rows = df_dots.loc[:j]
        with Drawing() as draw:
            draw.stroke_width = stroke_width
            draw.stroke_color = color_stroke
            draw.fill_color = color_fill
            draw.path_start()
            
            prev_x, prev_y, prev_hov, _ = rows.loc[0]
            prev_x, prev_y = int(prev_x), int(prev_y)
            draw.path_move(to=(prev_x, prev_y))
            
            for i in range(1, len(rows)):
                x, y, hov, eod = rows.loc[i]
                x, y = int(x), int(y)
                if (prev_hov, hov) == (1,0): draw.path_move(to=(x, y))
                elif hov == 1: pass
                else: draw.path_line(to=(x, y))
                prev_x, prev_y, prev_hov = x, y, hov
                if eod == 1: break
            draw.path_move(to=(prev_x, prev_y))
            draw.path_finish()

            with Image(width=w, height=h, background=color_background) as image:
                draw(image)
                np_image_RGBA = np.array(image)
                np_image_grey = RGB2Grey(np_image_RGBA)
                np_images.append(np_image_grey)
                pil_images.append(PIL.Image.fromarray(np_image_grey))

    np_images = np.stack(np_images)

    if mnist:
        np_images = resize_and_pad_raster_img(np_images, w=28, h=28, pad_width=4)
        h_diff, w_diff = get_diff_to_center_of_mass(np_images[-1])
        np_images = np.asarray([shift(np_image, h_diff, w_diff) for np_image in np_images])
        pil_images = [PIL.Image.fromarray(np_image) for np_image in np_images]

    if gif_path is not None:
        with PIL.Image.new('L', (w, h)) as im:
            im.save(gif_path, save_all=True, append_images=pil_images, duration=sampling_length, loop=0)
            #print('Saved the GIF format of the raster image sequence in', gif_path)

    if png_path is not None:
        pil_images[-1].convert('L').save(png_path)

    return np_images

def rasterize_dev(df_dots, w, h, sampling_length, stroke_width, gif_path, png_path, eod_stop=False):
    # (원본 로직 생략 없이 그대로 사용)
    w = int(w); h = int(h)
    color_background = Color('black')
    color_stroke = Color('white')
    color_fill = Color('none')

    np_images, pil_images = list(), list()
    np_image_grey = np.zeros((w, h))
    pil_image = PIL.Image.fromarray(np_image_grey).convert('RGBA')
    pil_images.append(pil_image)
    np_images.append(np.asarray(pil_image))

    for j in range(1, len(df_dots)):
        rows = df_dots.loc[:j]
        with Drawing() as draw:
            draw.stroke_width = stroke_width
            draw.stroke_color = color_stroke
            draw.fill_color = color_fill
            draw.stroke_line_join = 'round'

            draw.path_start()
            prev_x, prev_y, prev_hov, _ = rows.loc[0]
            prev_x, prev_y = int(prev_x), int(prev_y)
            draw.path_move(to=(prev_x, prev_y))
            
            for i in range(1, len(rows)):
                x, y, hov, eod = rows.loc[i]
                x, y = int(x), int(y)
                if (prev_hov, hov) == (1,0): draw.path_move(to=(x, y))
                elif hov == 1: pass
                else: draw.path_line(to=(x, y))
                prev_x, prev_y, prev_hov = x, y, hov
                if eod == 1 and eod_stop: break

            draw.path_move(to=(prev_x, prev_y))
            draw.path_finish()

            point_width = stroke_width / 5
            draw.stroke_width = point_width / 4
            draw.stroke_color = Color('red')
            draw.fill_color = Color('red')

            for i in range(1, len(rows)):
                x, y, hov, eod = rows.loc[i]
                x, y = int(x), int(y)
                if hov == 1:
                    draw.stroke_width = point_width / 4 
                    draw.stroke_color = Color('cyan')
                    draw.fill_color = Color('cyan')
                    draw.circle(origin=(x, y), perimeter=(x+point_width/4,y+point_width/4))
                else:
                    draw.stroke_width = point_width / 4
                    draw.stroke_color = Color('red')
                    draw.fill_color = Color('red')
                draw.circle(origin=(x, y), perimeter=(x+point_width/4,y+point_width/4))
                if eod == 1 and eod_stop: break

            with Image(width=w, height=h, background=color_background) as image:
                draw(image)
                np_image_RGBA = np.array(image)
                np_images.append(np_image_RGBA)
                pil_images.append(PIL.Image.fromarray(np_image_RGBA))

        if eod == 1 and eod_stop: break

    with PIL.Image.new('RGBA', (w, h)) as im:
        im.save(gif_path, save_all=True, append_images=pil_images, duration=sampling_length, loop=0)
        #print('Saved the GIF format of the raster image sequence in', gif_path)

    pil_images[-1].convert('RGBA').save(png_path)
    return np.stack(np_images)

def resize_and_pad_raster_img(d_img, h, w, pad_width):
    d_img_cropped = crop_square(d_img)
    d_img_cropped_resized = resize_d_img(d_img_cropped, h-2*pad_width, w-2*pad_width)
    return np.pad(d_img_cropped_resized, ((0,0),(pad_width, pad_width),(pad_width, pad_width)), mode='constant')

def resize_d_img(d_img, h, w):
    time_steps = d_img.shape[0]
    d_img_new = np.zeros((time_steps, h, w))
    for t in range(time_steps):
        d_img_new[t,:,:] = cv2.resize(d_img[t,:,:], (h, w), interpolation=cv2.INTER_AREA)
    return d_img_new

def get_diff_to_center_of_mass(img):
    h_cm = ((img.sum(axis=1) / img.sum() * np.arange(img.shape[0]))).sum().round().astype(int)
    w_cm = ((img.sum(axis=0) / img.sum() * np.arange(img.shape[1]))).sum().round().astype(int)
    h_diff = int(round((img.shape[0] - 1) / 2 - h_cm))
    w_diff = int(round((img.shape[1] - 1) / 2 - w_cm))
    return h_diff, w_diff

def crop_square(d_img):
    time_steps, h_img, w_img = d_img.shape
    pl, pr, pt, pb = get_pad_params(d_img)
    d_img_cropped = d_img[:,pt:h_img-pb,pl:w_img-pr]

    h_img_no_pad, w_img_no_pad = h_img - pt - pb, w_img - pl - pr

    if h_img_no_pad > w_img_no_pad:
        d_img_new = np.zeros((time_steps, h_img_no_pad, h_img_no_pad))
        pad = h_img_no_pad - w_img_no_pad
        pad_L = pad // 2
        pad_R = pad_L + (pad % 2)
        for t in range(time_steps):
            d_img_new[t,:,:] = np.pad(d_img_cropped[t,:,:], ((0,0),(pad_L,pad_R)), mode='constant')
    else:
        d_img_new = np.zeros((time_steps, w_img_no_pad, w_img_no_pad))
        pad = w_img_no_pad - h_img_no_pad
        pad_T = pad // 2
        pad_B = pad_T + (pad % 2)
        for t in range(time_steps):
            d_img_new[t,:,:] = np.pad(d_img_cropped[t,:,:], ((pad_T,pad_B),(0,0)), mode='constant')
    return d_img_new

def get_pad_params(d_img):
    img_last = d_img[-1,:,:]
    sum_along_h = img_last.sum(axis=0) 
    sum_along_w = img_last.sum(axis=1) 
    return get_pad_len(sum_along_h), get_pad_len(sum_along_h, True), get_pad_len(sum_along_w), get_pad_len(sum_along_w, True)

def get_pad_len(array, from_back=False):
    pad_len = 0
    if from_back: array = np.flip(array)
    for val in array:
        if val != 0: break
        pad_len += 1
    return pad_len

def resize_vector_img(df_vector, h_target=28, w_target=28, pad=4):
    x, y = df_vector.x, df_vector.y
    bound_L, bound_R, bound_T, bound_B = x.min(), x.max(), y.min(), y.max()
    crop_w, crop_h = (bound_R - bound_L + 1), (bound_B - bound_T + 1)
    
    x_new, y_new = x - bound_L, y - bound_T

    if crop_w > crop_h:
        edge_len = crop_w
        y_new += (crop_w - crop_h) // 2
    else:
        edge_len = crop_h
        x_new += (crop_h - crop_w) // 2

    w_ratio = (w_target - 2*pad - 1) / (edge_len - 1)
    h_ratio = (h_target - 2*pad - 1) / (edge_len - 1)
    
    x_new = (x_new * w_ratio).round().astype(int) + pad
    y_new = (y_new * h_ratio).round().astype(int) + pad

    df_vector.x, df_vector.y = x_new, y_new

def shift(X, dy, dx):
    X = np.roll(np.roll(X, dy, axis=0), dx, axis=1)
    if dy>0: X[:dy, :] = 0
    elif dy<0: X[dy:, :] = 0
    if dx>0: X[:, :dx] = 0
    elif dx<0: X[:, dx:] = 0
    return X

def rotate_batch(degree, x_batch, y_batch):
    if degree == 0: return x_batch, y_batch
    angle = np.deg2rad(degree)
    angle_random = np.random.uniform(-angle, angle)
    x_batch_rot, y_batch_rot = deepcopy(x_batch), deepcopy(y_batch)
    
    # tf.math.cos/sin 대신 np.cos/np.sin 사용
    cos_val = np.cos(angle_random)
    sin_val = np.sin(angle_random)
    
    for i in range(len(x_batch)):
        x_batch_rot[i][:,0] = x_batch[i][:,0] * cos_val - x_batch[i][:,1] * sin_val
        y_batch_rot[i][:,0] = y_batch[i][:,0] * cos_val - y_batch[i][:,1] * sin_val
        x_batch_rot[i][:,1] = x_batch[i][:,0] * sin_val + x_batch[i][:,1] * cos_val
        y_batch_rot[i][:,1] = y_batch[i][:,0] * sin_val + y_batch[i][:,1] * cos_val

    return x_batch_rot, y_batch_rot

def mask_dxdy_batch(x_batch, p_mask_dxdy):
    x_batch_masked = deepcopy(x_batch)
    for i in range(len(x_batch)):
        length = x_batch_masked[i].shape[0]
        indexes_masked = np.where(np.random.binomial(1, p_mask_dxdy, length) == 1)[0]
        x_batch_masked[i][indexes_masked,0] = 0
        x_batch_masked[i][indexes_masked,1] = 0
    return x_batch_masked