import torch
import numpy as np
from utils import parse_pred, get_df_v_from_y_target, sample_output_seq, get_seq_length
from smallestenclosingcircle import make_circle


def get_target_gmm_mean_distance(y_trgt, y_pred, seq_lengths, d_mean, d_norm, s_mean, s_norm, hparams):
    # tf.sequence_mask 대체: PyTorch 브로드캐스팅 활용
    maxlen = y_pred.shape[-2]

    # y_trgt, y_pred Numpy 배열이면 PyTorch 텐서로 변환
    if isinstance(y_trgt, np.ndarray):
        y_trgt = torch.tensor(y_trgt, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    target_device = y_pred.device if hasattr(y_pred, 'device') else torch.device('cpu')
    #print('dist device', target_device) # y_pred numpy.ndarray면 cuda:0, numpy.int64면 cpu

    #print(type(seq_lengths)) # .dtype == numpy.ndarray
    '''
    <class 'numpy.ndarray'>로 나오다가 마지막에 
    <class 'numpy.int64'>
    '''
    # if isinstance(seq_lengths, np.ndarray) or isinstance(seq_lengths, list):
    #     seq_lengths = torch.tensor(seq_lengths, device=target_device)
    # elif isinstance(seq_lengths, torch.Tensor):
    #     seq_lengths = seq_lengths.to(target_device)

    seq_lengths = torch.as_tensor(seq_lengths, device=target_device)
    
    # 텐서의 차원이 0 == 단일 숫자(scalar) -> unsqueeze(0)을 써서 [150] 같은 1차원 배열로 감싸줄 것
    if seq_lengths.dim() == 0:
        seq_lengths = seq_lengths.unsqueeze(0)
        
    seq_mask = torch.arange(maxlen, device=y_pred.device)[None, :] < seq_lengths[:, None] ## error
    
    d_trgt = y_trgt[..., 0].to(target_device)
    s_trgt = y_trgt[..., 1].to(target_device)
    pi, mu1, mu2 = parse_pred(y_pred, hparams)[0:3]

    # Get the weighted sum of each set of mu1 and mu2 elements.
    #device = d_trgt.device
    # print(device)  => cpu
    #### 
    mu1_weighted_sum = (torch.sum(pi * mu1, dim=-1)).to(target_device)
    mu2_weighted_sum = (torch.sum(pi * mu2, dim=-1)).to(target_device)

    # Get the absolute error.
    
    d_ae = torch.abs(d_trgt - mu1_weighted_sum)
    s_ae = torch.abs(s_trgt - mu2_weighted_sum)

    # Mask the padding values to be 0.
    seq_mask = (seq_mask.to(y_pred.dtype)).to(target_device)
    d_ae_masked = d_ae * seq_mask
    s_ae_masked = s_ae * seq_mask

    # Average the masked sequence along the sequence axis.
    seq_lengths_f = (seq_lengths.to(y_pred.dtype)).to(target_device)
    d_mae = torch.sum(d_ae_masked, dim=-1) / seq_lengths_f
    s_mae = torch.sum(s_ae_masked, dim=-1) / seq_lengths_f

    # Average the sequence mean in the batch set and extract scalar value (.item())
    d_mae_target_gmm_mean = (torch.mean(d_mae) * d_norm).item()
    s_mae_target_gmm_mean = (torch.mean(s_mae) * s_norm).item()

    return d_mae_target_gmm_mean, s_mae_target_gmm_mean

def get_gmm_mean_std(y_pred, seq_lengths, dx_norm, dy_norm, hparams):
    maxlen = y_pred.shape[-2]
    #print(type(y_pred)) # .dtypes() => numpy type에서만
    '''
    <class 'torch.Tensor'>
    '''
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    target_device = y_pred.device if hasattr(y_pred, 'device') else torch.device('cpu')
    #print('std device', target_device)

    # if isinstance(seq_lengths, np.ndarray) or isinstance(seq_lengths, list):
    #     seq_lengths = torch.tensor(seq_lengths, device=target_device)
    # elif isinstance(seq_lengths, torch.Tensor):
    #     seq_lengths = seq_lengths.to(target_device)
    seq_lengths = torch.as_tensor(seq_lengths, device=target_device)

    if seq_lengths.dim() == 0:
        seq_lengths = seq_lengths.unsqueeze(0)

    seq_mask = (torch.arange(maxlen, device=target_device)[None, :] < seq_lengths[:, None])#.to(torch.float32) #y_pred.dtype)

    pi, _, _, dx_std, dy_std, corr, _, _ = parse_pred(y_pred, hparams)

    dx_std = dx_std * dx_norm
    dy_std = dy_std * dy_norm
    det = torch.pow(torch.square(dx_std * dy_std) * (1 - torch.square(corr)), 0.25) # determinent = generalized variance

    # Get the weighted sum of each set of elements.
    dx_std_weighted_sum = torch.sum(pi * dx_std, dim=-1)
    dy_std_weighted_sum = torch.sum(pi * dy_std, dim=-1)
    corr_weighted_sum = torch.sum(pi * corr, dim=-1)
    det_weighted_sum = torch.sum(pi * det, dim=-1)

    # Mask the padding values to be 0.
    dx_std_masked = dx_std_weighted_sum * seq_mask
    dy_std_masked = dy_std_weighted_sum * seq_mask
    corr_masked = corr_weighted_sum * seq_mask
    det_masked = det_weighted_sum * seq_mask

    # Average the masked sequence along the sequence axis.
    seq_lengths_f = seq_lengths.to(y_pred.dtype)
    dx_std_mean = torch.mean(torch.sum(dx_std_masked, dim=-1) / seq_lengths_f).item()
    dy_std_mean = torch.mean(torch.sum(dy_std_masked, dim=-1) / seq_lengths_f).item()
    corr_mean = torch.mean(torch.sum(corr_masked, dim=-1) / seq_lengths_f).item()
    corr_abs_mean = torch.mean(torch.sum(torch.abs(corr_masked), dim=-1) / seq_lengths_f).item()
    det_mean = torch.mean(torch.sum(det_masked, dim=-1) / seq_lengths_f).item()

    return dx_std_mean, dy_std_mean, corr_mean, corr_abs_mean, det_mean


def get_dxdy_distance(df_v_trgt, df_v):
    dx_mae = np.mean(np.absolute(df_v_trgt.dx.to_numpy() - df_v.dx.to_numpy()))
    dy_mae = np.mean(np.absolute(df_v_trgt.dy.to_numpy() - df_v.dy.to_numpy()))

    return dx_mae, dy_mae

def get_sampled_dxdy_distance(y_trgt, y_pred, seq_lengths, dx_norm, dy_norm, d2x_norm, d2y_norm, hparams):
    # PyTorch 텐서일 수 있으므로 numpy로 변환
    if isinstance(y_trgt, torch.Tensor): y_trgt = y_trgt.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    if isinstance(seq_lengths, torch.Tensor): seq_lengths = seq_lengths.detach().cpu().numpy()

    batch_size_eval = hparams['batch_size_eval']
    batch_size = y_trgt.shape[0]

    indexes = np.arange(batch_size)
    if batch_size_eval < batch_size:
        np.random.shuffle(indexes)
        indexes = indexes[:batch_size_eval]

    mae_dx = np.zeros(len(indexes))
    mae_dy = np.zeros(len(indexes))
    for i, i_sample in enumerate(indexes):
        y_trgt_sample = y_trgt[i_sample]
        y_pred_sample = y_pred[i_sample]
        seq_length_sample = seq_lengths[i_sample]

        # Parse outputs from the y_target
        df_v_trgt = get_df_v_from_y_target(y_trgt_sample, seq_length_sample, 0, dx_norm, 0, dy_norm)

        # Sample a GMM output from the predicted GMM parameters.
        df_v = sample_output_seq(y_pred_sample, seq_length_sample, dx_norm, dy_norm, d2x_norm, d2y_norm, hparams, use_eod=False)

        # Collect MAE of dx and dy.
        mae_dx[i] = np.mean(np.absolute(df_v_trgt.dx.to_numpy() - df_v.dx.to_numpy()))
        mae_dy[i] = np.mean(np.absolute(df_v_trgt.dy.to_numpy() - df_v.dy.to_numpy()))

    # Average mae_dx along batch samples.
    mean_mae_dx = mae_dx.mean()
    mean_mae_dy = mae_dy.mean()

    return mean_mae_dx, mean_mae_dy


def get_hov_accuracy(y_trgt, y_pred, seq_lengths, hparams):
    maxlen = y_pred.shape[-2]
    target_device = y_pred.device if hasattr(y_pred, 'device') else torch.device('cpu')
    if isinstance(seq_lengths, np.ndarray) or isinstance(seq_lengths, list):
        seq_lengths = torch.tensor(seq_lengths, device=target_device)
    elif isinstance(seq_lengths, torch.Tensor):
        seq_lengths = seq_lengths.to(target_device)

    seq_mask = torch.arange(maxlen, device=y_pred.device)[None, :] < seq_lengths[:, None]

    hov_trgt = y_trgt[..., 2].to(target_device)
    hov_pred = torch.squeeze(parse_pred(y_pred, hparams)[-2], dim=-1)
    hov_pred_labels = (hov_pred > 0.5).to(hov_pred.dtype)

    correct_set = (hov_trgt == hov_pred_labels) & seq_mask
    correct_counts = torch.sum(correct_set.to(torch.int), dim=-1)
    accuracy_set = correct_counts.to(y_pred.dtype) / seq_lengths.to(y_pred.dtype)

    # Recall
    correct_recall_set = correct_set & hov_trgt.to(torch.bool)
    correct_recall_counts = torch.sum(correct_recall_set.to(hov_trgt.dtype), dim=-1)
    recall_denom = torch.sum(hov_trgt * seq_mask.to(hov_trgt.dtype), dim=-1)
    recall_set = correct_recall_counts / recall_denom
    
    n_none_nan_rc = torch.sum(~torch.isnan(recall_set))
    recall_set = torch.nan_to_num(recall_set, nan=0.0)

    # Precision
    correct_precision_set = correct_set & hov_pred_labels.to(torch.bool)
    correct_precision_counts = torch.sum(correct_precision_set.to(hov_pred.dtype), dim=-1)
    precision_denom = torch.sum(hov_pred_labels * seq_mask.to(hov_pred.dtype), dim=-1)
    precision_set = correct_precision_counts / precision_denom
    
    n_none_nan_pc = torch.sum(~torch.isnan(precision_set))
    precision_set = torch.nan_to_num(precision_set, nan=0.0)

    accuracy = torch.mean(accuracy_set)
    if n_none_nan_rc.item() != 0:
        recall = torch.sum(recall_set) / n_none_nan_rc
    else:
        recall = torch.zeros_like(accuracy)
        
    if n_none_nan_pc.item() != 0:
        precision = torch.sum(precision_set) / n_none_nan_pc
    else:
        precision = torch.zeros_like(accuracy)

    return accuracy.item(), recall.item(), precision.item()

def get_eod_accuracy(y_trgt, y_pred, seq_lengths, hparams):
    maxlen = y_pred.shape[-2]
    target_device = y_pred.device if hasattr(y_pred, 'device') else torch.device('cpu')
    if isinstance(seq_lengths, np.ndarray) or isinstance(seq_lengths, list):
        seq_lengths = torch.tensor(seq_lengths, device=target_device)
    elif isinstance(seq_lengths, torch.Tensor):
        seq_lengths = seq_lengths.to(target_device)

    seq_mask = torch.arange(maxlen, device=y_pred.device)[None, :] < seq_lengths[:, None]

    eod_trgt = y_trgt[..., 3].to(target_device)
    eod_pred = torch.squeeze(parse_pred(y_pred, hparams)[-1], dim=-1)
    eod_pred_labels = (eod_pred > 0.5).to(eod_pred.dtype)

    correct_set = (eod_trgt == eod_pred_labels) & seq_mask
    correct_counts = torch.sum(correct_set.to(torch.int), dim=-1)
    accuracy_set = correct_counts.to(y_pred.dtype) / seq_lengths.to(y_pred.dtype)

    # Recall
    correct_recall_set = correct_set & eod_trgt.to(torch.bool)
    correct_recall_counts = torch.sum(correct_recall_set.to(eod_trgt.dtype), dim=-1)
    recall_denom = torch.sum(eod_trgt * seq_mask.to(eod_trgt.dtype), dim=-1)
    recall_set = correct_recall_counts / recall_denom

    # Precision
    correct_precision_set = correct_set & eod_pred_labels.to(torch.bool)
    correct_precision_counts = torch.sum(correct_precision_set.to(eod_pred_labels.dtype), dim=-1)
    precision_denom = torch.sum(eod_pred_labels * seq_mask.to(eod_pred_labels.dtype), dim=-1)
    precision_set = correct_precision_counts / precision_denom
    
    n_none_nan_pc = torch.sum(~torch.isnan(precision_set))
    precision_set = torch.nan_to_num(precision_set, nan=0.0)

    accuracy = torch.mean(accuracy_set)
    recall = torch.mean(torch.nan_to_num(recall_set, nan=0.0))
    
    if n_none_nan_pc.item() != 0:
        precision = torch.sum(precision_set) / n_none_nan_pc
    else:
        precision = torch.zeros_like(accuracy)

    return accuracy.item(), recall.item(), precision.item()

def get_eod_distance(y_trgt, y_pred, seq_lengths, hparams):
    # PyTorch 텐서를 NumPy로 변환
    eod_pred = torch.squeeze(parse_pred(y_pred, hparams)[-1], dim=-1)
    eod_pred_labels = (eod_pred > 0.5).to(eod_pred.dtype)
    eod_pred_labels = eod_pred_labels.detach().cpu().numpy()
    
    if isinstance(seq_lengths, torch.Tensor):
        seq_lengths = seq_lengths.detach().cpu().numpy()
        
    eod_indexes = seq_lengths - 1
    eod_pred_indexes = np.zeros_like(eod_indexes)

    for i in range(eod_pred_labels.shape[0]):
        seq = eod_pred_labels[i, ...]
        locations = np.where(seq == 1)[0]
        first_location = 0
        if len(locations) != 0:
            first_location = locations[0]
        eod_pred_indexes[i] = first_location

    mae_eod = np.mean(np.absolute(eod_indexes - eod_pred_indexes))

    return mae_eod

# 아래의 통계 및 유틸리티 함수들은 순수 NumPy 기반이므로 원본 유지

def compute_stat_distance(p_samples, q_samples, bins, range):

    def get_historgram(samples, bins, range):
        h = np.histogram(samples, bins=bins, range=range)[0]
        if np.sum(h) != 0:
            h = h / np.sum(h)
        return h

    def total_variation_function(p, q):
        result = np.sum(np.absolute(p-q)) / 2
        result = np.clip(result, 0, 1)
        return result

    p = get_historgram(p_samples, bins, range)
    q = get_historgram(q_samples, bins, range)
    if (np.sum(p) == 0) or (np.sum(q) == 0):
        distance = 1
    else:
        distance = total_variation_function(p, q)

    return distance

def get_stat_data_from_test_set(character_set, test_set, dx_scale, dy_scale):
    inputs_test, targets_test, seq_lengths_test, characters_test = test_set

    stat_data_character = dict()
    for character in character_set:
        indexes_character = np.squeeze(np.argwhere(characters_test == character))
        dx_means = np.zeros(len(indexes_character))
        dy_means = np.zeros(len(indexes_character))
        dspeed_means = np.zeros(len(indexes_character))
        speed_inversions = np.zeros(len(indexes_character))
        speed_inversions_normalized = np.zeros(len(indexes_character))
        deg_inversions = np.zeros(len(indexes_character))
        deg_inversions_normalized = np.zeros(len(indexes_character))
        mv_lengths = np.zeros(len(indexes_character))
        x_lengths = np.zeros(len(indexes_character))
        y_lengths = np.zeros(len(indexes_character))
        diameters = np.zeros(len(indexes_character))
        for i, target in enumerate(targets_test[indexes_character]):
            dx = target[:,0] * dx_scale
            dy = target[:,1] * dy_scale
            hov = target[:,2]
            eod = target[:,3]

            dx_means[i] = np.mean(dx)
            dy_means[i] = np.mean(dy)

            dspeed = np.sqrt(dx**2 + dy**2)
            dspeed_means[i] = np.mean(dspeed)

            speed_inversions[i] = count_inversions(dspeed)

            deg_inversions[i] = count_deg_inversions(dx, dy)

            mv_lengths[i] = np.sum(dspeed)

            seq_length = target.shape[0]

            speed_inversions_normalized[i] = speed_inversions[i] / seq_length
            deg_inversions_normalized[i] = deg_inversions[i] / seq_length

            x_seq = np.zeros(seq_length+1)
            y_seq = np.zeros(seq_length+1)
            for j in range(seq_length):
                x_seq[j+1] = x_seq[j] + dx[j]
                y_seq[j+1] = y_seq[j] + dy[j]
            x_lengths[i] = np.absolute(x_seq.max() - x_seq.min())
            y_lengths[i] = np.absolute(y_seq.max() - y_seq.min())

            dots = [(x, y) for x, y in zip(x_seq, y_seq)]
            x_center, y_center, radius = make_circle(dots)
            diameters[i] = radius * 2

        stat_data_character[character] = {
            'speed_means':dspeed_means,
            'speed_inversions':speed_inversions,
            'speed_inversions_normalized':speed_inversions_normalized,
            'deg_inversions':deg_inversions,
            'deg_inversions_normalized':deg_inversions_normalized,
            'dx_means':dx_means,
            'dy_means':dy_means,
            'durations':seq_lengths_test[indexes_character].astype(np.int64),
            'lengths':mv_lengths,
            'widths':x_lengths,
            'heights':y_lengths,
            'diameters':diameters,
        }

    stat_data_all = dict()
    for key in stat_data_character[list(stat_data_character.keys())[0]].keys():
        stat_data_all[key] = list()
        for character in character_set:
            stat_data_all[key].append(stat_data_character[character][key])
        stat_data_all[key] = np.concatenate(stat_data_all[key])

    return stat_data_character, stat_data_all

def get_stat_data_from_outputs(outputs, dx_scale, dy_scale):
    n_samples = len(outputs)
    seq_lengths = np.asarray([get_seq_length(output[:,3]) for output in outputs])

    stat_data = dict()
    dx_means = np.zeros(n_samples)
    dy_means = np.zeros(n_samples)
    dspeed_means = np.zeros(n_samples)
    speed_inversions = np.zeros(n_samples)
    speed_inversions_normalized = np.zeros(n_samples)
    deg_inversions = np.zeros(n_samples)
    deg_inversions_normalized = np.zeros(n_samples)
    mv_lengths = np.zeros(n_samples)
    x_lengths = np.zeros(n_samples)
    y_lengths = np.zeros(n_samples)
    diameters = np.zeros(n_samples)
    for i, output in enumerate(outputs):
        dx = output[:seq_lengths[i],0] * dx_scale
        dy = output[:seq_lengths[i],1] * dy_scale
        dx_means[i] = np.mean(dx)
        dy_means[i] = np.mean(dy)

        dspeed = np.sqrt(dx**2 + dy**2)
        dspeed_means[i] = np.mean(dspeed)

        speed_inversions[i] = count_inversions(dspeed)

        deg_inversions[i] = count_deg_inversions(dx, dy)

        mv_lengths[i] = np.sum(dspeed)

        seq_length = seq_lengths[i]

        speed_inversions_normalized[i] = speed_inversions[i] / seq_length
        deg_inversions_normalized[i] = deg_inversions[i] / seq_length

        x_seq = np.zeros(seq_length+1)
        y_seq = np.zeros(seq_length+1)
        for j in range(seq_length-1):
            x_seq[j+1] = x_seq[j] + dx[j]
            y_seq[j+1] = y_seq[j] + dy[j]
        x_lengths[i] = np.absolute(x_seq.max() - x_seq.min())
        y_lengths[i] = np.absolute(y_seq.max() - y_seq.min())

        dots = [(x, y) for x, y in zip(x_seq, y_seq)]
        x_center, y_center, radius = make_circle(dots)
        diameters[i] = radius * 2

    stat_data = {
        'speed_means':dspeed_means,
        'speed_inversions':speed_inversions,
        'speed_inversions_normalized':speed_inversions_normalized,
        'deg_inversions':deg_inversions,
        'deg_inversions_normalized':deg_inversions_normalized,
        'dx_means':dx_means,
        'dy_means':dy_means,
        'durations':seq_lengths,
        'lengths':mv_lengths,
        'widths':x_lengths,
        'heights':y_lengths,
        'diameters':diameters,
    }

    return stat_data

def count_increment_inversions(increments):
    n_inversions = 0
    inversion_state_now = None

    for i, inversion_state_next in enumerate(increments):
        if inversion_state_next == 0:
            continue
        elif inversion_state_now is None:
            inversion_state_now = inversion_state_next
            continue

        if inversion_state_next * inversion_state_now < 0:
            inversion_state_now = inversion_state_next
            n_inversions += 1

    return n_inversions

def count_inversions(seq):
    increments = seq[1:] - seq[:-1] # future - present = increment
    increments = np.clip(increments, -1, 1) # 1: increasing, 0: same, -1: decreasing.

    n_inversions = count_increment_inversions(increments)

    return n_inversions

def count_deg_inversions(dx_seq, dy_seq):
    degrees = np.arctan2(dy_seq, dx_seq)
    deg_increments = degrees[1:] - degrees[:-1]

    deg_increments -= (deg_increments >  np.pi)*2*np.pi
    deg_increments += (deg_increments < -np.pi)*2*np.pi

    n_inversions = count_increment_inversions(deg_increments)

    return n_inversions

def rm_outliers(data):
    '''
    Remove outliers using an interquartile range.
    '''
    # [핵심 수정] 데이터가 비어 있으면 그대로 반환
    if len(data) == 0:
        return data
    
    q3, q1 = np.percentile(data, [75, 25])
    IQR = q3 - q1

    upper_bound = q3 + 1.5 * IQR
    lower_bound = q1 - 1.5 * IQR

    data = data[data <= upper_bound]
    data = data[data >= lower_bound]

    return data