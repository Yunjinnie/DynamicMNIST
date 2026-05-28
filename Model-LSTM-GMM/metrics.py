import tensorflow as tf
import numpy as np
from utils import parse_pred, get_df_v_from_y_target, sample_output_seq, get_seq_length
from smallestenclosingcircle import make_circle


def get_target_gmm_mean_distance(y_trgt, y_pred, seq_lengths, d_mean, d_norm, s_mean, s_norm, hparams):
    seq_mask = tf.sequence_mask(seq_lengths, maxlen=tf.shape(y_pred)[-2])

    d_trgt = y_trgt[...,0]
    s_trgt = y_trgt[...,1]
    pi, mu1, mu2 = parse_pred(y_pred, hparams)[0:3]

    # Get the weighted sum of each set of mu1 and mu2 elements.
    mu1_weighted_sum = tf.reduce_sum(tf.multiply(pi, mu1), axis=-1)
    mu2_weighted_sum = tf.reduce_sum(tf.multiply(pi, mu2), axis=-1)

    # Get the absolute error.
    d_ae = tf.abs(d_trgt-mu1_weighted_sum)
    s_ae = tf.abs(s_trgt-mu2_weighted_sum)

    # Mask the padding values to be 0.
    seq_mask = tf.cast(seq_mask, y_pred.dtype)
    d_ae_masked = d_ae * seq_mask
    s_ae_masked = s_ae * seq_mask

    # Average the masked sequence along the sequence axis.
    d_mae = tf.reduce_sum(d_ae_masked, axis=-1) / seq_lengths
    s_mae = tf.reduce_sum(s_ae_masked, axis=-1) / seq_lengths

    # Average the sequence mean in the batch set.
    d_mae_target_gmm_mean = tf.reduce_mean(d_mae) * d_norm
    s_mae_target_gmm_mean = tf.reduce_mean(s_mae) * s_norm

    d_mae_target_gmm_mean = tf.get_static_value(d_mae_target_gmm_mean)
    s_mae_target_gmm_mean = tf.get_static_value(s_mae_target_gmm_mean)

    return d_mae_target_gmm_mean, s_mae_target_gmm_mean

def get_gmm_mean_std(y_pred, seq_lengths, dx_norm, dy_norm, hparams):
    seq_mask = tf.sequence_mask(seq_lengths, maxlen=tf.shape(y_pred)[-2])
    seq_mask = tf.cast(seq_mask, y_pred.dtype)

    pi, _, _, dx_std, dy_std, corr, _, _ = parse_pred(y_pred, hparams)

    dx_std = dx_std * dx_norm
    dy_std = dy_std * dy_norm
    det = tf.pow(tf.square(dx_std*dy_std)*(1-tf.square(corr)), 1/4) # determinent = generalized variance

    # Get the weighted sum of each set of elements.
    dx_std_weighted_sum = tf.reduce_sum(tf.multiply(pi, dx_std), axis=-1)
    dy_std_weighted_sum = tf.reduce_sum(tf.multiply(pi, dy_std), axis=-1)
    corr_weighted_sum = tf.reduce_sum(tf.multiply(pi, corr), axis=-1)
    det_weighted_sum = tf.reduce_sum(tf.multiply(pi, det), axis=-1)

    # Mask the padding values to be 0.
    dx_std_masked = dx_std_weighted_sum * seq_mask
    dy_std_masked = dy_std_weighted_sum * seq_mask
    corr_masked = corr_weighted_sum * seq_mask
    det_masked = det_weighted_sum * seq_mask

    # Average the masked sequence along the sequence axis.
    dx_std_mean = tf.reduce_sum(dx_std_masked, axis=-1) / seq_lengths
    dy_std_mean = tf.reduce_sum(dy_std_masked, axis=-1) / seq_lengths
    corr_mean = tf.reduce_sum(corr_masked, axis=-1) / seq_lengths
    corr_abs_mean = tf.reduce_sum(tf.math.abs(corr_masked), axis=-1) / seq_lengths
    det_mean = tf.reduce_sum(det_masked, axis=-1) / seq_lengths

    # Average the sequence mean in the batch set.
    dx_std_mean = tf.reduce_mean(dx_std_mean)
    dy_std_mean = tf.reduce_mean(dy_std_mean)
    corr_mean = tf.reduce_mean(corr_mean)
    corr_abs_mean = tf.reduce_mean(corr_abs_mean)
    det_mean = tf.reduce_mean(det_mean)

    return dx_std_mean, dy_std_mean, corr_mean, corr_abs_mean, det_mean


def get_dxdy_distance(df_v_trgt, df_v):
    dx_mae = np.mean(np.absolute(df_v_trgt.dx.to_numpy() - df_v.dx.to_numpy()))
    dy_mae = np.mean(np.absolute(df_v_trgt.dy.to_numpy() - df_v.dy.to_numpy()))

    return dx_mae, dy_mae

def get_sampled_dxdy_distance(y_trgt, y_pred, seq_lengths, dx_norm, dy_norm, d2x_norm, d2y_norm, hparams):
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
    seq_mask = tf.sequence_mask(seq_lengths, maxlen=y_pred.shape[-2])

    hov_trgt = y_trgt[...,2]
    hov_pred = tf.squeeze(parse_pred(y_pred, hparams)[-2], axis=-1)
    hov_pred_labels = tf.cast(hov_pred > 0.5, dtype=hov_pred.dtype)

    correct_set = tf.math.logical_and(hov_trgt == hov_pred_labels, seq_mask)
    correct_counts = tf.math.count_nonzero(correct_set, axis=-1)
    accuracy_set = correct_counts / seq_lengths

    correct_recall_set = tf.math.logical_and(correct_set, tf.cast(hov_trgt, tf.bool))
    correct_recall_counts = tf.math.count_nonzero(correct_recall_set, dtype=hov_trgt.dtype, axis=-1)
    recall_set = correct_recall_counts / tf.reduce_sum(hov_trgt * tf.cast(seq_mask, hov_trgt.dtype), axis=-1)

    correct_recall_set = tf.math.logical_and(correct_set, tf.cast(hov_trgt, tf.bool))
    correct_recall_counts = tf.math.count_nonzero(correct_recall_set, dtype=hov_pred.dtype, axis=-1)
    recall_set = correct_recall_counts / tf.reduce_sum(hov_trgt * tf.cast(seq_mask, hov_pred.dtype), axis=-1)
    n_none_nan_rc = tf.math.count_nonzero(tf.math.logical_not(tf.math.is_nan(recall_set)), dtype=recall_set.dtype)
    recall_set = tf.where(tf.math.is_nan(recall_set), tf.zeros_like(recall_set), recall_set)

    correct_precision_set = tf.math.logical_and(correct_set, tf.cast(hov_pred_labels, tf.bool))
    correct_precision_counts = tf.math.count_nonzero(correct_precision_set, dtype=hov_pred.dtype, axis=-1)
    precision_set = correct_precision_counts / tf.reduce_sum(hov_pred_labels * tf.cast(seq_mask, hov_pred.dtype), axis=-1)
    n_none_nan_pc = tf.math.count_nonzero(tf.math.logical_not(tf.math.is_nan(precision_set)), dtype=precision_set.dtype)
    precision_set = tf.where(tf.math.is_nan(precision_set), tf.zeros_like(precision_set), precision_set)

    accuracy = tf.reduce_mean(accuracy_set)
    if tf.get_static_value(n_none_nan_rc) != 0:
        recall = tf.reduce_sum(recall_set) / n_none_nan_rc
    else:
        recall = tf.zeros_like(accuracy)
    if tf.get_static_value(n_none_nan_pc) != 0:
        precision = tf.reduce_sum(precision_set) / n_none_nan_pc
    else:
        precision = tf.zeros_like(accuracy)

    accuracy = tf.get_static_value(accuracy)
    recall = tf.get_static_value(recall)
    precision = tf.get_static_value(precision)

    return accuracy, recall, precision

def get_eod_accuracy(y_trgt, y_pred, seq_lengths, hparams):
    seq_mask = tf.sequence_mask(seq_lengths, maxlen=y_pred.shape[-2])

    eod_trgt = y_trgt[...,3]
    eod_pred = tf.squeeze(parse_pred(y_pred, hparams)[-1], axis=-1)
    eod_pred_labels = tf.cast(eod_pred > 0.5, dtype=eod_pred.dtype)

    correct_set = tf.math.logical_and(eod_trgt == eod_pred_labels, seq_mask)
    correct_counts = tf.math.count_nonzero(correct_set, axis=-1)
    accuracy_set = correct_counts / seq_lengths

    correct_recall_set = tf.math.logical_and(correct_set, tf.cast(eod_trgt, tf.bool))
    correct_recall_counts = tf.math.count_nonzero(correct_recall_set, dtype=eod_trgt.dtype, axis=-1)
    recall_set = correct_recall_counts / tf.reduce_sum(eod_trgt * tf.cast(seq_mask, eod_trgt.dtype), axis=-1)

    correct_precision_set = tf.math.logical_and(correct_set, tf.cast(eod_pred_labels, tf.bool))
    correct_precision_counts = tf.math.count_nonzero(correct_precision_set, dtype=eod_pred_labels.dtype, axis=-1)
    precision_set = correct_precision_counts / tf.reduce_sum(eod_pred_labels * tf.cast(seq_mask, eod_pred_labels.dtype), axis=-1)
    n_none_nan = tf.math.count_nonzero(tf.math.logical_not(tf.math.is_nan(precision_set)), dtype=precision_set.dtype)
    precision_set = tf.where(tf.math.is_nan(precision_set), tf.zeros_like(precision_set), precision_set)

    accuracy = tf.reduce_mean(accuracy_set)
    recall = tf.reduce_mean(recall_set)
    precision = tf.reduce_sum(precision_set) / n_none_nan

    accuracy = tf.get_static_value(accuracy)
    recall = tf.get_static_value(recall)
    precision = tf.get_static_value(precision)

    return accuracy, recall, precision

def get_eod_distance(y_trgt, y_pred, seq_lengths, hparams):
    seq_mask = tf.sequence_mask(seq_lengths, maxlen=y_pred.shape[-2])

    eod_trgt = y_trgt[...,3]
    eod_pred = tf.squeeze(parse_pred(y_pred, hparams)[-1], axis=-1)
    eod_pred_labels = tf.cast(eod_pred > 0.5, dtype=eod_pred.dtype)
    eod_pred_labels = eod_pred_labels.numpy()
    eod_indexes = seq_lengths - 1

    eod_pred_indexes = np.zeros_like(eod_indexes)

    for i in range(eod_pred_labels.shape[0]):
        seq = eod_pred_labels[i,...]
        locations = np.where(seq == 1)[0]
        first_location = 0
        if len(locations) != 0:
            first_location = locations[0]
        eod_pred_indexes[i] = first_location

    mae_eod = np.mean(np.absolute(eod_indexes - eod_pred_indexes))

    return mae_eod

def compute_stat_distance(p_samples, q_samples, bins, range):

    def get_historgram(samples, bins, range):
        h = np.histogram(samples, bins=bins, range=range)[0]
        # np.sum(h[0]) == 0 occurs when samples are out of the given range.
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
    q3, q1 = np.percentile(data, [75, 25])
    IQR = q3 - q1

    upper_bound = q3 + 1.5 * IQR
    lower_bound = q1 - 1.5 * IQR

    data = data[data <= upper_bound]
    data = data[data >= lower_bound]

    return data
