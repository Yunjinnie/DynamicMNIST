import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from os.path import join, isfile, splitext
from os import listdir


def get_file_list(dir_path, ext='.csv'):
    '''
    ext should include the dot in front, e.g., '.csv'
    '''
    if ext is None:
        file_list = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    else:
        file_list = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and splitext(f)[1] == ext]
    return file_list

def get_batch_iterations(n_samples, batch_size):
    iterations = n_samples // batch_size
    if n_samples % batch_size != 0:
        iterations += 1
    return iterations

def get_cartesian_dot(angle, radius):
    '''Polar dot to cartesian dot'''
    x = radius * np.cos(np.deg2rad(angle))
    y = radius * np.sin(np.deg2rad(angle))
    return x, y

def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def get_pi_idx(x, pdf):
    N = pdf.shape[0]
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
    if (accumulate >= x):
        return i
    print('error with sampling ensemble')
    return -1

def sample_gmm(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod):
    '''This function does not work properly. The cause has not been found.'''
    idx = get_pi_idx(np.random.rand(), z_pi)
    next_dx, next_dy = sample_gaussian_2d(z_mu1[idx], z_mu2[idx], z_sigma1[idx], z_sigma2[idx], z_corr[idx])
    #next_dx, next_dy = sample_gaussian_2d(z_mu1[idx], z_mu2[idx], z_sigma1[idx], z_sigma2[idx], np.zeros_like(z_corr[idx])) # test
    #hov = 1 if 0.5 < np.squeeze(z_hov,0) else 0
    hov = 1 if np.random.rand() < z_hov else 0
    #eod = 1 if 0.5 < np.squeeze(z_eod,0) else 0
    eod = 1 if np.random.rand() < z_eod else 0

    return next_dx, next_dy, hov, eod

def sample_gmm_batch(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod):
    stack_dx = list(); stack_dy = list(); stack_hov = list(); stack_eod = list()
    batch_size = z_pi.shape[0]
    for i in range(batch_size):
        dx, dy, hov, eod = sample_gmm(z_pi[i], z_mu1[i], z_mu2[i], z_sigma1[i], z_sigma2[i], z_corr[i], z_hov[i], z_eod[i])
        stack_dx.append(dx); stack_dy.append(dy); stack_hov.append(hov); stack_eod.append(eod)
    dx = np.stack(stack_dx); dy = np.stack(stack_dy); hov = np.stack(stack_hov); eod = np.stack(stack_eod)

    return dx, dy, hov, eod

def sample_gmm_batch_tf(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov=None, z_eod=None, numpy_output=True, only_gmm=False):
    z_mu = tf.stack([z_mu1, z_mu2], axis=-1)
    cov_sig_1_2 = z_corr*z_sigma1*z_sigma2
    cov_m = tf.stack([tf.stack([tf.square(z_sigma1), cov_sig_1_2], axis=-1), tf.stack([cov_sig_1_2, tf.square(z_sigma2)], axis=-1)], axis=-1)
    mvgmm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=z_pi),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=z_mu, scale_tril=tf.linalg.cholesky(cov_m), validate_args=False, allow_nan_stats=True)
    )
    dxdy = tf.squeeze(mvgmm.sample(1),0) # shape == [batch_size, 2]. 2 of (dx,dy).
    dx = dxdy[...,0]
    dy = dxdy[...,1]

    if only_gmm:
        if numpy_output:
            return dx.numpy(), dy.numpy()
        else:
            return dx, dy

    #hov = tf.cast(tf.squeeze(z_hov,-1) > 0.5, tf.int64).numpy()
    #eod = tf.cast(tf.squeeze(z_eod,-1) > 0.5, tf.int64).numpy()
    hov = tfp.distributions.Bernoulli(probs=tf.squeeze(z_hov,-1), dtype=tf.int32).sample()
    eod = tfp.distributions.Bernoulli(probs=tf.squeeze(z_eod,-1), dtype=tf.int32).sample()

    if numpy_output:
        return dx.numpy(), dy.numpy(), hov.numpy(), eod.numpy()
    else:
        return dx, dy, hov, eod

def sample_gmm_seqs_tf(pred, hparams):
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod = parse_pred(pred, hparams)

    batch_size = tf.shape(z_pi)[0]
    max_seq_length = tf.shape(z_pi)[1]

    z_pi = tf.reshape(z_pi, [-1, tf.shape(z_pi)[2]])
    z_mu1 = tf.reshape(z_mu1, [-1, tf.shape(z_mu1)[2]])
    z_mu2 = tf.reshape(z_mu2, [-1, tf.shape(z_mu2)[2]])
    z_sigma1 = tf.reshape(z_sigma1, [-1, tf.shape(z_sigma1)[2]])
    z_sigma2 = tf.reshape(z_sigma2, [-1, tf.shape(z_sigma2)[2]])
    z_corr = tf.reshape(z_corr, [-1, tf.shape(z_corr)[2]])

    var_x, var_y, hov, eod = sample_gmm_batch_tf(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod, numpy_output=False)

    var_x = tf.reshape(var_x, [batch_size, max_seq_length])
    var_y = tf.reshape(var_y, [batch_size, max_seq_length])
    hov = tf.reshape(hov, [batch_size, max_seq_length])
    eod = tf.reshape(eod, [batch_size, max_seq_length])

    return var_x, var_y, hov, eod

def sample_output_seq(y_pred_sample, seq_len_sample, dx_norm, dy_norm, d2x_norm, d2y_norm, hparams, use_eod=False):
    y_pred_sample = y_pred_sample[:seq_len_sample]
    t = 0
    eod = 0
    dx, dy = 0, 0
    dx_seq = list(); dy_seq = list(); hov_seq = list(); eod_seq = list()
    while t < seq_len_sample and (eod != 1 or not use_eod):
        var_x, var_y, hov, eod = sample_gmm_batch_tf(*parse_pred(y_pred_sample[t,:], hparams))
        if hparams['target'] == 'v':
            dx = var_x * dx_norm
            dy = var_y * dy_norm
        else:
            d2x = var_x * d2x_norm
            d2y = var_y * d2y_norm
            dx += d2x
            dy += d2y
        dx_seq.append(dx); dy_seq.append(dy); hov_seq.append(hov); eod_seq.append(eod)
        t += 1

    # DataFrame of direction and speed.
    df_v = pd.DataFrame({'dx':dx_seq, 'dy':dy_seq, 'hover':hov_seq, 'eod':eod_seq})

    return df_v

def sample_output_seq_v1(y_pred_sample, seq_len_sample, dx_norm, dy_norm, d2x_norm, d2y_norm, hparams, use_eod=False):
    y_pred_sample = y_pred_sample[:seq_len_sample]
    t = 0
    eod = 0
    dx_seq = list(); dy_seq = list(); hov_seq = list(); eod_seq = list()
    while t < seq_len_sample and (eod != 1 or not use_eod):
        next_dx, next_dy, hov, eod = sample_gmm_batch_tf(*parse_pred(y_pred_sample[t,:], hparams))
        dx_seq.append(next_dx * dx_norm); dy_seq.append(next_dy * dy_norm); hov_seq.append(hov); eod_seq.append(eod)
        t += 1

    # DataFrame of direction and speed.
    df_v = pd.DataFrame({'dx':dx_seq, 'dy':dy_seq, 'hover':hov_seq, 'eod':eod_seq})

    return df_v

def tf_2d_normal( x1, x2, mu1, mu2, s1, s2, rho ):
    # From https://github.com/edwin-de-jong/incremental-sequence-learning/blob/ab1cd9ef815094fcd0f272f1b4fc6d2f841ea2a5/model.py#L74
    # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    #dims: mu1, mu2: batch_nrpoints x nrmixtures

    rho = tf.zeros_like(rho) # Test

    norm1 = tf.subtract( x1, mu1 ) #batch_nrpoints x nrmixtures
    norm2 = tf.subtract( x2, mu2 )
    s1s2 = tf.multiply( s1, s2 )
    normprod = tf.multiply( norm1, norm2 ) #batch_nrpoints x nrmixtures; here x1 and x2 are combined

    epsilon = 1e-10
    z = tf.square( tf.divide( norm1, s1 + epsilon ) ) + tf.square( tf.divide( norm2, s2 + epsilon ) ) - 2 * tf.divide( tf.multiply( rho, normprod ), s1s2 + epsilon ) #batch_nrpoints x nrmixtures
    negRho = 1 - tf.square( rho ) #EdJ: Problem: can become 0 if corr is 1 --> denom becomes zero --> nan result, resolved by multiplying z_corr_tanh with 0.95
    result5 = tf.exp( tf.divide( - z, 2 * negRho ) )

    denom = 2 * np.pi * tf.multiply( s1s2, tf.sqrt( negRho ) )
    result6 = tf.divide( result5, denom )

    return result6 #still batch_nrpoints x nrmixtures

def parse_pred(pred, hparams, only_gmm=False):
    n_g_mixtures = hparams['n_g_mixtures']
    v_eod_dim = hparams['v_eod_dim']
    v_hover_dim = hparams['v_hover_dim']

    z_pi = pred[...,:n_g_mixtures]
    z_mu1 = pred[...,n_g_mixtures:2*n_g_mixtures]
    z_mu2 = pred[...,2*n_g_mixtures:3*n_g_mixtures]
    z_sigma1 = pred[...,3*n_g_mixtures:4*n_g_mixtures]
    z_sigma2 = pred[...,4*n_g_mixtures:5*n_g_mixtures]
    z_corr = pred[...,5*n_g_mixtures:6*n_g_mixtures]

    if not only_gmm:
        z_hov = pred[...,-(v_eod_dim+v_hover_dim):-(v_eod_dim)]
        z_eod = pred[...,-(v_eod_dim):]

        return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod
    else:
        return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr


def parse_target(target):
    x1_data = target[...,:1]
    x2_data = target[...,1:2]
    hov_data = target[...,2:3]
    eod_data = target[...,3:]

    return x1_data, x2_data, hov_data, eod_data

def parse_inputs(inputs):
    var_x = inputs[...,:1]
    var_y = inputs[...,1:2]
    hov = inputs[...,2:3]
    eod = inputs[...,3:4]
    character_vector = inputs[...,4:]

    return var_x, var_y, hov, eod, character_vector

def get_dot_seq_from_dx_dy(df_v, w_img, h_img, dt):
    seq_len = df_v.shape[0] + 1
    x_seq = np.zeros(seq_len)
    y_seq = np.zeros(seq_len)
    hov_seq = np.zeros(seq_len) # Its start and end should be 0.
    eod_seq = np.zeros(seq_len)

    x, y = 0, 0
    v_hov_pre = 0 # Not hovering: 0, hovering: 1.
    x_seq[0] = x
    y_seq[0] = y
    eod_seq[1:] = df_v.eod

    for i, row in df_v.iterrows():
        dx = row.dx
        dy = row.dy
        v_hov = row.hover
        eod = row.eod
        x += dx; y += dy
        x_seq[i+1] = x; y_seq[i+1] = y
        # Set hover. At least one neighboring movement is not hovering => not hovering.
        hov_seq[i] = 0 if (v_hov_pre == 0 or v_hov == 0) else 1
        v_hov_pre = v_hov
    # Set the last hover.
    hov_seq[-1] = 0 # No hovering.

    # Move the trajectory to be on the center.
    x_center = (x_seq.max() + x_seq.min()) // 2
    y_center = (y_seq.max() + y_seq.min()) // 2
    x_mv = w_img // 2 - x_center
    y_mv = h_img // 2 - y_center
    x_seq += x_mv
    y_seq += y_mv

    df_dots = pd.DataFrame({'x':x_seq, 'y':y_seq, 'hover':hov_seq, 'eod':eod_seq})

    return df_dots

def get_df_v_from_y_target(y_target, seq_len, dx_mean, dx_norm, dy_mean, dy_norm):
    dx_seq = y_target[:seq_len,0] * dx_norm + dx_mean
    dy_seq = y_target[:seq_len,1] * dy_norm + dy_mean
    hov_seq = y_target[:seq_len,2]
    eod_seq = y_target[:seq_len,3]

    df_v = pd.DataFrame({'dx':dx_seq, 'dy':dy_seq, 'hover':hov_seq, 'eod':eod_seq})

    return df_v

def get_seq_length(eods):
    seq_length = len(eods)
    for i, eod in enumerate(eods):
        if eod == 1:
            seq_length = i + 1
            break
    return seq_length

def get_writing_length(df_v):
    return get_seq_length(df_v.eod)

def cut_by_eod(df_v):
    length = get_writing_length(df_v)
    return df_v[:length]

def plot_behavior_distributions(character_set, stat_test_data_character, stat_gen_data_character, itr, logdir, bins=50, dt=20):

    features = ['speed_means', 'lengths', 'durations', 'diameters']

    feature_in_title = {
        'speed_means':'speed',
        'durations':'duration',
        'lengths':'length',
        'diameters':'diameter',
    }

    feature_in_xlabel = {
        'speed_means':'Speed (px/ms)',
        'durations':'Duration (ms)',
        'lengths':'Length (px)',
        'diameters':'Diameter (px)',
    }

    pdf_paths = dict()
    png_paths = dict()

    #leg_locs = ['upper right', 'upper right', 'upper left', 'upper left']
    leg_locs = ['best', 'best', 'best', 'best' ]
    for feature, leg_loc in zip(features, leg_locs):
        test_data = np.concatenate([stat_test_data_character[character][feature] for character in character_set])
        gen_data = np.concatenate([stat_gen_data_character[character][feature] for character in character_set])
        if feature == 'speed_means':
            test_data = test_data / dt
            gen_data = gen_data / dt
        elif feature == 'durations':
            test_data = test_data * dt
            gen_data = gen_data * dt

        rv_min = min(test_data.min(), gen_data.min())
        rv_max = max(test_data.max(), gen_data.max())
        rv_range = (rv_min, rv_max)

        density_test, bin_edges = np.histogram(test_data, bins=bins, range=rv_range, density=True)
        density_test = density_test / density_test.sum()
        density_gen, bin_edges = np.histogram(gen_data, bins=bins, range=rv_range, density=True)
        density_gen = density_gen / density_gen.sum()
        p_max = max(density_test.max(), density_gen.max())

        rv_test = (bin_edges[1:] + bin_edges[:-1]) / 2
        rv_gen = (bin_edges[1:] + bin_edges[:-1]) / 2
        widths = (bin_edges[1:] - bin_edges[:-1])

        plt.figure(figsize=(2.5,2.5), dpi=150)
        #plt.ylim((0,0.12))
        plt.ylim((0,p_max))
        plt.xlim(rv_range)
        plt.bar(rv_test, density_test, width=widths, alpha=0.5, edgecolor='black', color='blue', label='Human')
        plt.bar(rv_gen, density_gen, width=widths, alpha=0.5, edgecolor='black', color='red', label='Model')
        plt.axvline(test_data.mean(), alpha=0.5, color='blue')
        plt.axvline(gen_data.mean(), alpha=0.5, color='red')
        #plt.title('Distributions of the Human and Model'.format(character), fontsize=10)
        handles, labels = plt.gca().get_legend_handles_labels()
        #order = [2,3,0,1]
        #handles = [handles[idx] for idx in order]
        #labels = [labels[idx] for idx in order]
        plt.legend(handles, labels, loc=leg_loc, fontsize=8)
        plt.xlabel(feature_in_xlabel[feature], fontsize=12)
        plt.xticks(fontsize=8)
        plt.ylabel('Population Ratio', fontsize=12)
        plt.yticks(fontsize=8)
        plt.tight_layout(pad=0.08)
        pdf_path = join(logdir, 'b_dist-{}-it-{}.pdf'.format(feature_in_title[feature], itr))
        png_path = join(logdir, 'b_dist-{}-it-{}.png'.format(feature_in_title[feature], itr))
        plt.savefig(pdf_path)
        plt.savefig(png_path)
        #plt.show()
        plt.close()

        pdf_paths[feature] = pdf_path
        png_paths[feature] = png_path

    return pdf_paths, png_paths
