import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from os.path import join, isfile, splitext
from os import listdir
import math

def get_file_list(dir_path, ext='.csv'):
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

# Numpy 기반 GMM 샘플링 (디버깅용 레거시 유지)
def sample_gmm(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod):
    idx = get_pi_idx(np.random.rand(), z_pi)
    next_dx, next_dy = sample_gaussian_2d(z_mu1[idx], z_mu2[idx], z_sigma1[idx], z_sigma2[idx], z_corr[idx])
    hov = 1 if np.random.rand() < z_hov else 0
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


# PyTorch 텐서 기반 GMM 샘플링 (GPU 호환, numpy_output 등 호환성 추가)
def sample_gmm_batch_pt(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov=None, z_eod=None, numpy_output=True, only_gmm=False):
    batch_size = z_pi.shape[0]
    # z_pi가 NumPy 배열이라면 PyTorch 텐서로 변환해줌
    if isinstance(z_pi, np.ndarray):
        # 경고 방지를 위해 데이터 타입을 강제로 float32 등으로 맞추는 것이 좋습니다.
        z_pi = torch.tensor(z_pi, dtype=torch.float32)
    # 들어오는 모든 NumPy 변수들을 PyTorch 텐서로 변환...
    # torch.as_tensor를 쓰면 이미 텐서인 애들은 냅두고 NumPy인 애들만 변환해줌
    z_pi = torch.as_tensor(z_pi, dtype=torch.float32)
    z_mu1 = torch.as_tensor(z_mu1, dtype=torch.float32)
    z_mu2 = torch.as_tensor(z_mu2, dtype=torch.float32)
    z_sigma1 = torch.as_tensor(z_sigma1, dtype=torch.float32)
    z_sigma2 = torch.as_tensor(z_sigma2, dtype=torch.float32)
    z_corr = torch.as_tensor(z_corr, dtype=torch.float32)
    z_hov = torch.as_tensor(z_hov, dtype=torch.float32)
    z_eod = torch.as_tensor(z_eod, dtype=torch.float32)
    
    # 혹시 z_eos 나 z_eod (펜 끝남 신호) 같은 변수도 있다면 같이 변환해 주세요!
    # 예: z_eos = torch.as_tensor(z_eos, dtype=torch.float32)

    # 1. 인덱스 샘플링 (Categorical)
    # pi(혼합 가중치)를 기반으로 어느 가우시안 분포를 사용할지 인덱스 샘플링
    # z_pi shape: [batch_size, n_mixtures]
    dist_categorical = dist.Categorical(probs=z_pi)
    idx = dist_categorical.sample().unsqueeze(-1) # [batch_size, 1]
    
    # 2. 선택된 인덱스의 mu, sigma, corr 파라미터 추출 (Gather parameters)
    mu1 = torch.gather(z_mu1, -1, idx).squeeze(-1) # [batch_size]
    mu2 = torch.gather(z_mu2, -1, idx).squeeze(-1)
    sigma1 = torch.gather(z_sigma1, -1, idx).squeeze(-1)
    sigma2 = torch.gather(z_sigma2, -1, idx).squeeze(-1)
    corr = torch.gather(z_corr, -1, idx).squeeze(-1)
    
    # 3. Reparameterization Trick을 통한 이변량 정규 분포 샘플링
    z1 = torch.randn_like(mu1)
    z2 = torch.randn_like(mu2)
    
    dx = mu1 + sigma1 * z1
    dy = mu2 + sigma2 * (z1 * corr + z2 * torch.sqrt(1.0 - corr**2 + 1e-8))

    if only_gmm:
        if numpy_output:
            return dx.cpu().numpy(), dy.cpu().numpy()
        return dx, dy

    # 4. Hover, EOD 샘플링 (Bernoulli)
    # 입력이 텐서인지 확인 후 처리 (1D 배열일 수 있으므로)
    hov_prob = z_hov.squeeze(-1) if z_hov is not None else torch.zeros_like(dx)
    eod_prob = z_eod.squeeze(-1) if z_eod is not None else torch.zeros_like(dx)
    
    hov = torch.bernoulli(hov_prob).to(torch.int32) # [batch_size]
    eod = torch.bernoulli(eod_prob).to(torch.int32) # [batch_size]

    if numpy_output:
        return dx.cpu().numpy(), dy.cpu().numpy(), hov.cpu().numpy(), eod.cpu().numpy()
    return dx, dy, hov, eod

def sample_gmm_seqs_pt(pred, hparams):
    parsed = parse_pred(pred, hparams)
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod = parsed

    batch_size = z_pi.shape[0]
    max_seq_length = z_pi.shape[1]

    # Flatten for batch sampling
    z_pi = z_pi.reshape(-1, z_pi.shape[-1])
    z_mu1 = z_mu1.reshape(-1, z_mu1.shape[-1])
    z_mu2 = z_mu2.reshape(-1, z_mu2.shape[-1])
    z_sigma1 = z_sigma1.reshape(-1, z_sigma1.shape[-1])
    z_sigma2 = z_sigma2.reshape(-1, z_sigma2.shape[-1])
    z_corr = z_corr.reshape(-1, z_corr.shape[-1])
    z_hov = z_hov.reshape(-1, 1)
    z_eod = z_eod.reshape(-1, 1)

    var_x, var_y, hov, eod = sample_gmm_batch_pt(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod, numpy_output=False)

    var_x = var_x.view(batch_size, max_seq_length)
    var_y = var_y.view(batch_size, max_seq_length)
    hov = hov.view(batch_size, max_seq_length)
    eod = eod.view(batch_size, max_seq_length)

    return var_x, var_y, hov, eod

# v1?
def sample_output_seq(y_pred_sample, seq_len_sample, dx_norm, dy_norm, d2x_norm, d2y_norm, hparams, use_eod=False):
    y_pred_sample = y_pred_sample[:seq_len_sample]
    t = 0
    eod = 0
    dx, dy = 0.0, 0.0
    dx_seq, dy_seq, hov_seq, eod_seq = [], [], [], []
    
    while t < seq_len_sample and (eod != 1 or not use_eod):
        parsed = parse_pred(y_pred_sample[t:t+1, :], hparams)
        var_x, var_y, hov, eod = sample_gmm_batch_pt(*parsed, numpy_output=True)
        
        var_x, var_y, hov, eod = var_x[0], var_y[0], hov[0], eod[0]

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

    df_v = pd.DataFrame({'dx':dx_seq, 'dy':dy_seq, 'hover':hov_seq, 'eod':eod_seq})
    return df_v

'''
원본 논문 코드

def pt_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """ PyTorch 최적화 이변량 정규분포 PDF 계산 """
    rho = torch.zeros_like(rho) # 원본 코드의 Test 주석 반영 (상관계수 무시)

    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    normprod = norm1 * norm2

    epsilon = 1e-10
    z = (norm1 / (s1 + epsilon))**2 + (norm2 / (s2 + epsilon))**2 - 2 * (rho * normprod) / (s1s2 + epsilon)
    
    negRho = 1.0 - rho**2 + epsilon
    result5 = torch.exp(-z / (2 * negRho))

    denom = 2 * math.pi * s1s2 * torch.sqrt(negRho)
    result6 = result5 / denom

    return result6 
'''

def pt_2d_normal(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    """
    이변량 정규 분포(Bivariate Normal Distribution)의 확률 밀도(PDF)를 계산
    x1, x2: [batch_size, seq_len, 1]
    mu1...rho: [batch_size, seq_len, n_mixtures]

    From https://github.com/edwin-de-jong/incremental-sequence-learning/blob/ab1cd9ef815094fcd0f272f1b4fc6d2f841ea2a5/model.py#L74
    eq # 24 and 25 of http://arxiv.org/abs/1308.0850
    dims: mu1, mu2: batch_nrpoints x nrmixtures
    """
    # 분산 안정화를 위해 아주 작은 값을 더해줌 (NaN 방지)
    sigma1 = torch.clamp(sigma1, min=1e-8)
    sigma2 = torch.clamp(sigma2, min=1e-8)
    #print(rho)
    # 1. tanh를 통해 범위를 (-1, 1)로 강제 변환
    rho = torch.tanh(rho)

    # 2. clamp를 통해 물리적으로 극한값 도달 차단 (여유를 두어 0.95 또는 0.99 사용)
    rho = torch.clamp(rho, min=-0.95, max=0.95)
    
    Z = ((x1 - mu1) / sigma1)**2 + ((x2 - mu2) / sigma2)**2 \
        - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
    
    denom = 2 * math.pi * sigma1 * sigma2 * torch.sqrt(1.0 - rho**2 + 1e-8)
    exp_term = torch.exp(-Z / (2.0 * (1.0 - rho**2 + 1e-8)))
    
    return exp_term / denom


def parse_pred(pred, hparams, only_gmm=False):
    n_g_mixtures = hparams['n_g_mixtures']
    v_eod_dim = hparams['v_eod_dim']
    v_hover_dim = hparams['v_hover_dim']

    z_pi = pred[..., :n_g_mixtures]
    z_mu1 = pred[..., n_g_mixtures : 2*n_g_mixtures]
    z_mu2 = pred[..., 2*n_g_mixtures : 3*n_g_mixtures]
    z_sigma1 = pred[..., 3*n_g_mixtures : 4*n_g_mixtures]
    z_sigma2 = pred[..., 4*n_g_mixtures : 5*n_g_mixtures]
    z_corr = pred[..., 5*n_g_mixtures : 6*n_g_mixtures]

    if not only_gmm:
        z_hov = pred[..., -(v_eod_dim + v_hover_dim) : -v_eod_dim]
        z_eod = pred[..., -v_eod_dim :]
        return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod
    else:
        return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr

def parse_target_pt(target):
    x1_data = target[..., :1]
    x2_data = target[..., 1:2]
    hov_data = target[..., 2:3]
    eod_data = target[..., 3:]
    return x1_data, x2_data, hov_data, eod_data

def parse_inputs_pt(inputs):
    var_x = inputs[..., :1]
    var_y = inputs[..., 1:2]
    hov = inputs[..., 2:3]
    eod = inputs[..., 3:4]
    character_vector = inputs[..., 4:]
    return var_x, var_y, hov, eod, character_vector

# DataFrame을 다루는 하위 함수들은 Tensor/Numpy 의존성이 크지 않으므로 유지함
def get_dot_seq_from_dx_dy(df_v, w_img, h_img, dt):
    seq_len = df_v.shape[0] + 1
    x_seq = np.zeros(seq_len)
    y_seq = np.zeros(seq_len)
    hov_seq = np.zeros(seq_len)  # Its start and end should be 0.
    eod_seq = np.zeros(seq_len)

    x, y = 0, 0
    v_hov_pre = 0 # Not hovering: 0, hovering: 1.
    x_seq[0] = x
    y_seq[0] = y
    eod_seq[1:] = df_v.eod

    for i, row in df_v.iterrows():
        dx, dy, v_hov = row.dx, row.dy, row.hover
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
    x_seq += (w_img // 2 - x_center)
    y_seq += (h_img // 2 - y_center)

    df_dots = pd.DataFrame({'x':x_seq, 'y':y_seq, 'hover':hov_seq, 'eod':eod_seq})
    
    return df_dots

def get_df_v_from_y_target(y_target, seq_len, dx_mean, dx_norm, dy_mean, dy_norm):
    # 만약 y_target이 PyTorch 텐서일 경우 numpy로 변환
    if torch.is_tensor(y_target):
        y_target = y_target.cpu().numpy()
        
    dx_seq = y_target[:seq_len, 0] * dx_norm + dx_mean
    dy_seq = y_target[:seq_len, 1] * dy_norm + dy_mean
    hov_seq = y_target[:seq_len, 2]
    eod_seq = y_target[:seq_len, 3]

    df_v = pd.DataFrame({'dx':dx_seq, 'dy':dy_seq, 'hover':hov_seq, 'eod':eod_seq})

    return df_v

def get_seq_length(eods):
    for i, eod in enumerate(eods):
        if eod == 1:
            return i + 1
    return len(eods)

def get_writing_length(df_v):
    return get_seq_length(df_v.eod)

def cut_by_eod(df_v):
    length = get_writing_length(df_v)
    return df_v[:length]

def plot_behavior_distributions(character_set, stat_test_data_character, stat_gen_data_character, itr, logdir, bins=50, dt=20):
    # 시각화 로직은 Matplotlib 기반이므로 기존 로직 유지
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
