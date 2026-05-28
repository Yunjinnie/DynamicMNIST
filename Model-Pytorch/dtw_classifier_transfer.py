import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os
import torch # TensorFlow 대신 PyTorch import
from tslearn.clustering import TimeSeriesKMeans
from tslearn.barycenters import softdtw_barycenter
from tslearn.utils import to_time_series
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.metrics import soft_dtw
from utils import get_dot_seq_from_dx_dy, parse_pred
from data_utils_transfer_revised import rasterize_dev

class DTWClassifier:
    def __init__(self, hp, avg_set, log_dir, dx_scale, dy_scale):
        self.hp = hp # Hyperparameters
        self.dt = hp['dt']
        self.avg_set = avg_set # The dataset to be averaged to be used centroids.
        
        # 만약 avg_set의 데이터가 PyTorch Tensor라면 Numpy로 변환
        self.inputs_avg = avg_set[0].cpu().numpy() if isinstance(avg_set[0], torch.Tensor) else avg_set[0]
        self.targets_avg = avg_set[1].cpu().numpy() if isinstance(avg_set[1], torch.Tensor) else avg_set[1]
        self.seq_lengths_avg = avg_set[2].cpu().numpy() if isinstance(avg_set[2], torch.Tensor) else avg_set[2]
        self.characters_avg = avg_set[3].cpu().numpy() if isinstance(avg_set[3], torch.Tensor) else avg_set[3]
        
        self.log_dir = os.path.join(log_dir, 'dtw_classifier')
        os.makedirs(self.log_dir, exist_ok=True)
        self.dx_scale = dx_scale
        self.dy_scale = dy_scale
        self.character_set = list(hp['character_set']) #### 맞는지 확인

        self.motor_programs = {c:1 for c in self.character_set}
        for dict_character_n_mp in self.hp['n_motor_programs']:
            for character, n_mp in dict_character_n_mp.items():
                self.motor_programs[character] = n_mp

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        self.character_set = list(self.motor_programs.keys())
        self.dim_names = {0:'dx',1:'dy',2:'x',3:'y', 4:'hover'}
        self.dims = list(self.dim_names.values())
        self.n_dims = len(self.dim_names)
        self.seed = 97531

        ## error
        self.indexes_by_motor_programs = self.cluster_motor_programs()
        #self.print_clutering_results()
        self.compute_barycenters()
        self.draw_barycenters() # 주석 처리되어 있던 부분 ... 이러니까 실행이 안되지

    def get_img_paths(self):
        img_paths = dict()
        img_paths['plot'] = dict()
        img_paths['gif'] = dict()
        for character, n_motor_programs in self.motor_programs.items():
            img_paths['plot'][character] = dict()
            img_paths['gif'][character] = dict()
            for motor_program in range(n_motor_programs):
                img_paths['plot'][character][motor_program] = os.path.join(self.log_dir, 'barycenter-character-{}-mp-{}.png'.format(character, motor_program))
                img_paths['gif'][character][motor_program] = os.path.join(self.log_dir, 'barycenter-img-character-{}-mp-{}.gif'.format(character, motor_program))

        return img_paths

    def cluster_motor_programs(self):
        indexes_by_motor_programs = {character:[list() for _ in range(self.motor_programs[character])] for character in self.character_set}

        # for character in self.character_set:
        for character, n_motor_programs in self.motor_programs.items():
            # 1. 해당 문자에 해당하는 시퀀스(seqs)들을 추출하는 부분 확인
            # self.characters -> self.characters_avg로 통일
            indexes_character = np.squeeze(np.argwhere(self.characters_avg == character))
            
            # 인덱스가 스칼라로 반환될 경우를 대비해 array화
            indexes_character = np.atleast_1d(indexes_character)

            # 샘플이 없는 경우 클러스터링을 건너뜀
            if len(indexes_character) == 0:
                print(f"[DTW Classifier] No samples found for character '{character}'. Skipping clustering.")
                continue

            if n_motor_programs == 1:
                # 1차원 배열로 평탄화하여 리스트에 추가    
                indexes_by_motor_programs[character][0] += list(np.atleast_1d(indexes_character))
                continue
            
            # error
            targets_character = [self.targets_avg[i] for i in indexes_character] #self.targets_avg[indexes_character]
            n_samples = len(targets_character) # targets_character.shape[0]
            n_clusters = n_motor_programs

            seqs = list()
            # seqs = self.inputs[indexes_character] # (N, Time, Features) 형태
        
            # 만약 seqs가 비어있지 않더라도, 시계열 데이터 가공 후 0이 될 수 있으므로 한 번 더 체크
            # if seqs.shape[0] == 0:
            #     continue

            groups = list()
            max_seq_len = 0

            for i_sample in range(n_samples):
                sample = targets_character[i_sample] # [L, 3] (dx, dy, hover) 가정
                # max_seq_len = max(max_seq_len, sample.shape[0])
                dx = sample[:,0]
                dy = sample[:,1]
                hover = sample[:,2]
                # x = accumulate_sequence(dx)
                # y = accumulate_sequence(dy)
                # 누적 좌표 계산 (x, y)
                x = np.cumsum(dx)
                y = np.cumsum(dy)

                seq = np.stack([dx,dy,x,y,hover], axis=-1)
                seqs.append(seq)
                max_seq_len = max(max_seq_len, seq.shape[0])
            #groups += [character] * n_samples

            '''
            # Convert data into numpy.ndarray.
            seqs_tmp = np.full((n_samples, max_seq_len, self.n_dims), np.NaN, dtype=object)
            for i in range(n_samples):
                seq_len = seqs[i].shape[0]
                seqs_tmp[i,:seq_len,:] = seqs[i]
            seqs = seqs_tmp
            groups = np.asarray(groups)

            # DBA-k-means
            print("Soft-DTW K-means to cluster characters according to the motor programs of a single character ...")
            dtw_km = TimeSeriesKMeans(n_clusters=n_clusters,
                                      max_iter=50,
                                      n_init=5,
                                      metric="softdtw",
                                      verbose=False,
                                      max_iter_barycenter=0,
                                      n_jobs=48,
                                      random_state=self.seed)
            ##
            # 에러가 발생했던 지점: fit_predict
            try:
                # 기존 코드
                clusters_pred = dtw_km.fit_predict(seqs)
                # (결과 저장 로직)
                for i_cluster in np.unique(clusters_pred):
                    indexes_cluster = np.squeeze(np.argwhere(clusters_pred == i_cluster))
                    indexes_by_motor_programs[character][i_cluster] += list(np.atleast_1d(indexes_character[indexes_cluster]))

            except ValueError as e:
                # 만약 여기서도 에러가 난다면 샘플 수가 너무 적은 것 (예: K-means인데 샘플이 1개뿐일 때)
                print(f"[DTW Classifier] Clustering failed for '{character}': {e}")
                continue
            '''

            # TimeSeriesKMeans를 위한 시퀀스 패딩 (tslearn 포맷: n_samples, max_len, dims)
            seqs_padded = np.zeros((n_samples, max_seq_len, self.n_dims))
            for i, s in enumerate(seqs):
                l = s.shape[0]
                seqs_padded[i, :l, :] = s

            print(f"Clustering character '{character}' into {n_clusters} motor programs...")
            
            dtw_km = TimeSeriesKMeans(
                n_clusters=n_clusters,
                max_iter=50,
                metric="softdtw",
                metric_params={"gamma": 0.01}, # softdtw의 평활화 파라미터
                n_init=1,
                n_jobs=48,
                random_state=self.seed
            )
            
            clusters_pred = dtw_km.fit_predict(seqs_padded)

            for i_sample, i_cluster in enumerate(clusters_pred):
                # 원래의 전체 데이터셋 인덱스를 저장
                real_idx = indexes_character[i_sample]
                indexes_by_motor_programs[character][i_cluster].append(real_idx)
            
        return indexes_by_motor_programs

    def print_clutering_results(self):
        print('[Results of clustering characters to classify motor programs]')
        for character, indexes_mps in self.indexes_by_motor_programs.items():
            for i_mp, indexes_mp in enumerate(indexes_mps):
                print('Character {} | MP {} | len = {}'.format(character, i_mp, len(indexes_mp)))

    def compute_barycenters(self):
        print('Compute barycenters ...')
        self.barycenters = {character:dict() for character in self.character_set}
        color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
            'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
            'tab:cyan']
        colors = {character:color_list[i%len(color_list)] for i, character in enumerate(self.character_set)}

        dx_abs_max = dy_abs_max = x_abs_max = y_abs_max = 0
        for i, sample in enumerate(self.targets_avg):
            dx = sample[:,0] * self.dx_scale
            dy = sample[:,1] * self.dy_scale
            x = accumulate_sequence(dx)
            y = accumulate_sequence(dy)
            dx_abs_max = max(dx_abs_max, np.max(np.absolute(dx)))
            dy_abs_max = max(dy_abs_max, np.max(np.absolute(dy)))
            x_abs_max = max(x_abs_max, np.max(np.absolute(x)))
            y_abs_max = max(y_abs_max, np.max(np.absolute(y)))

        for character, indexes_mps in self.indexes_by_motor_programs.items():
            for motor_program, indexes_mp in enumerate(indexes_mps):
                targets_character = self.targets_avg[indexes_mp]
                n_samples = len(indexes_mp)

                # 샘플이 0개인 경우 softdtw_barycenter를 계산할 수 없으므로 건너뜁니다.
                if n_samples == 0:
                    print(f" Warning: Character {character}, Motor Program {motor_program} has 0 samples. Skipping...")
                    continue


                seqs = list()
                groups = list()
                max_seq_len = 0

                for i_sample in range(n_samples):
                    sample = targets_character[i_sample]
                    max_seq_len = max(max_seq_len, sample.shape[0])
                    dx = sample[:,0]
                    dy = sample[:,1]
                    hover = sample[:,2]
                    x = accumulate_sequence(dx)
                    y = accumulate_sequence(dy)
                    seq = np.stack([dx,dy,x,y,hover], axis=-1)
                    seqs.append(seq)
                groups += [character] * n_samples

                seqs = np.array([to_time_series(seq, remove_nans=True) for seq in seqs], dtype=object)
                groups = np.asarray(groups)

                dtw_barycenter = softdtw_barycenter(seqs, max_iter=100, tol=0.1)
                self.barycenters[character][motor_program] = dict()
                self.barycenters[character][motor_program]['barycenter'] = dtw_barycenter

                mean_time = np.mean([seq.shape[0] for seq in seqs])
                barycenter_mean_time = np.squeeze(TimeSeriesResampler(sz=int(mean_time)).fit_transform(dtw_barycenter.T)).T
                self.barycenters[character][motor_program]['barycenter_mean_time'] = barycenter_mean_time

                plt.figure(figsize=(3.5, self.n_dims*2))

                dim_scales = [self.dx_scale, self.dy_scale, self.dx_scale, self.dy_scale, 1]
                y_ranges = [
                    (-dx_abs_max, dx_abs_max), (-dy_abs_max, dy_abs_max),
                    (-x_abs_max, x_abs_max), (-y_abs_max, y_abs_max),
                    (0-0.1, 1+0.1)
                ]
                for idim in range(self.n_dims):
                    plt.subplot(self.n_dims, 1, idim+1)
                    plt.title("Barycenter (Character {}, Motor Progam {}) - {}".format(character, motor_program, self.dim_names[idim]))
                    for i, seq in enumerate(seqs):
                        plt.ylim(y_ranges[idim])
                        plt.yticks(fontsize=18)
                        plt.xticks(fontsize=18)
                        if idim == self.n_dims - 1:
                            plt.tick_params(left=True, right=False , labelleft=True, labelbottom=True, bottom=True)
                        else:
                            plt.tick_params(left=True, right=False , labelleft=True, labelbottom=False, bottom=False)
                        plt.plot(seq[:,idim]*dim_scales[idim], "-", alpha=.1, color=colors[character])
                    plt.axhline(0, linestyle=':', color='black')
                    plt.plot(barycenter_mean_time[:,idim]*dim_scales[idim], "b-")

                plt.tight_layout()
                plt.savefig(os.path.join(self.log_dir, 'barycenter-character-{}-mp-{}.pdf'.format(character, motor_program)))
                plt.savefig(os.path.join(self.log_dir, 'barycenter-character-{}-mp-{}.svg'.format(character, motor_program)))
                plt.savefig(os.path.join(self.log_dir, 'barycenter-character-{}-mp-{}.png'.format(character, motor_program)))
                plt.close()

                df_bc = pd.DataFrame({col:dtw_barycenter[:,idim]*dim_scales[idim] for idim, col in self.dim_names.items()})
                df_bc.to_csv(os.path.join(self.log_dir, 'barycenter-character-{}-mp-{}.csv'.format(character, motor_program)), index=False)
                df_bc_r = pd.DataFrame({col:barycenter_mean_time[:,idim]*dim_scales[idim] for idim, col in self.dim_names.items()})
                df_bc_r.to_csv(os.path.join(self.log_dir, 'barycenter_mean_time_rescale-character-{}-mp-{}.csv'.format(character, motor_program)), index=False)

    def draw_barycenters(self):
        # Access: self.barycenters[character][motor_program]['barycenter_mean_time']
        print('Draw barycenters ...')
        for character in self.barycenters.keys():
            for motor_program, barycenter in self.barycenters[character].items():
                dx_seq = barycenter['barycenter_mean_time'][:,0]
                dy_seq = barycenter['barycenter_mean_time'][:,1]
                hov_seq = (barycenter['barycenter_mean_time'][:,4] > 0.5).astype(np.int64)
                eod_seq = np.zeros_like(dx_seq); eod_seq[-1] = 1

                df_v = pd.DataFrame.from_dict({
                    'dx':dx_seq * self.dx_scale,
                    'dy':dy_seq * self.dy_scale,
                    'hover':hov_seq,
                    'eod':eod_seq
                })

                #### 경로 체크할 것
                gif_path = os.path.join(self.log_dir, 'barycenter-img-character-{}-mp-{}.gif'.format(character, motor_program))
                png_path = os.path.join(self.log_dir, 'barycenter-img-character-{}-mp-{}.png'.format(character, motor_program))

                df_dots = get_dot_seq_from_dx_dy(df_v, w_img=256, h_img=256, dt=self.dt)
                np_images = rasterize_dev(df_dots, w=256, h=256, sampling_length=self.dt, stroke_width=10, gif_path=gif_path, png_path=png_path, eod_stop=False)

                print(f"[체크 1] 저장 시도 경로: {gif_path}")
                print(f"[체크 2] 원본 궤적(df_v) 길이: {len(df_v)}")
                print(f"[체크 3] 변환된 점(df_dots) 길이: {len(df_dots)}")

    def classify(self, seqs_eval, labels_eval, get_eval_pairs=False):
        # 만약 입력이 PyTorch 텐서라면 numpy로 변환
        if isinstance(seqs_eval, torch.Tensor):
            seqs_eval = seqs_eval.cpu().numpy()
        if isinstance(labels_eval, torch.Tensor):
            labels_eval = labels_eval.cpu().numpy()
            
        n_samples_total = 0
        n_correct_total = 0

        accuracy_character = dict()
        targets = list()
        predictions = list()

        for character in self.character_set:
            indexes_character = np.squeeze(np.argwhere(labels_eval == character))
            if indexes_character.size == 0:
                continue
                
            samples = seqs_eval[indexes_character]
            # 만약 샘플이 1개라서 1D 배열로 나오면 2D로 변경
            if samples.ndim == 2:
                samples = np.expand_dims(samples, axis=0)
                
            n_samples = samples.shape[0]
            pred = np.empty(n_samples, dtype=object)
            
            for i_sample in range(n_samples):
                sample = samples[i_sample]
                dx = sample[:,0]
                dy = sample[:,1]
                hov = sample[:,2]
                x = accumulate_sequence(dx)
                y = accumulate_sequence(dy)
                x_seq_sample = np.stack([dx,dy,x,y,hov], axis=-1)

                dtw_min = np.inf
                character_dtw_min = 0
                mp_dtw_min = 0
                for character_bc, motor_programs_bc in self.barycenters.items():
                    for motor_program_bc, bc in motor_programs_bc.items():
                        barycenter = bc['barycenter_mean_time']
                        dtw_value = soft_dtw(x_seq_sample, barycenter)
                        if dtw_value < dtw_min:
                            dtw_min = dtw_value
                            character_dtw_min = character_bc
                            mp_dtw_min = motor_program_bc
                            
                pred[i_sample] = (character_dtw_min, mp_dtw_min)
                targets.append(character)
                predictions.append(character_dtw_min)
                
            classes, counts = np.unique(pred, return_counts=True)
            n_correct = 0
            for (character_dtw_min, mp_dtw_min), count in zip(classes, counts):
                if character == character_dtw_min:
                    n_correct += count
            accuracy_character[character] = n_correct / n_samples
            n_samples_total += n_samples
            n_correct_total += n_correct

        accuracy_total = n_correct_total / max(1, n_samples_total) # 0으로 나누기 방지

        if get_eval_pairs:
             return accuracy_total, accuracy_character, (targets, predictions)
        else:
            return accuracy_total, accuracy_character

def accumulate_sequence(sequence):
    acm = 0
    acm_seq = np.zeros_like(sequence)
    for i, element in enumerate(sequence):
        acm += element
        acm_seq[i] = acm
    return acm_seq

def compute_barycenter_outputs(log_dir, character, y_pred, seq_lengths, dx_scale, dy_scale, hparams):
    '''
    Compute the barycenter of sequence outputs in a batch.
    Outputs are in the format of [batch_size, max_seq_length, dim_output]
    '''
    # 입력값이 Numpy array인 경우 PyTorch Tensor로 일관성 있게 변환
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(seq_lengths, torch.Tensor):
        seq_lengths = torch.tensor(seq_lengths)

    colors = {
        0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red',
        4:'tab:purple', 5:'tab:brown', 6:'tab:pink', 7:'tab:gray',
        8:'tab:olive', 9:'tab:cyan'
    }

    dim_scales = [dx_scale, dy_scale, dx_scale, dy_scale, 1]
    dim_names = {0:'dx_mean',1:'dy_mean',2:'dx_std',3:'dy_std', 4:'correlation'}
    n_dims = len(dim_scales)

    pi, dx_mean, dy_mean, dx_std, dy_std, corr, _, _ = parse_pred(y_pred, hparams)

    # 💡 [핵심 변경 사항] TensorFlow 연산을 PyTorch 연산으로 대체
    # tf.multiply -> 단순 곱셈 (*), tf.reduce_sum -> torch.sum
    dx_mean_weighted_sum = torch.sum(pi * dx_mean, dim=-1)
    dy_mean_weighted_sum = torch.sum(pi * dy_mean, dim=-1)
    dx_std_weighted_sum = torch.sum(pi * dx_std, dim=-1)
    dy_std_weighted_sum = torch.sum(pi * dy_std, dim=-1)
    corr_weighted_sum = torch.sum(pi * corr, dim=-1)

    n_samples = seq_lengths.shape[0]
    seqs = list()

    for i_sample in range(n_samples):
        # 텐서 스칼라 값을 정수로 변환
        seq_length = int(seq_lengths[i_sample].item() if hasattr(seq_lengths[i_sample], 'item') else seq_lengths[i_sample])

        # `.numpy()` 대신 텐서에서 데이터를 바로 추출 (CPU로 이동 후)
        dx_mean_val = dx_mean_weighted_sum[i_sample, :seq_length].detach().cpu().numpy()
        dy_mean_val = dy_mean_weighted_sum[i_sample, :seq_length].detach().cpu().numpy()
        dx_std_val = dx_std_weighted_sum[i_sample, :seq_length].detach().cpu().numpy()
        dy_std_val = dy_std_weighted_sum[i_sample, :seq_length].detach().cpu().numpy()
        corr_val = corr_weighted_sum[i_sample, :seq_length].detach().cpu().numpy()

        seq = np.stack([dx_mean_val, dy_mean_val, dx_std_val, dy_std_val, corr_val], axis=-1)
        seqs.append(seq)

    seqs = np.array([to_time_series(seq, remove_nans=True) for seq in seqs], dtype=object)

    dtw_barycenter = softdtw_barycenter(seqs, max_iter=100, tol=0.1)

    mean_time = np.mean([seq.shape[0] for seq in seqs])
    barycenter_mean_time = np.squeeze(TimeSeriesResampler(sz=int(mean_time)).fit_transform(dtw_barycenter.T)).T

    plt.figure(figsize=(4.8, n_dims*2))

    for idim, dim_name in dim_names.items():
        plt.subplot(n_dims, 1, idim+1)
        plt.title("Barycenter (Character {}) - {}".format(character, dim_name))
        for i, seq in enumerate(seqs):
            plt.plot(seq[:,idim]*dim_scales[idim], "-", alpha=.1, color=colors[character%len(colors)])
        plt.axhline(0, linestyle=':', color='black')
        plt.plot(barycenter_mean_time[:,idim]*dim_scales[idim], "b-")

    plt.tight_layout()
    png_path = os.path.join(log_dir, 'barycenter-gmm_params-character-{}.png'.format(character))
    pdf_path = os.path.join(log_dir, 'barycenter-gmm_params-character-{}.pdf'.format(character))
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()

    df_bc_r = pd.DataFrame({col:barycenter_mean_time[:,idim]*dim_scales[idim] for idim, col in dim_names.items()})
    csv_path = os.path.join(log_dir, 'barycenter_mean_time_rescale-character-{}.csv'.format(character))
    df_bc_r.to_csv(csv_path, index=False)

    return png_path, pdf_path, csv_path