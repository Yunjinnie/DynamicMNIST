import os
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"

import numpy as np
import pandas as pd
import torch
import wandb
import scipy.stats as stats
from shutil import copyfile, rmtree

from metrics import get_target_gmm_mean_distance, get_hov_accuracy, get_eod_accuracy, get_sampled_dxdy_distance, get_gmm_mean_std
from metrics import get_stat_data_from_test_set, get_stat_data_from_outputs, compute_stat_distance, rm_outliers
from data_utils_transfer_revised import get_character_from_vector, rasterize_dev
from utils import get_df_v_from_y_target, get_dot_seq_from_dx_dy, sample_output_seq, cut_by_eod, get_batch_iterations, get_seq_length, \
    plot_behavior_distributions

# 분류기 임포트는 사용 환경에 맞게 유지
from dtw_classifier import DTWClassifier
from cnn_classifier import CNNClassifier

class Logger:
    '''
    Save files in 'dir_log/prj_name/run_id'.
    Save hparams.yaml in 'dir_log/prj_name'.
    '''
    def __init__(self, project_name, run_id, hparams, off_wandb):
        self.character_set = list(hparams['character_set'])
        self.prj_name = project_name
        self.run_id = run_id
        self.hp = hparams

        self.dx_mean = 0
        self.dy_mean = 0
        self.dx_norm = 1
        self.dy_norm = 1

        self.dir_log = hparams['dir_log']
        self.dir_log_path = os.path.join(self.dir_log, project_name, run_id)
        self.dir_checkpoint_path = os.path.join(self.dir_log_path, 'ckpt')
        self.dir_generation_path = os.path.join(self.dir_log_path, 'gen')
        self.dir_distribution_path = os.path.join(self.dir_log_path, 'b_dist')
        self.dir_gmm_params_path = os.path.join(self.dir_log_path, 'gmm_params')

        self.v_character_dim = hparams['v_character_dim']
        self.w_img = hparams['w_img']
        self.h_img = hparams['h_img']
        self.dt = hparams['dt']
        self.n_epochs = hparams['n_epochs']
        self.batch_size = hparams['batch_size']
        self.stroke_width = hparams['stroke_width']

        self.accuracy_total = None
        self.accuracy_total_max = 0
        self.accuracy_total_dtw = None
        self.accuracy_total_cnn = None
        self.accuracy_total_dtw_test = None
        self.accuracy_total_cnn_test = None
        self.accuracy_character = None

        self.core_metrics = ['speed_means', 'durations', 'lengths', 'diameters']

        self.make_log_dir()

        if off_wandb:
            self.off_wandb_logging()

        wandb.init(
            project=project_name,
            name=run_id,
            id=run_id,
            resume=None
        )

        # PyTorch 환경에서는 TF 로그 레벨 설정이 필수는 아니지만, 충돌 방지를 위해 남겨둡니다.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def init_dtw_classifier(self, val_set):
        ### error
        # img_path == dtw_classifier 경로와 맞출 것
        self.dtw_classifier = DTWClassifier(self.hp, val_set, self.dir_log_path, self.dx_norm, self.dy_norm)
        img_paths = self.dtw_classifier.get_img_paths()
        #print('img paths:', img_paths)
        '''
        img paths: {'plot': {'0': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-0-mp-0.png'}, '1': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-1-mp-0.png'}, '2': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-2-mp-0.png'}, '3': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-3-mp-0.png'}, '4': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-4-mp-0.png'}, '5': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-5-mp-0.png'}, '6': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-6-mp-0.png'}, '7': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-7-mp-0.png'}, '8': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-8-mp-0.png'}, '9': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-9-mp-0.png', 1: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-character-9-mp-1.png'}}, 
        'gif': {'0': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-0-mp-0.gif'}, '1': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-1-mp-0.gif'}, '2': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-2-mp-0.gif'}, '3': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-3-mp-0.gif'}, '4': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-4-mp-0.gif'}, '5': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-5-mp-0.gif'}, '6': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-6-mp-0.gif'}, '7': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-7-mp-0.gif'}, '8': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-8-mp-0.gif'}, '9': {0: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-9-mp-0.gif', 1: 'logs/digit-10/run-260403142543/dtw_classifier/barycenter-img-character-9-mp-1.gif'}}}
        '''
        for character, n_motor_programs in self.dtw_classifier.motor_programs.items():
            for motor_program in range(n_motor_programs):
                #print('character', character)
                #print('motor program', motor_program)
                img_plot = img_paths['plot'][character][motor_program]
                img_gif = img_paths['gif'][character][motor_program]
                #print(img_plot)
                #print(img_gif)
                # 기존 코드에 try-except로 감싸서 파일이 없으면 무시하고 넘어가게 함
                try:
                    wandb.log({
                    'dtw-classification/barycenter/plot/character-{}-{}'.format(character, motor_program):wandb.Image(img_plot),
                    'dtw-classification/barycenter/gif/character-{}-{}'.format(character, motor_program):wandb.Image(img_gif),
                }, step=0)
                    
                except FileNotFoundError:
                    print(f"[Warning] {img_gif} 파일을 찾을 수 없어 wandb 업로드를 건너뜁니다.")
                    pass
                

    def init_cnn_classifier(self, data_loader):
        self.cnn_classifier = CNNClassifier(self.hp, data_loader.get_norms())

    def compute_stat_between_human_sets(self, train_set, val_set, test_set):
        stat_counter_data_character, stat_counter_data_all = get_stat_data_from_test_set(self.character_set, val_set, self.dx_norm, self.dy_norm)
        stat_test_data_character, stat_test_data_all = get_stat_data_from_test_set(self.character_set, test_set, self.dx_norm, self.dy_norm)

        metrics = list(stat_test_data_all.keys())

        d_stats = {metric:list() for metric in metrics}
        d_means = {metric:list() for metric in metrics}
        error_means = {metric:list() for metric in metrics}
        pvalues = {metric:{character:list() for character in self.character_set} for metric in metrics} 
        pvalues_mean_diff = {metric:{character:list() for character in self.character_set} for metric in metrics}
        pvalues_all = dict()

        p_samples_all = {metric:list() for metric in metrics}
        q_samples_all = {metric:list() for metric in metrics}

        for character in self.character_set:
            for metric in metrics:
                rv_range = (stat_test_data_all[metric].min(), stat_test_data_all[metric].max())
                p_samples = stat_test_data_character[character][metric]
                q_samples = stat_counter_data_character[character][metric]
                p_samples = rm_outliers(p_samples)
                q_samples = rm_outliers(q_samples)
                p_samples_all[metric].append(p_samples)
                q_samples_all[metric].append(q_samples)
                p_stats = stats.describe(p_samples)
                q_stats = stats.describe(q_samples)
                
                d_stat = compute_stat_distance(p_samples, q_samples, bins=100, range=rv_range)
                d_stats[metric].append(d_stat)
                d_mean = np.absolute(q_stats.mean - p_stats.mean)
                d_means[metric].append(d_mean)
                error_mean = q_stats.mean - p_stats.mean
                error_means[metric].append(error_mean)
                pvalue = stats.mannwhitneyu(p_samples, q_samples).pvalue
                pvalues[metric][character].append(pvalue) 
                pvalue_mean_diff = stats.ttest_ind(p_samples, q_samples, equal_var=False).pvalue
                pvalues_mean_diff[metric][character].append(pvalue_mean_diff)

                wandb.log({
                    'test-btw-human/{}/stat_distance/character-{}'.format(metric, character):d_stat,
                    'test-btw-human/{}/mean_distance/character-{}'.format(metric, character):d_mean,
                    'test-btw-human/{}/mean_error/character-{}'.format(metric, character):error_mean,
                    'test-btw-human/{}/p-value/character-{}'.format(metric, character):pvalue,
                    'test-btw-human/{}/p-value-mean_diff/character-{}'.format(metric, character):pvalue_mean_diff,
                    'test-btw-human/{}/range/character-{}-target.max'.format(metric, character):p_samples.max(),
                    'test-btw-human/{}/range/character-{}-target.mean'.format(metric, character):p_samples.mean(),
                    'test-btw-human/{}/range/character-{}-target.min'.format(metric, character):p_samples.min(),
                    'test-btw-human/{}/range/character-{}-pred.max'.format(metric, character):q_samples.max(),
                    'test-btw-human/{}/range/character-{}-pred.mean'.format(metric, character):q_samples.mean(),
                    'test-btw-human/{}/range/character-{}-pred.min'.format(metric, character):q_samples.min(),
                }, step=0)

        for metric in metrics:
            p_samples_all[metric] = np.concatenate(p_samples_all[metric])
            q_samples_all[metric] = np.concatenate(q_samples_all[metric])

        for metric in metrics:
            rv_range = (stat_test_data_all[metric].min(), stat_test_data_all[metric].max())
            p_samples = p_samples_all[metric]
            q_samples = q_samples_all[metric]
            p_stats = stats.describe(p_samples)
            q_stats = stats.describe(q_samples)
            
            d_stat = compute_stat_distance(p_samples, q_samples, bins=100, range=rv_range)
            d_mean = np.absolute(q_stats.mean - p_stats.mean)
            error_mean = q_stats.mean - p_stats.mean
            pvalue = stats.mannwhitneyu(p_samples, q_samples).pvalue
            pvalues_all[metric] = pvalue
            pvalue_mean_diff = stats.ttest_ind(p_samples, q_samples, equal_var=False).pvalue
            
            wandb.log({
                'test-btw-human/{}/stat_distance/total'.format(metric):d_stat,
                'test-btw-human/{}/mean_distance/total'.format(metric):d_mean,
                'test-btw-human/{}/mean_error/total'.format(metric):error_mean,
                'test-btw-human/{}/p-value/total'.format(metric):pvalue,
                'test-btw-human/{}/p-value-mean_diff/total'.format(metric):pvalue_mean_diff,
                'test-btw-human/{}/range/total-target.max'.format(metric):p_samples.max(),
                'test-btw-human/{}/range/total-target.mean'.format(metric):p_samples.mean(),
                'test-btw-human/{}/range/total-target.min'.format(metric):p_samples.min(),
                'test-btw-human/{}/range/total-pred.max'.format(metric):q_samples.max(),
                'test-btw-human/{}/range/total-pred.mean'.format(metric):q_samples.mean(),
                'test-btw-human/{}/range/total-pred.min'.format(metric):q_samples.min(),
            }, step=0)

        self.pvalues_stat_between_human_sets = pvalues
        self.pvalues_all_stat_between_human_sets = pvalues_all

    def set_dataset_size(self, size_train, size_test):
        self.size_train = size_train
        self.size_test = size_test
        self.iters_per_epoch = get_batch_iterations(self.size_train, self.batch_size)
        self.iters_total = self.iters_per_epoch * self.n_epochs

    def set_norms(self, dx_norm, dy_norm, d2x_norm, d2y_norm):
        self.dx_norm = dx_norm
        self.dy_norm = dy_norm
        self.d2x_norm = d2x_norm
        self.d2y_norm = d2y_norm

    def make_log_dir(self):
        dir_log_path = os.path.join(self.hp['dir_log'], self.prj_name, self.run_id)
        os.makedirs(dir_log_path, exist_ok=True)
        os.makedirs(self.dir_checkpoint_path, exist_ok=True)
        os.makedirs(self.dir_generation_path, exist_ok=True)
        os.makedirs(self.dir_distribution_path, exist_ok=True)
        os.makedirs(self.dir_gmm_params_path, exist_ok=True)

    def off_wandb_logging(self):
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ['WANDB_SILENT'] = 'true'

    def archive_file(self, file_path):
        file_name_ext = os.path.split(file_path)[-1]
        save_path = os.path.join(self.dir_log_path, file_name_ext)
        copyfile(file_path, save_path)

    def save_model_now(self):
        return self.behavioral_plausibility_strict()

    def behavioral_plausibility_strict(self):
        for m in self.core_metrics:
            for character, pvalue in enumerate(self.pvalues[m]):
                if pvalue <= 0.05:
                    return False
        return True

    def save_model(self, model, epoch_float, iterations):
        # PyTorch 스타일의 체크포인트 저장 방식
        filename = '{}-{}-epoch-{:0{}d}-it-{:0{}d}.pt'.format(
            self.prj_name, self.run_id,
            int(epoch_float), int(np.log10(self.n_epochs))+1,
            iterations, int(np.log10(self.iters_total))+1
        )
        
        checkpoint = {
            'epoch': int(epoch_float),
            'iterations': iterations,
            'model_state_dict': model.state_dict(),
            # 옵티마이저 정보도 저장하려면 아래 주석을 해제하세요.
            # 'optimizer_state_dict': model.optimizer.state_dict()
        }

        if not self.hp['save_only_best_model']:
            filepath = os.path.join(self.dir_checkpoint_path, filename)
            torch.save(checkpoint, filepath)

        if self.accuracy_total_max <= (self.accuracy_total or 0):
            dir_best_ckpt = os.path.join(self.dir_checkpoint_path, 'best')
            rmtree(dir_best_ckpt, ignore_errors=True)
            os.makedirs(dir_best_ckpt, exist_ok=True)
            filepath = os.path.join(dir_best_ckpt, filename)
            torch.save(checkpoint, filepath)

    def log_train(self, model, itr, epoch_float, batch, preds, smoothing_ratios, loss, sub_losses, corrector_outputs):
        inputs, targets, seq_lengths, mask = batch
        loss_gmm, loss_hov, loss_eod, loss_smoothing = sub_losses
        cr_pred, cr_loss = corrector_outputs

        dx_mae_target_gmm_mean, dy_mae_target_gmm_mean = get_target_gmm_mean_distance(targets, preds, seq_lengths, self.dx_mean, self.dx_norm, self.dy_mean, self.dy_norm, self.hp)
        dx_std_mean, dy_std_mean, corr_mean, corr_abs_mean, det_mean = get_gmm_mean_std(preds, seq_lengths, self.dx_norm, self.dy_norm, self.hp)
        hov_accuracy, hov_recall, hov_precision = get_hov_accuracy(targets, preds, seq_lengths, self.hp)
        eod_accuracy, eod_recall, eod_precision = get_eod_accuracy(targets, preds, seq_lengths, self.hp)

        # PyTorch 브로드캐스팅을 활용한 마스킹 및 평균 계산 최적화
        if smoothing_ratios.numel() > 1:
            smoothing_ratios = smoothing_ratios.squeeze(-1)
            # 패딩 영역을 무시하기 위한 마스크 생성
            s_mask = (torch.arange(smoothing_ratios.shape[1], device=smoothing_ratios.device)[None, :] < seq_lengths[:, None]).to(smoothing_ratios.dtype)
            mean_smooth_ratio = torch.sum(smoothing_ratios * s_mask) / torch.sum(seq_lengths).to(smoothing_ratios.dtype)
        else:
            mean_smooth_ratio = smoothing_ratios

        # PyTorch Optimizer에서 Learning Rate 추출
        lr = model.optimizer.param_groups[0]['lr'] if hasattr(model, 'optimizer') else 0.0

        wandb.log({
            'iterations':itr,
            'epoch':epoch_float,
            'learning_rate':lr,
            'train/loss/total':loss.item(),
            'train/loss/gmm':loss_gmm.item() if hasattr(loss_gmm, 'item') else loss_gmm,
            'train/loss/hov':loss_hov.item() if hasattr(loss_hov, 'item') else loss_hov,
            'train/loss/eod':loss_eod.item() if hasattr(loss_eod, 'item') else loss_eod,
            'train/loss/corrector':cr_loss.item() if hasattr(cr_loss, 'item') else cr_loss,
            'train/gmm/dx_mae_target_gmm_mean':dx_mae_target_gmm_mean,
            'train/gmm/dy_mae_target_gmm_mean':dy_mae_target_gmm_mean,
            'train/gmm/dx_std_mean':dx_std_mean,
            'train/gmm/dy_std_mean':dy_std_mean,
            'train/gmm/corr_mean':corr_mean,
            'train/gmm/corr_abs_mean':corr_abs_mean,
            'train/gmm/det_mean':det_mean,
            'train/hov/accuracy':hov_accuracy,
            'train/hov/recall':hov_recall,
            'train/hov/precision':hov_precision,
            'train/eod/accuracy':eod_accuracy,
            'train/eod/recall':eod_recall,
            'train/eod/precision':eod_precision,
            'train/smoothing_ratio/mean':mean_smooth_ratio.item() if hasattr(mean_smooth_ratio, 'item') else mean_smooth_ratio,
        }, step=itr)

        self.samples_train_synth = dict()
        batch_size = inputs.shape[0]
        # 입력이 PyTorch 텐서라면 numpy로 변환하여 사용
        inputs_np = inputs.detach().cpu().numpy() if isinstance(inputs, torch.Tensor) else inputs
        targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        seq_lengths_np = seq_lengths.detach().cpu().numpy() if isinstance(seq_lengths, torch.Tensor) else seq_lengths

        for i in range(batch_size):
            character = get_character_from_vector(inputs_np[i, 0, -self.v_character_dim:], self.character_set)
            if character not in self.samples_train_synth.keys():
                self.samples_train_synth[character] = {
                    'target_sample': targets_np[i],
                    'pred_sample': preds_np[i],
                    'seq_length_sample': seq_lengths_np[i]
                }
            if len(self.samples_train_synth.keys()) == len(self.character_set):
                break

    def log_train_synth(self, itr, epoch_float, batch, preds, loss, sub_losses):
        inputs, targets, seq_lengths, mask = batch

        for character, sample in self.samples_train_synth.items():
            target_sample = sample['target_sample']
            pred_sample = sample['pred_sample']
            seq_length_sample = sample['seq_length_sample']

            df_v_trgt = get_df_v_from_y_target(target_sample, seq_length_sample, self.dx_mean, self.dx_norm, self.dy_mean, self.dy_norm)
            df_dots_trgt = get_dot_seq_from_dx_dy(df_v_trgt, self.w_img, self.h_img, self.dt)
            gif_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-train-TF-{}-trgt.gif'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            png_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-train-TF-{}-trgt.png'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            np_images = rasterize_dev(df_dots_trgt, self.w_img, self.h_img, self.dt, self.stroke_width, gif_path, png_path)

            wandb.log({'train/image-gif-prediction-TF/image/target/character-{}'.format(character): wandb.Image(gif_path)}, step=itr)

            df_v = sample_output_seq(pred_sample, seq_length_sample, self.dx_norm, self.dy_norm, self.d2x_norm, self.d2y_norm, self.hp, use_eod=False)
            df_dots = get_dot_seq_from_dx_dy(df_v, self.w_img, self.h_img, self.dt)

            gif_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-train-TF-{}-synt.gif'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            png_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-train-TF-{}-synt.png'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            np_images = rasterize_dev(df_dots, self.w_img, self.h_img, self.dt, self.stroke_width, gif_path, png_path)

            wandb.log({'train/image-gif-prediction-TF/image/prediction/character-{}'.format(character): wandb.Image(gif_path)}, step=itr)

            dx_mae_target_gmm_mean, dy_mae_target_gmm_mean = get_target_gmm_mean_distance(target_sample, pred_sample, seq_length_sample, self.dx_mean, self.dx_norm, self.dy_mean, self.dy_norm, self.hp)
            dx_std_mean, dy_std_mean, corr_mean, corr_abs_mean, det_mean = get_gmm_mean_std(pred_sample, seq_length_sample, self.dx_norm, self.dy_norm, self.hp)

            dx_seq_trgt = df_v_trgt.dx.to_numpy()
            dy_seq_trgt = df_v_trgt.dy.to_numpy()
            dx_seq_pred = df_v.dx.to_numpy()
            dy_seq_pred = df_v.dy.to_numpy()
            mean_mae_dx = np.mean(np.abs(dx_seq_trgt - dx_seq_pred)) * self.dx_norm
            mean_mae_dy = np.mean(np.abs(dy_seq_trgt - dy_seq_pred)) * self.dy_norm

            wandb.log({
                'train/image-gif-prediction-TF/dx/mean-mae-dx-pred_sampled/character-{}'.format(character): mean_mae_dx,
                'train/image-gif-prediction-TF/dy/mean-mae-dy-pred_sampled/character-{}'.format(character): mean_mae_dy,
                'train/image-gif-prediction-TF/gmm/mean-mae-dx-pred_mean/character-{}'.format(character): dx_mae_target_gmm_mean,
                'train/image-gif-prediction-TF/gmm/mean-mae-dy-pred_mean/character-{}'.format(character): dy_mae_target_gmm_mean,
                'train/image-gif-prediction-TF/gmm/dx_std_mean/character-{}'.format(character): dx_std_mean,
                'train/image-gif-prediction-TF/gmm/dy_std_mean/character-{}'.format(character): dy_std_mean,
                'train/image-gif-prediction-TF/gmm/corr_mean/character-{}'.format(character): corr_mean,
                'train/image-gif-prediction-TF/gmm/corr_abs_mean/character-{}'.format(character): corr_abs_mean,
                'train/image-gif-prediction-TF/gmm/det_mean/character-{}'.format(character): det_mean,
            }, step=itr)

        dx_mean_mae, dy_mean_mae = get_sampled_dxdy_distance(targets, preds, seq_lengths, self.dx_norm, self.dy_norm, self.d2x_norm, self.d2y_norm, self.hp)
        wandb.log({
            'train/dx/dx_mean_mae':dx_mean_mae*self.dx_norm,
            'train/dy/dy_mean_mae':dy_mean_mae*self.dy_norm,
        }, step=itr)

    def log_val_init(self, iterations_val):
        self.iterations_val = iterations_val
        self.sum_loss_val = 0
        self.sum_loss_gmm_val = 0
        self.sum_loss_hov_val = 0
        self.sum_loss_eod_val = 0
        self.sum_loss_cr = 0
        self.sum_dx_mae_target_gmm_mean = 0
        self.sum_dy_mae_target_gmm_mean = 0
        self.sum_dx_std_mean = 0
        self.sum_dy_std_mean = 0
        self.sum_corr_mean = 0
        self.sum_corr_abs_mean = 0
        self.sum_det_mean = 0
        self.sum_hov_accuracy = 0
        self.sum_hov_recall = 0
        self.sum_hov_precision = 0
        self.sum_eod_accuracy = 0
        self.sum_eod_recall = 0
        self.sum_eod_precision = 0
        self.sum_mean_smooth_ratio = 0

    def log_val_eval(self, batch, preds, smoothing_ratios, loss, sub_losses, corrector_outputs):
        inputs, targets, seq_lengths, mask = batch
        loss_gmm, loss_hov, loss_eod, loss_smoothing = sub_losses
        cr_pred, cr_loss = corrector_outputs

        dx_mae_target_gmm_mean, dy_mae_target_gmm_mean = get_target_gmm_mean_distance(targets, preds, seq_lengths, self.dx_mean, self.dx_norm, self.dy_mean, self.dy_norm, self.hp)
        dx_std_mean, dy_std_mean, corr_mean, corr_abs_mean, det_mean = get_gmm_mean_std(preds, seq_lengths, self.dx_norm, self.dy_norm, self.hp)
        hov_accuracy, hov_recall, hov_precision = get_hov_accuracy(targets, preds, seq_lengths, self.hp)
        eod_accuracy, eod_recall, eod_precision = get_eod_accuracy(targets, preds, seq_lengths, self.hp)

        if smoothing_ratios.numel() > 1:
            smoothing_ratios = smoothing_ratios.squeeze(-1)
            s_mask = (torch.arange(smoothing_ratios.shape[1], device=smoothing_ratios.device)[None, :] < seq_lengths[:, None]).to(smoothing_ratios.dtype)
            mean_smooth_ratio = torch.sum(smoothing_ratios * s_mask) / torch.sum(seq_lengths).to(smoothing_ratios.dtype)
        else:
            mean_smooth_ratio = smoothing_ratios

        self.sum_loss_val += loss.item()
        self.sum_loss_gmm_val += loss_gmm.item() if hasattr(loss_gmm, 'item') else loss_gmm
        self.sum_loss_hov_val += loss_hov.item() if hasattr(loss_hov, 'item') else loss_hov
        self.sum_loss_eod_val += loss_eod.item() if hasattr(loss_eod, 'item') else loss_eod
        self.sum_loss_cr += cr_loss.item() if hasattr(cr_loss, 'item') else cr_loss

        self.sum_dx_mae_target_gmm_mean += dx_mae_target_gmm_mean
        self.sum_dy_mae_target_gmm_mean += dy_mae_target_gmm_mean

        self.sum_dx_std_mean += dx_std_mean
        self.sum_dy_std_mean += dy_std_mean
        self.sum_corr_mean += corr_mean
        self.sum_corr_abs_mean += corr_abs_mean
        self.sum_det_mean += det_mean

        self.sum_hov_accuracy += hov_accuracy
        self.sum_hov_recall += hov_recall
        self.sum_hov_precision += hov_precision

        self.sum_eod_accuracy += eod_accuracy
        self.sum_eod_recall += eod_recall
        self.sum_eod_precision += eod_precision

        self.sum_mean_smooth_ratio += mean_smooth_ratio.item() if hasattr(mean_smooth_ratio, 'item') else mean_smooth_ratio

        self.samples_val_synth = dict()
        batch_size = inputs.shape[0]
        inputs_np = inputs.detach().cpu().numpy() if isinstance(inputs, torch.Tensor) else inputs
        targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets
        preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        seq_lengths_np = seq_lengths.detach().cpu().numpy() if isinstance(seq_lengths, torch.Tensor) else seq_lengths

        for i in range(batch_size):
            character = get_character_from_vector(inputs_np[i, 0, -self.v_character_dim:], self.character_set)
            if character not in self.samples_val_synth.keys():
                self.samples_val_synth[character] = {
                    'target_sample': targets_np[i],
                    'pred_sample': preds_np[i],
                    'seq_length_sample': seq_lengths_np[i]
                }
            if len(self.samples_val_synth.keys()) == len(self.character_set):
                break

    def log_val(self, itr):
        wandb.log({
            'val/loss/total':self.sum_loss_val/self.iterations_val,
            'val/loss/gmm':self.sum_loss_gmm_val/self.iterations_val,
            'val/loss/hov':self.sum_loss_hov_val/self.iterations_val,
            'val/loss/eod':self.sum_loss_eod_val/self.iterations_val,
            'val/loss/corrector':self.sum_loss_cr/self.iterations_val,
            'val/gmm/dx_mae_target_gmm_mean':self.sum_dx_mae_target_gmm_mean/self.iterations_val,
            'val/gmm/dy_mae_target_gmm_mean':self.sum_dy_mae_target_gmm_mean/self.iterations_val,
            'val/gmm/dx_std_mean':self.sum_dx_std_mean/self.iterations_val,
            'val/gmm/dy_std_mean':self.sum_dy_std_mean/self.iterations_val,
            'val/gmm/corr_mean':self.sum_corr_mean/self.iterations_val,
            'val/gmm/corr_abs_mean':self.sum_corr_abs_mean/self.iterations_val,
            'val/hov/accuracy':self.sum_hov_accuracy/self.iterations_val,
            'val/hov/recall':self.sum_hov_recall/self.iterations_val,
            'val/hov/precision':self.sum_hov_precision/self.iterations_val,
            'val/eod/accuracy':self.sum_eod_accuracy/self.iterations_val,
            'val/eod/recall':self.sum_eod_recall/self.iterations_val,
            'val/eod/precision':self.sum_eod_precision/self.iterations_val,
            'val/smooth_ratio':self.sum_mean_smooth_ratio/self.iterations_val,
        }, step=itr)

    def log_val_synth(self, itr, epoch_float, batch, preds, loss, sub_losses, model):
        inputs, targets, seq_lengths, mask = batch

        for character, sample in self.samples_val_synth.items():
            target_sample = sample['target_sample']
            pred_sample = sample['pred_sample']
            seq_length_sample = sample['seq_length_sample']

            df_v_trgt = get_df_v_from_y_target(target_sample, seq_length_sample, self.dx_mean, self.dx_norm, self.dy_mean, self.dy_norm)
            df_dots_trgt = get_dot_seq_from_dx_dy(df_v_trgt, self.w_img, self.h_img, self.dt)
            gif_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-val-TF-{}-trgt.gif'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            png_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-val-TF-{}-trgt.png'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            np_images = rasterize_dev(df_dots_trgt, self.w_img, self.h_img, self.dt, self.stroke_width, gif_path, png_path)
            wandb.log({'val/image-gif-prediction-TF/image/target/character-{}'.format(character): wandb.Image(gif_path)}, step=itr)

            df_v = sample_output_seq(pred_sample, seq_length_sample, self.dx_norm, self.dy_norm, self.d2x_norm, self.d2y_norm, self.hp, use_eod=False)
            df_dots = get_dot_seq_from_dx_dy(df_v, self.w_img, self.h_img, self.dt)

            gif_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-val-TF-{}-synt.gif'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            png_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-val-TF-{}-synt.png'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            np_images = rasterize_dev(df_dots, self.w_img, self.h_img, self.dt, self.stroke_width, gif_path, png_path)
            wandb.log({'val/image-gif-prediction-TF/image/prediction/character-{}'.format(character): wandb.Image(gif_path)}, step=itr)

            dx_mae_target_gmm_mean, dy_mae_target_gmm_mean = get_target_gmm_mean_distance(target_sample, pred_sample, seq_length_sample, self.dx_mean, self.dx_norm, self.dy_mean, self.dy_norm, self.hp)
            dx_std_mean, dy_std_mean, corr_mean, corr_abs_mean, det_mean = get_gmm_mean_std(pred_sample, seq_length_sample, self.dx_norm, self.dy_norm, self.hp)

            dx_seq_trgt = df_v_trgt.dx.to_numpy()
            dy_seq_trgt = df_v_trgt.dy.to_numpy()
            dx_seq_pred = df_v.dx.to_numpy()
            dy_seq_pred = df_v.dy.to_numpy()
            mean_mae_dx = np.mean(np.abs(dx_seq_trgt - dx_seq_pred)) * self.dx_norm
            mean_mae_dy = np.mean(np.abs(dy_seq_trgt - dy_seq_pred)) * self.dy_norm
            wandb.log({
                'val/image-gif-prediction-TF/dx/mean-mae-dx-pred_sampled/character-{}'.format(character): mean_mae_dx,
                'val/image-gif-prediction-TF/dy/mean-mae-dy-pred_sampled/character-{}'.format(character): mean_mae_dy,
                'val/image-gif-prediction-TF/gmm/mean-mae-dx-pred_mean/character-{}'.format(character): dx_mae_target_gmm_mean,
                'val/image-gif-prediction-TF/gmm/mean-mae-dy-pred_mean/character-{}'.format(character): dy_mae_target_gmm_mean,
                'val/image-gif-prediction-TF/gmm/dx_std_mean/character-{}'.format(character): dx_std_mean,
                'val/image-gif-prediction-TF/gmm/dy_std_mean/character-{}'.format(character): dy_std_mean,
                'val/image-gif-prediction-TF/gmm/corr_mean/character-{}'.format(character): corr_mean,
                'val/image-gif-prediction-TF/gmm/corr_abs_mean/character-{}'.format(character): corr_abs_mean,
                'val/image-gif-prediction-TF/gmm/det_mean/character-{}'.format(character): det_mean,
            }, step=itr)

        dx_mean_mae, dy_mean_mae = get_sampled_dxdy_distance(targets, preds, seq_lengths, self.dx_norm, self.dy_norm, self.d2x_norm, self.d2y_norm, self.hp)
        wandb.log({
            'val/dx/dx_mean_mae':dx_mean_mae*self.dx_norm,
            'val/dy/dy_mean_mae':dy_mean_mae*self.dy_norm,
        }, step=itr)

    def log_test_synth(self, itr, epoch_float, model, test_set):
        inputs_test, targets_test, seq_lengths_test, characters_test = test_set
        stat_test_data_character, stat_test_data_all = get_stat_data_from_test_set(self.character_set, test_set, self.dx_norm, self.dy_norm)
        metrics = list(stat_test_data_all.keys())
        
        tf_sampling = self.hp['test_tf_sampling']
        batch_size = self.hp['batch_size_test']

        outputs_total = list()
        characters_total = list()

        dx_std_means = list()
        dy_std_means = list()
        corr_means = list()
        corr_abs_means = list()
        det_means = list()

        d_stats = {metric:list() for metric in metrics}
        d_means = {metric:list() for metric in metrics}
        error_means = {metric:list() for metric in metrics}
        self.pvalues = {metric:list() for metric in metrics}
        self.pvalues_mean_diff = {metric:list() for metric in metrics}

        stat_gen_data_character = dict()

        p_samples_all = {metric:list() for metric in metrics}
        q_samples_all = {metric:list() for metric in metrics}
        self.pvalue_all = dict()

        print('[TEST] Generating test handwriting samples...')
        for character in self.character_set:
            # 모델 인퍼런스 호출 (PyTorch)
            returns = model.infer_batch(character, batch_size, eod_stop=False, eod_slice=False, bias=0, tf_sampling=tf_sampling, return_preds=True, use_corrector=self.hp['use_corrector'])
            
            # 반환값이 PyTorch 텐서일 경우 NumPy로 변환
            outputs = returns[0].detach().cpu().numpy() if isinstance(returns[0], torch.Tensor) else returns[0]
            preds = returns[1].detach().cpu().numpy() if isinstance(returns[1], torch.Tensor) else returns[1]

            stat_outputs = get_stat_data_from_outputs(outputs, self.dx_norm, self.dy_norm)
            stat_gen_data_character[character] = stat_outputs

            seq_lengths = [get_seq_length(outputs[i,:,3]) for i in range(outputs.shape[0])]
            outputs_total += [outputs[i,:seq_lengths[i],:] for i in range(outputs.shape[0])]
            characters_total += [character] * outputs.shape[0]

            dx_std_mean, dy_std_mean, corr_mean, corr_abs_mean, det_mean = get_gmm_mean_std(preds, seq_lengths, self.dx_norm, self.dy_norm, self.hp)
            dx_std_means.append(dx_std_mean); dy_std_means.append(dy_std_mean)
            corr_means.append(corr_mean); corr_abs_means.append(corr_abs_mean)
            det_means.append(det_mean)

            i_sample = 0
            dx_seq = outputs[i_sample,:,0]
            dy_seq = outputs[i_sample,:,1]
            hov_seq = outputs[i_sample,:,2]
            eod_seq = outputs[i_sample,:,3]

            df_v_synt = pd.DataFrame({'dx':dx_seq*self.dx_norm + self.dx_mean, 'dy':dy_seq*self.dy_norm + self.dy_mean, 'hover':hov_seq, 'eod':eod_seq})
            df_v_synt_eod = cut_by_eod(df_v_synt)

            df_dots_synt_eod = get_dot_seq_from_dx_dy(df_v_synt_eod, self.w_img, self.h_img, self.dt)
            gif_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-val-SF-{}.gif'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            png_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-val-SF-{}.png'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            np_images = rasterize_dev(df_dots_synt_eod, self.w_img, self.h_img, self.dt, self.stroke_width, gif_path, png_path, eod_stop=True)
            wandb.log({'test/image-gif-prediction-SF/eod/prediction-{}'.format(character): wandb.Image(gif_path)}, step=itr)

            df_dots_synt = get_dot_seq_from_dx_dy(df_v_synt, self.w_img, self.h_img, self.dt)
            gif_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-val-SF-{}-no_eod.gif'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            png_path = os.path.join(self.dir_generation_path, 'epoch-{:0{}d}-it-{:0{}d}-val-SF-{}-no_eod.png'.format(int(epoch_float), int(np.log10(self.n_epochs))+1, itr, int(np.log10(self.iters_total))+1, character))
            np_images = rasterize_dev(df_dots_synt, self.w_img, self.h_img, self.dt, self.stroke_width, gif_path, png_path, eod_stop=False)
            wandb.log({'test/image-gif-prediction-SF/no_eod/prediction-{}'.format(character): wandb.Image(gif_path)}, step=itr)

            wandb.log({
                'test/dx_std_mean/character-{}'.format(character): dx_std_mean,
                'test/dy_std_mean/character-{}'.format(character): dy_std_mean,
                'test/corr_mean/character-{}'.format(character): corr_mean,
                'test/corr_abs_mean/character-{}'.format(character): corr_abs_mean,
                'test/det_mean/character-{}'.format(character): det_mean,
            }, step=itr)

            for metric in metrics:
                rv_range = (stat_test_data_all[metric].min(), stat_test_data_all[metric].max())
                p_samples = stat_test_data_character[character][metric]
                q_samples = stat_outputs[metric]
                p_samples = rm_outliers(p_samples)
                q_samples = rm_outliers(q_samples)
                p_samples_all[metric].append(p_samples)
                q_samples_all[metric].append(q_samples)
                p_stats = stats.describe(p_samples)
                q_stats = stats.describe(q_samples)
                
                d_stat = compute_stat_distance(p_samples, q_samples, bins=100, range=rv_range)
                d_stats[metric].append(d_stat)
                d_mean = np.absolute(q_stats.mean - p_stats.mean)
                d_means[metric].append(d_mean)
                error_mean = q_stats.mean - p_stats.mean
                error_means[metric].append(error_mean)
                pvalue = stats.mannwhitneyu(p_samples, q_samples).pvalue
                pvalue_mean_diff = stats.ttest_ind(p_samples, q_samples, equal_var=False).pvalue
                self.pvalues[metric].append(pvalue)
                self.pvalues_mean_diff[metric].append(pvalue_mean_diff)
                wandb.log({
                    'test/{}/stat_distance/character-{}'.format(metric, character):d_stat,
                    'test/{}/mean_distance/character-{}'.format(metric, character):d_mean,
                    'test/{}/mean_error/character-{}'.format(metric, character):error_mean,
                    'test/{}/p-value/character-{}'.format(metric, character):pvalue,
                    'test/{}/p-value-mean_diff/character-{}'.format(metric, character):pvalue_mean_diff,
                    'test/{}/range/character-{}-target.max'.format(metric, character):p_samples.max(),
                    'test/{}/range/character-{}-target.mean'.format(metric, character):p_samples.mean(),
                    'test/{}/range/character-{}-target.min'.format(metric, character):p_samples.min(),
                    'test/{}/range/character-{}-pred.max'.format(metric, character):q_samples.max(),
                    'test/{}/range/character-{}-pred.mean'.format(metric, character):q_samples.mean(),
                    'test/{}/range/character-{}-pred.min'.format(metric, character):q_samples.min(),
                }, step=itr)

        wandb.log({
            'test/variability/dx_std_mean/total': np.mean(dx_std_means),
            'test/variability/dy_std_mean/total': np.mean(dy_std_means),
            'test/variability/corr_means/total': np.mean(corr_means),
            'test/variability/corr_abs_means/total': np.mean(corr_abs_means),
            'test/variability/det_means/total': np.mean(det_means),
        }, step=itr)

        for metric in metrics:
            p_samples_all[metric] = np.concatenate(p_samples_all[metric])
            q_samples_all[metric] = np.concatenate(q_samples_all[metric])

        for metric in metrics:
            wandb.log({
                'test/{}/stat_distance/total_mean'.format(metric):np.mean(d_stats[metric]),
                'test/{}/mean_distance/total_mean'.format(metric):np.mean(d_means[metric]),
                'test/{}/mean_error/total_mean'.format(metric):np.mean(error_means[metric]),
                'test/{}/p-value/total_mean'.format(metric):np.mean(self.pvalues[metric]),
                'test/{}/p-value-mean_diff/total_mean'.format(metric):np.mean(self.pvalues_mean_diff[metric]),
            }, step=itr)

        for metric in metrics:
            rv_range = (stat_test_data_all[metric].min(), stat_test_data_all[metric].max())
            p_samples = p_samples_all[metric]
            q_samples = q_samples_all[metric]
            p_stats = stats.describe(p_samples)
            q_stats = stats.describe(q_samples)
            
            d_stat = compute_stat_distance(p_samples, q_samples, bins=100, range=rv_range)
            d_mean = np.absolute(q_stats.mean - p_stats.mean)
            error_mean = q_stats.mean - p_stats.mean
            pvalue = stats.mannwhitneyu(p_samples, q_samples).pvalue
            self.pvalue_all[metric] = pvalue
            pvalue_mean_diff = stats.ttest_ind(p_samples, q_samples, equal_var=False).pvalue
            wandb.log({
                'test/{}/stat_distance/total'.format(metric):d_stat,
                'test/{}/mean_distance/total'.format(metric):d_mean,
                'test/{}/mean_error/total'.format(metric):error_mean,
                'test/{}/p-value/total'.format(metric):pvalue,
                'test/{}/p-value-mean_diff/total'.format(metric):pvalue_mean_diff,
                'test/{}/range/total-target.max'.format(metric):p_samples.max(),
                'test/{}/range/total-target.mean'.format(metric):p_samples.mean(),
                'test/{}/range/total-target.min'.format(metric):p_samples.min(),
                'test/{}/range/total-pred.max'.format(metric):q_samples.max(),
                'test/{}/range/total-pred.mean'.format(metric):q_samples.mean(),
                'test/{}/range/total-pred.min'.format(metric):q_samples.min(),
            }, step=itr)

        wandb.log({
            'test/behavioral_plausibility':int(self.behavioral_plausibility_strict()),
            'test/behavioral_plausibility_strict':int(self.behavioral_plausibility_strict()),
        }, step=itr)

        pdf_paths, png_paths = plot_behavior_distributions(
            self.character_set,
            stat_test_data_character, stat_gen_data_character,
            itr=itr, logdir=self.dir_distribution_path, dt=self.hp['dt'])
        for feature, png_path in png_paths.items():
            wandb.log({
                'test/behavioral_distribution/{}'.format(feature):wandb.Image(png_path),
            }, step=itr)

        outputs_total = np.asarray(outputs_total, dtype=object)
        characters_total = np.asarray(characters_total)

        print('\n' + '='*50)
        print(f'[TEST] DTW classification (Step: {itr})')
        print('='*50)
        self.accuracy_total_dtw, accuracy_character = self.dtw_classifier.classify(outputs_total, characters_total)
        print(f"[Prediction] DTW Total Accuracy: {self.accuracy_total_dtw:.4f}")
        wandb.log({'test/dtw-classification/prediction-SF/accuracy':self.accuracy_total_dtw}, step=itr)
        for character, accuracy in accuracy_character.items():
            wandb.log({'test/dtw-classification/prediction-SF/accuracy-character-{}'.format(character):accuracy}, step=itr)
            #print()

        if self.accuracy_total_dtw_test is None:
            # 타겟 데이터(사람 데이터)가 PyTorch 텐서일 경우를 대비해 변환
            targets_test_np = targets_test.detach().cpu().numpy() if isinstance(targets_test, torch.Tensor) else targets_test
            self.accuracy_total_dtw_test, self.accuracy_character = self.dtw_classifier.classify(targets_test_np, characters_test)
            print(f"[Test Set Target] DTW Total Accuracy: {self.accuracy_total_dtw_test:.4f}")
        wandb.log({'test/dtw-classification/test_set/accuracy':self.accuracy_total_dtw_test}, step=itr)
        for character, accuracy in self.accuracy_character.items():
            wandb.log({'test/dtw-classification/test_set/accuracy-character-{}'.format(character):accuracy}, step=itr)

        accuracy_gap = self.accuracy_total_dtw_test - self.accuracy_total_dtw
        print(f"[DTW Gap] (Test - Pred): {accuracy_gap:.4f}\n")
        wandb.log({'test/dtw-classification/prediction-SF/accuracy_gap/total':accuracy_gap}, step=itr)
        for character in accuracy_character.keys():
            gap = self.accuracy_character[character] - accuracy_character[character]
            wandb.log({'test/dtw-classification/prediction-SF/accuracy_gap/character-{}'.format(character):gap}, step=itr)

        print('='*50)
        print(f'[TEST] CNN classification (Step: {itr})')
        print('='*50)
        self.accuracy_total_cnn, precisions, recalls, f1scores = self.cnn_classifier.classify(outputs_total, characters_total)
        
        print(f"[Prediction] CNN Total Accuracy: {self.accuracy_total_cnn:.4f}")
        wandb.log({'test/cnn-classification/prediction-SF/accuracy':self.accuracy_total_cnn}, step=itr)
        
        for character in precisions.keys():
            wandb.log({
                'test/cnn-classification/prediction-SF/precision-character-{}'.format(character):precisions[character],
                'test/cnn-classification/prediction-SF/recall-character-{}'.format(character):recalls[character],
                'test/cnn-classification/prediction-SF/f1score-character-{}'.format(character):f1scores[character],
            }, step=itr)

        if self.accuracy_total_cnn_test is None:
            targets_test_np = targets_test.detach().cpu().numpy() if isinstance(targets_test, torch.Tensor) else targets_test
            self.accuracy_total_cnn_test, self.precisions_cnn_test, self.recalls_cnn_test, self.f1scores_cnn_test = self.cnn_classifier.classify(targets_test_np, characters_test)
            print(f"[Test Set Target] CNN Total Accuracy: {self.accuracy_total_cnn_test:.4f}")

        cnn_accuracy_gap = self.accuracy_total_cnn_test - self.accuracy_total_cnn
        # [추가] 터미널 출력
        print(f"[CNN Gap] (Test - Pred): {cnn_accuracy_gap:.4f}\n")

        wandb.log({
            'test/ccn-classification/test_set/accuracy':self.accuracy_total_cnn_test,
            'test/ccn-classification/prediction-SF/gap/accuracy':(self.accuracy_total_cnn_test - self.accuracy_total_cnn),
        }, step=itr)

        for character in self.precisions_cnn_test.keys():
            wandb.log({
                'test/cnn-classification/test_set/precision-character-{}'.format(character):self.precisions_cnn_test[character],
                'test/ccn-classification/prediction-SF/gap/precision-character-{}'.format(character):(self.precisions_cnn_test[character] - precisions[character]),
                'test/cnn-classification/test_set/recall-character-{}'.format(character):self.recalls_cnn_test[character],
                'test/ccn-classification/prediction-SF/gap/recall-character-{}'.format(character):(self.recalls_cnn_test[character] - recalls[character]),
                'test/cnn-classification/test_set/f1score-character-{}'.format(character):self.f1scores_cnn_test[character],
                'test/ccn-classification/prediction-SF/gap/f1score-character-{}'.format(character):(self.f1scores_cnn_test[character] - f1scores[character]),
            }, step=itr)

        self.accuracy_total = np.mean([self.accuracy_total_dtw, self.accuracy_total_cnn])
        accuracy_total_test = np.mean([self.accuracy_total_dtw_test, self.accuracy_total_cnn_test])
        self.accuracy_total_max = max(self.accuracy_total_max, self.accuracy_total)
        
        # [추가] 최종 요약 성적표 출력
        is_best = self.accuracy_total >= self.accuracy_total_max
        is_human = self.accuracy_total >= accuracy_total_test
        
        print('='*50)
        print(f'[TEST] Final Performance Summary (Step: {itr})')
        print('='*50)
        print(f"Current Total Accuracy (DTW+CNN mean): {self.accuracy_total:.4f}")
        print(f"Max Total Accuracy So Far: {self.accuracy_total_max:.4f}")
        print(f"Is Best Model?: {'Yes!' if is_best else 'No'}")
        print(f"Reached Human Level?: {'Yes!' if is_human else 'No'}")
        print('='*50 + '\n')

        wandb.log({
                'test/best-performance/accuracy':self.accuracy_total,
                'test/best-performance/accuracy_max':self.accuracy_total_max,
                'test/best-performance/is_best':1 if self.accuracy_total >= self.accuracy_total_max else 0,
                'test/best-performance/is_human_level':1 if self.accuracy_total >= accuracy_total_test else 0,
        }, step=itr)