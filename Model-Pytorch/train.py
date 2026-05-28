import os
import argparse
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from glob import glob

from utils import get_batch_iterations
from data_utils import import_hparams, BatchGenerator, DataLoader
from loss import Loss
from model import Model
from logger import Logger

os.environ["WANDB_INSECURE_DISABLE_SSL"] = "true"

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # CuDNN 결정론적 연산 보장 (재현성을 위해)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu_util(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    # PyTorch는 GPU 메모리를 필요한 만큼 동적으로 할당하므로, TF의 set_memory_growth와 같은 별도의 설정이 필요하지 않음

def train(hparams, logger, checkpoint):
    # 사용할 디바이스 설정 (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    data_loader = DataLoader(hparams); dl = data_loader
    train_set = inputs_train, targets_train, seq_lengths_train, characters_train = data_loader.load_dataset('train')
    val_set = inputs_val, targets_val, seq_lengths_val, characters_val = data_loader.load_dataset('val')
    test_set = inputs_test, targets_test, seq_lengths_test, characters_test = data_loader.load_dataset('test')
    print('#Train =', inputs_train.shape[0]) # 8000
    print('#Val   =', inputs_val.shape[0]) # 1000
    print('#Test  =', inputs_test.shape[0]) # 1000

    logger.set_dataset_size(len(inputs_train), len(inputs_val))

    generator = BatchGenerator(hparams)
    loss_fn = Loss(hparams)
    
    # Model 초기화 (PyTorch 기반 Model 래퍼 클래스로 가정)
    model = Model(hparams, dl).to(device)

    print('Before scaling, (dx_mean, dy_mean, dx_std, dy_std)', (dl.dx_mean, dl.dy_mean, dl.dx_std, dl.dy_std))
    print('Before scaling, (d2x_mean, d2y_mean, d2x_std, d2y_std)', (dl.d2x_mean, dl.d2y_mean, dl.d2x_std, dl.d2y_std))
    logger.set_norms(dl.dx_norm, dl.dy_norm, dl.d2x_norm, dl.d2y_norm)
    dx_seqs = np.concatenate([target[:,0] for target in targets_train])
    dy_seqs = np.concatenate([target[:,1] for target in targets_train])
    dx_mean = np.mean(dx_seqs); dy_mean = np.mean(dy_seqs); dx_std = np.std(dx_seqs); dy_std = np.std(dy_seqs)
    print('After scaling, targets (x_mean, y_mean, x_std, y_std)', (dx_mean, dy_mean, dx_std, dy_std))

    logger.init_dtw_classifier(dl.transform_set_a2v(val_set)) ## error
    logger.init_cnn_classifier(dl)
    logger.compute_stat_between_human_sets(dl.transform_set_a2v(train_set), dl.transform_set_a2v(val_set), dl.transform_set_a2v(test_set))

    # TRAINING LOOP ============================================================
    itr = 0
    epoch_first = 0
    n_epochs = hparams['n_epochs']
    batch_size = hparams['batch_size']
    period_to_save_model = hparams['period_to_save_model']
    train_synth_period = hparams['train_synth_period']
    val_period_epoch = hparams['val_period_epoch']
    val_period_itr = hparams['val_period_itr']
    test_period_epoch = hparams['test_period_epoch']
    test_period_itr = hparams['test_period_itr']

    if checkpoint != '':
        model.load_weights(checkpoint)
        # PyTorch에서는 .numpy() 대신 .item()을 사용하거나 Python native type을 바로 사용합니다.
        itr = int(model.iterations) if hasattr(model.iterations, '__int__') else model.iterations.item()
        epoch_first = (int(model.epochs) if hasattr(model.epochs, '__int__') else model.epochs.item()) + 1
        logger.set_norms(dl.dx_norm, dl.dy_norm, dl.d2x_norm, dl.d2y_norm)

    for epoch in range(epoch_first, n_epochs):
        iterations_per_epoch = get_batch_iterations(len(inputs_train), batch_size)
        
        for step, batch_train in tqdm(enumerate(generator(inputs_train, targets_train)), desc=f'[Train] Iterations (epoch {epoch})', initial=0, total=iterations_per_epoch):
            itr += 1
            epoch_float = itr / iterations_per_epoch
            
            # 텐서 파싱 및 디바이스 이동
            inputs, targets, seq_lengths, mask = batch_train
            print('inputs:', inputs.shape) # inputs: torch.Size([128, *, 14])
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            
            # Forward + Backward + Optimizer step (model.train_step 내부에서 처리된다고 가정)
            pred_train, states_train, h_seqs, smoothing_ratios, loss_train, sub_losses_train = model.train_step(inputs, targets, mask)
            print('pred_train:', pred_train.shape) # pred_train: torch.Size([128, *, 104])
            # Corrector 준비 및 훈련
            cr_inputs_train, cr_targets_train, cr_mask_train = dl.generate_dataset_for_corrector(pred_train, states_train, h_seqs, inputs, targets, mask)
            print('cr_inputs_train:', cr_inputs_train.shape) # cr_inputs_train: torch.Size([128, *, 1039])
            cr_inputs_train, cr_targets_train, cr_mask_train = cr_inputs_train.to(device), cr_targets_train.to(device), cr_mask_train.to(device)
            cr_pred_train, cr_loss_train = model.corrector.train_step(cr_inputs_train, cr_targets_train, cr_mask_train)
            
            logger.log_train(model, itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, smoothing_ratios, loss_train, sub_losses_train, (cr_pred_train, cr_loss_train))

            # VALIDATION (based on iteration) ====================
            if ((itr-1) % val_period_itr) == (val_period_itr-1):
                iterations_val = get_batch_iterations(len(inputs_val), batch_size)
                logger.log_val_init(iterations_val)
                for step, batch_val in tqdm(enumerate(generator(inputs_val, targets_val)), desc=f'[Val] Iterations (epoch {epoch})', initial=0, total=iterations_val):
                    inputs, targets, seq_lengths, mask = batch_val
                    inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                    
                    # Note: 원본 코드에서 test_step 반환값 개수가 달랐던 잠재적 버그를 6개로 통일하여 명확히 잡았습니다.
                    pred_val, states_val, h_seqs_val, smoothing_ratios_val, loss_val, sub_losses_val = model.test_step(inputs, targets, mask)
                    
                    cr_inputs_val, cr_targets_val, cr_mask_val = dl.generate_dataset_for_corrector(pred_val, states_val, h_seqs_val, inputs, targets, mask)
                    cr_inputs_val, cr_targets_val, cr_mask_val = cr_inputs_val.to(device), cr_targets_val.to(device), cr_mask_val.to(device)
                    cr_pred_val, cr_loss_val = model.corrector.test_step(cr_inputs_val, cr_targets_val, cr_mask_val)
                    
                    logger.log_val_eval(dl.transform_set_a2v(batch_val), pred_val, smoothing_ratios_val, loss_val, sub_losses_val, (cr_pred_val, cr_loss_val))
                logger.log_val(itr)

            # TEST (based on iteration) ====================
            if ((itr-1) % test_period_itr) == (test_period_itr-1):
                logger.log_test_synth(itr, epoch_float, model, dl.transform_set_a2v(test_set))
                if logger.save_model_now():
                    logger.save_model(model, epoch_float, itr)

            if itr == hparams['stop_itr']:
                exit()

        if (epoch+1) % train_synth_period == 0:
            logger.log_train_synth(itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, loss_train, sub_losses_train)

        # VALIDATION (based on epoch) ====================
        if ((epoch+1) % val_period_epoch) == 0:
            iterations_val = get_batch_iterations(len(inputs_val), batch_size)
            logger.log_val_init(iterations_val)
            for step, batch_val in tqdm(enumerate(generator(inputs_val, targets_val)), desc=f'[Val] Iterations (epoch {epoch})', initial=0, total=iterations_val):
                inputs, targets, seq_lengths, mask = batch_val
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                
                pred_val, states_val, h_seqs_val, smoothing_ratios, loss_val, sub_losses_val = model.test_step(inputs, targets, mask)
                
                cr_inputs_val, cr_targets_val, cr_mask_val = dl.generate_dataset_for_corrector(pred_val, states_val, h_seqs_val, inputs, targets, mask)
                cr_inputs_val, cr_targets_val, cr_mask_val = cr_inputs_val.to(device), cr_targets_val.to(device), cr_mask_val.to(device)
                cr_pred_val, cr_loss_val = model.corrector.test_step(cr_inputs_val, cr_targets_val, cr_mask_val)
                
                logger.log_val_eval(dl.transform_set_a2v(batch_val), pred_val, smoothing_ratios, loss_val, sub_losses_val, (cr_pred_val, cr_loss_val))
            logger.log_val(itr)

            # Generate random validation image
            logger.log_val_synth(itr, epoch_float, dl.transform_set_a2v(batch_val), pred_val, loss_val, sub_losses_val, model)

        if (epoch+1) % test_period_epoch == 0:
            logger.log_test_synth(itr, epoch_float, model, dl.transform_set_a2v(test_set))
            if logger.save_model_now():
                logger.save_model(model, epoch_float, itr)
                
        if epoch % period_to_save_model == (period_to_save_model - 1):
            logger.save_model(model, epoch_float, itr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a recurrent neural network to generate single handwritten characters.')
    parser.add_argument('-p', '--project', type=str, required=True, help='Specify a project name that groups running sessions.')
    parser.add_argument('-r', '--run', type=str, default='run', help='Specify a run name that identifies the current running execution.')
    parser.add_argument('--run_resume', type=str, default='', help='Specify the WandB run ID where you want to resume to log.')
    parser.add_argument('--checkpoint', type=str, default='', help='Specify a checkpoint path where the model import model parameters.')
    parser.add_argument('--hparams', type=str, default='hparams.yaml', help='Specify an YAML path that contains hyperparameters of this training.')
    parser.add_argument('--off_wandb', action='store_true', help='Turn off WandB logging.')
    parser.add_argument('--debug', action='store_true', help='Turn on the test mode where the size of splitted datasets is the batch size.')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPUs instead of CPUs')
    parser.add_argument('-g', '--gpus', type=str, default='0', help='Specify GPU numbers, e.g., "1" for a single GPU and "0,2" for mulitple GPUs.')

    args = parser.parse_args()

    hparams = import_hparams(args.hparams, v_type='cartesian')

    str_datetime = datetime.now().strftime("%y%m%d%H%M%S")
    run_id = f'{args.run}-{str_datetime}'
    print('- Project name:', args.project)
    print('- Run name:', run_id)
    print(f"- Target motor command: {hparams['target']}")
    print(f"- Input feeback: {hparams['input_kine']}")
    if args.run_resume != '':
        run_id = args.run_resume

    set_random_seed(hparams['seed'])

    if args.use_gpu:
        set_gpu_util(args.gpus)

    logger = Logger(args.project, run_id, hparams, args.off_wandb)
    for path in [args.hparams] + glob('*.py'):
        logger.archive_file(path)

    if args.debug:
        hparams['train_used_ratio'] = hparams['batch_size']
        hparams['val_used_ratio'] = hparams['batch_size_eval']
        hparams['test_used_ratio'] = hparams['batch_size_eval']
        hparams['batch_size_test'] = 1

    print('hparam and python files are saved!')

    train(hparams, logger, args.checkpoint) ## error