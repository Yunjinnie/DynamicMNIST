import os
import argparse
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from glob import glob

from utils import get_batch_iterations
from data_utils_transfer_revised import import_hparams, BatchGenerator, DataLoader
from loss import Loss
from model_transfer import Model
from logger_transfer import Logger

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
    
    # YAML에서 현재 스테이지 값을 가져옵니다 (예: 1, 2, 3)
    stage = hparams['train_mode']['stage']
    dataset = hparams['train_mode']['dataset']

    # Load the dataset
    data_loader = DataLoader(hparams); dl = data_loader

    # 1. Digit(숫자) 데이터셋 - '전체 세트'와 '개별 변수'를 동시에 확보
    train_set_digit = inputs_train_digit, targets_train_digit, seq_lengths_train_digit, characters_train_digit = dl.load_dataset('train', dataset_type='digit')
    val_set_digit = inputs_val_digit, targets_val_digit, seq_lengths_val_digit, characters_val_digit = dl.load_dataset('val', dataset_type='digit')
    test_set_digit = inputs_test_digit, targets_test_digit, seq_lengths_test_digit, characters_test_digit = dl.load_dataset('test', dataset_type='digit')

    # 2. Alphabet(알파벳) 데이터셋
    # inputs_train_alpha, targets_train_alpha, _, _ = dl.load_dataset('train', dataset_type='char')
    train_set_alpha = inputs_train_alpha, targets_train_alpha, seq_lengths_train_alpha, characters_alpha_train = dl.load_dataset('train', dataset_type='char')
    val_set_alpha = inputs_val_alpha, targets_val_alpha, seq_lengths_val_alpha, characters_alpha_val = dl.load_dataset('val', dataset_type='char')
    test_set_alpha = inputs_test_alpha, targets_test_alpha, seq_lengths_test_alpha, characters_alpha_test = dl.load_dataset('test', dataset_type='char')

    # Stage 1, 3은 숫자를 사용하고, Stage 2는 알파벳을 사용.
    if stage == 1 or stage == 3:
        print(">>> [Data Loader] Loading DIGIT dataset...")
        train_set = train_set_digit  # 이제 4개 요소가 모두 담긴 튜플이 할당됨  #inputs_train_digit
        val_set = val_set_digit
        test_set = test_set_digit

        # [중요] 로거의 캐릭터 셋을 숫자(0-9)로 한정
        # dl.character_set_digit 처럼 로더 내부에 정의된 숫자 리스트를 할당
        logger.character_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    elif stage == 2:
        print(">>> [Data Loader] Loading Character dataset...")
        #inputs_train, targets_train = dl.load_data(data_path=hp.data.alphabet_path)
        train_set = train_set_alpha
        val_set = val_set_alpha
        test_set = test_set_alpha

        # [중요] 로거의 캐릭터 셋을 알파벳(a-Z)으로 한정
        logger.character_set = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # train_set = inputs_train, targets_train, seq_lengths_train, characters_train = data_loader.load_dataset('train')
    # val_set = inputs_val, targets_val, seq_lengths_val, characters_val = data_loader.load_dataset('val')
    # test_set = inputs_test, targets_test, seq_lengths_test, characters_test = data_loader.load_dataset('test')

    # 3. 기존 코드 호환성을 위한 Unpacking (변수 풀어주기)

    # 현재 train_set에 몇 개가 들어있는지 확인
    print(f"DEBUG: train_set length is {len(train_set)}")
    # 첫 번째 요소의 타입 확인 (보통 Tensor나 Array)
    print(f"DEBUG: type of first element is {type(train_set[0])}")

    #inputs_train, targets_train, seq_lengths_train, characters_train, *extra = train_set
    #inputs_val, targets_val, seq_lengths_val, characters_val, *extra_val = val_set
    #inputs_test, targets_test, seq_lengths_test, characters_test, *extra_test = test_set

    # 3. Unpacking (모델 학습에 필요한 개별 변수 추출)
    # 이제 train_set은 항상 (Input, Target, Length, Char)의 4개 요소를 갖게 됨
    inputs_train, targets_train, seq_lengths_train, characters_train = train_set
    inputs_val, targets_val, seq_lengths_val, characters_val = val_set
    inputs_test, targets_test, seq_lengths_test, characters_test = test_set

    # Stage 1일 때
    print(f"Stage 1 Character Set ({len(logger.character_set)}개): {logger.character_set}")
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 딱 10개만 나와야 정상

    # 입력 벡터(원-핫 인코딩)의 차원 확인
    print(f"Input characters shape: {characters_train.shape}")
    # Stage 1에서는 (Batch, 10) 형태여야 함 (Batch, 62)라면 여기서 꼬인 것...

    print('#Train =', inputs_train.shape[0]) # 8000
    print('#Val   =', inputs_val.shape[0]) # 1000
    print('#Test  =', inputs_test.shape[0]) # 1000

    logger.set_dataset_size(len(inputs_train), len(inputs_val))

    generator = BatchGenerator(hparams)
    loss_fn = Loss(hparams)
    
    # Model 초기화 (PyTorch 기반 Model 래퍼 클래스로 가정)
    model = Model(hparams, dl).to(device)

    # 로거의 캐릭터 셋을 현재 로더가 부른 데이터(숫자면 숫자만)로 동기화
    logger.character_set = dl.character_set

    print('Before scaling, (dx_mean, dy_mean, dx_std, dy_std)', (dl.dx_mean, dl.dy_mean, dl.dx_std, dl.dy_std))
    print('Before scaling, (d2x_mean, d2y_mean, d2x_std, d2y_std)', (dl.d2x_mean, dl.d2y_mean, dl.d2x_std, dl.d2y_std))
    logger.set_norms(dl.dx_norm, dl.dy_norm, dl.d2x_norm, dl.d2y_norm)

    # targets_train의 첫 번째 요소 형태 확인
    print(f"DEBUG: targets_train[0] type: {type(targets_train[0])}")
    print(f"DEBUG: targets_train[0] shape: {targets_train[0].shape}")

    ####### Error
    #dx_seqs = np.concatenate([target[:,0] for target in targets_train])
    #dy_seqs = np.concatenate([target[:,1] for target in targets_train])
    # 수정 후: target이 2차원이면 0번 열을 쓰고, 1차원이면 그대로 사용
    dx_seqs = np.concatenate([target[:, 0] if target.ndim > 1 else target for target in targets_train])
    dy_seqs = np.concatenate([target[:, 1] if target.ndim > 1 else target for target in targets_train]) # dy도 마찬가지

    dx_mean = np.mean(dx_seqs); dy_mean = np.mean(dy_seqs); dx_std = np.std(dx_seqs); dy_std = np.std(dy_seqs)
    print('After scaling, targets (x_mean, y_mean, x_std, y_std)', (dx_mean, dy_mean, dx_std, dy_std))

    print(f"DEBUG: Type of val_set: {type(val_set)}")
    print(f"DEBUG: Length of val_set: {len(val_set)}") 
    # 여기서 4가 나와야 정상. 200이 나오면 잘못된 변수를 넘긴 것

    # 이제 4개짜리 튜플이 정상적으로 넘어가므로 에러가 발생하지 않음
    logger.init_dtw_classifier(dl.transform_set_a2v(train_set)) # val_set ## error ## error again
    logger.init_cnn_classifier(dl)
    logger.compute_stat_between_human_sets(dl.transform_set_a2v(train_set), dl.transform_set_a2v(val_set), dl.transform_set_a2v(test_set))

    
    print("="*50)
    print(f"DEBUG: Data Loader Character Set Length: {len(dl.character_set)}")
    print(f"DEBUG: Model Input Dimension Expected: {hparams['input_dim']}")
    # 전체 Train 데이터의 첫 번째 샘플(인덱스 0)의 형태를 확인
    print(f"DEBUG: Input Sequence Shape (1st sample): {inputs_train[0].shape}")
    #print(f"DEBUG: Input Vector Shape from Batch: {inputs.shape}")
    print("="*50)

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
    
    ###### stage 별 분기
    
    for epoch in range(epoch_first, n_epochs):
        iterations_per_epoch = get_batch_iterations(len(inputs_train), batch_size)
        
        for step, batch_train in tqdm(enumerate(generator(inputs_train, targets_train)), desc=f'[Train] Iterations (epoch {epoch})', initial=0, total=iterations_per_epoch):
            itr += 1
            epoch_float = itr / iterations_per_epoch
            
            # 텐서 파싱 및 디바이스 이동
            inputs, targets, seq_lengths, mask = batch_train
            print('inputs:', inputs.shape) # inputs: torch.Size([128, *, 14])
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            
            '''
            # Forward + Backward + Optimizer step (model.train_step 내부에서 처리된다고 가정)
            pred_train, states_train, h_seqs, smoothing_ratios, loss_train, sub_losses_train = model.train_step(inputs, targets, mask)
            print('pred_train:', pred_train.shape) # pred_train: torch.Size([128, *, 104])
            # Corrector 준비 및 훈련
            cr_inputs_train, cr_targets_train, cr_mask_train = dl.generate_dataset_for_corrector(pred_train, states_train, h_seqs, inputs, targets, mask)
            print('cr_inputs_train:', cr_inputs_train.shape) # cr_inputs_train: torch.Size([128, *, 1039])
            cr_inputs_train, cr_targets_train, cr_mask_train = cr_inputs_train.to(device), cr_targets_train.to(device), cr_mask_train.to(device)
            cr_pred_train, cr_loss_train = model.corrector.train_step(cr_inputs_train, cr_targets_train, cr_mask_train)
            '''

            # =========================================================
            # Stage 1: Main Model 단독 학습 (데이터: Digit)
            # =========================================================
            if stage == 1:
                # Corrector는 건너뛰고 Main Model만 단독 학습
                model.train()
                pred_train, states_train, h_seqs, smoothing_ratios, loss_train, sub_losses_train = model.train_step(inputs, targets, mask)

                # Stage 1은 코렉터를 쓰지 않으므로 더미 코렉터 Loss 생성
                cr_pred_dummy = torch.zeros_like(targets)
                cr_loss_dummy = torch.tensor(0.0, device=device)

                ## 수정해야 함
                logger.log_train(model, itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, smoothing_ratios, loss_train, sub_losses_train, (cr_pred_dummy, cr_loss_dummy))

            
            # =========================================================
            # Stage 2: Corrector 사전 학습 (데이터: Alphabet)
            # =========================================================
            elif stage == 2:
                # 1. Main Model은 가중치를 Freeze하기 위해 평가 모드로 전환
                model.eval()
                with torch.no_grad():
                    # 순전파만 진행하여 데이터 생성에 필요한 상태값(h_seqs 등)만 추출
                    pred_train, states_train, h_seqs, smoothing_ratios = model(inputs)

                # 핵심 로직 메인 모델이 알파벳에 대해 내뱉은 엉망인 pred_train을 쓰지 않음
                # 정답(targets)에 인위적인 노이즈를 섞어 가짜 메인 모델 예측으로 둔갑
                noise = torch.randn_like(targets) * 0.05  # 노이즈 강도는 오차 수준에 맞게 조절
                simulated_pred = targets + noise

                # 2. Corrector 데이터 생성 (pred_train 대신 simulated_pred 사용)
                cr_inputs_train, cr_targets_train, cr_mask_train = dl.generate_dataset_for_corrector(
                    simulated_pred, states_train, h_seqs, inputs, targets, mask
                )
                cr_inputs_train, cr_targets_train, cr_mask_train = cr_inputs_train.to(device), cr_targets_train.to(device), cr_mask_train.to(device)
                
                # 3. Corrector만 훈련 (노이즈 제거 및 스타일 체득)
                model.corrector.train()
                cr_pred_train, cr_loss_train = model.corrector.train_step(cr_inputs_train, cr_targets_train, cr_mask_train)

                # 추가된 부분: Logger가 터지지 않도록 메인 모델의 Dummy Loss 할당
                loss_dummy = torch.tensor(0.0, device=device)
                sub_losses_dummy = (
                    torch.tensor(0.0, device=device), # loss_gmm
                    torch.tensor(0.0, device=device), # loss_hov
                    torch.tensor(0.0, device=device), # loss_eod
                    torch.tensor(0.0, device=device)  # loss_smoothing
                )

                logger.log_train(model, itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, smoothing_ratios, loss_dummy, sub_losses_dummy, (cr_pred_train, cr_loss_train))


            # =========================================================
            # Stage 3: Corrector Fine-tuning (데이터: Digit)
            # =========================================================
            elif stage == 3:
                # 1. Main Model 가중치 동결
                model.eval()
                with torch.no_grad():
                    # 이번엔 숫자가 들어오므로, 메인 모델이 꽤 괜찮은 숫자 궤적(pred_train)을 뱉어내도록
                    pred_train, states_train, h_seqs, smoothing_ratios = model(inputs)
                
                # 2. 메인 모델의 실제 숫자 예측값을 그대로 사용하여 데이터 생성
                cr_inputs_train, cr_targets_train, cr_mask_train = dl.generate_dataset_for_corrector(
                    pred_train, states_train, h_seqs, inputs, targets, mask
                )
                cr_inputs_train, cr_targets_train, cr_mask_train = cr_inputs_train.to(device), cr_targets_train.to(device), cr_mask_train.to(device)
                
                # 3. Corrector만 미세 조정 (숫자 궤적에 smoothing 스타일 입히기)
                model.corrector.train()
                cr_pred_train, cr_loss_train = model.corrector.train_step(cr_inputs_train, cr_targets_train, cr_mask_train)

                loss_dummy = torch.tensor(0.0, device=device)
                sub_losses_dummy = (
                    torch.tensor(0.0, device=device), 
                    torch.tensor(0.0, device=device), 
                    torch.tensor(0.0, device=device), 
                    torch.tensor(0.0, device=device)
                )

                logger.log_train(model, itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, smoothing_ratios, loss_dummy, sub_losses_dummy, (cr_pred_train, cr_loss_train))

            # VALIDATION (based on iteration) ====================
            if ((itr-1) % val_period_itr) == (val_period_itr-1):
                iterations_val = get_batch_iterations(len(inputs_val), batch_size)
                logger.log_val_init(iterations_val)
                for step, batch_val in tqdm(enumerate(generator(inputs_val, targets_val)), desc=f'[Val] Iterations (epoch {epoch})', initial=0, total=iterations_val):
                    inputs, targets, seq_lengths, mask = batch_val
                    inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                    
                    # Note: 원본 코드에서 test_step 반환값 개수가 달랐던 잠재적 버그를 6개로 통일하여 명확히 잡음
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

        if stage== 1:
            if (epoch+1) % train_synth_period == 0:
                logger.log_train_synth(itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, loss_train, sub_losses_train)

        elif stage==2 or stage ==3:
            if (epoch+1) % train_synth_period == 0:
                logger.log_train_synth(itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, cr_loss_train, sub_losses_dummy)

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
    parser.add_argument('--hparams', type=str, default='hparams_transfer.yaml', help='Specify an YAML path that contains hyperparameters of this training.')
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