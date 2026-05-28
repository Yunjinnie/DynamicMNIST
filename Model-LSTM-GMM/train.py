import os
import argparse
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from glob import glob
from utils import get_batch_iterations
from data_utils import import_hparams
from data_utils import BatchGenerator
from data_utils import DataLoader
from loss import Loss
from model import Model
from logger import Logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def set_gpu_util(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def train(hparams, logger, checkpoint):
    # Load the dataset to be used:
    # inputs={(dx,dy,hov,eod)}
    # targets={(dx,dy,hov,eod)}
    # seq_lengths={seq_length}
    # characters={character}

    data_loader = DataLoader(hparams); dl = data_loader
    train_set = inputs_train, targets_train, seq_lengths_train, characters_train = data_loader.load_dataset('train')
    val_set = inputs_val, targets_val, seq_lengths_val, characters_val = data_loader.load_dataset('val')
    test_set = inputs_test, targets_test, seq_lengths_test, characters_test = data_loader.load_dataset('test')
    print('#Train =', inputs_train.shape[0])
    print('#Val   =', inputs_val.shape[0])
    print('#Test  =', inputs_test.shape[0])

    # Give the dataset sizes to logger.
    logger.set_dataset_size(len(inputs_train), len(inputs_val))

    # Construct a batch generator.
    generator = BatchGenerator(hparams)

    # Construct the whole loss function.
    loss_fn = Loss(hparams)

    # Initialize the whole model.
    model = Model(hparams, dl)

    # Compute dx_norm and dy_norm.
    print('Before scaling, (dx_mean, dy_mean, dx_std, dy_std)', (dl.dx_mean, dl.dy_mean, dl.dx_std, dl.dy_std))
    print('Before scaling, (d2x_mean, d2y_mean, d2x_std, d2y_std)', (dl.d2x_mean, dl.d2y_mean, dl.d2x_std, dl.d2y_std))
    logger.set_norms(dl.dx_norm, dl.dy_norm, dl.d2x_norm, dl.d2y_norm)
    dx_seqs = np.concatenate([target[:,0] for target in targets_train]); dy_seqs = np.concatenate([target[:,1] for target in targets_train])
    dx_mean = np.mean(dx_seqs); dy_mean = np.mean(dy_seqs); dx_std = np.std(dx_seqs); dy_std = np.std(dy_seqs)
    print('After scaling, targets (x_mean, y_mean, x_std, y_std)', (dx_mean, dy_mean, dx_std, dy_std))

    # Initialize an DTW classifier.
    logger.init_dtw_classifier(dl.transform_set_a2v(val_set))
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

    # If a checkpoint or pretrained weights are provided, ...
    if checkpoint != '':
        model.load_weights(checkpoint)
        itr = model.iterations.numpy()
        epoch_first = model.epochs.numpy() + 1
        logger.set_norms(dl.dx_norm, dl.dy_norm, dl.d2x_norm, dl.d2y_norm)

    for epoch in range(epoch_first, n_epochs):
        #logger.log_test_synth(0, 0, model, test_set)
        # TRAINING ====================
        iterations_per_epoch = get_batch_iterations(len(inputs_train), batch_size)
        for step, batch_train in tqdm(enumerate(generator(inputs_train, targets_train)), desc='[Train] Iterations (epoch {})'.format(epoch), initial=0, total=iterations_per_epoch):
            itr += 1
            epoch_float = itr / iterations_per_epoch
            # Parse the batch.
            inputs, targets, seq_lengths, mask = batch_train
            # Forward propagation and backward propgation.
            pred_train, states_train, h_seqs, smoothing_ratios, loss_train, sub_losses_train = model.train_step(inputs, targets, mask)
            # Prepare a batch for the coorector.
            cr_inputs_train, cr_targets_train, cr_mask_train = dl.generate_dataset_for_corrector(pred_train, states_train, h_seqs, inputs, targets, mask)
            cr_pred_train, cr_loss_train = model.corrector.train_step(cr_inputs_train, cr_targets_train, cr_mask_train)
            # Log the training result of this mini-batch.
            logger.log_train(model, itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, smoothing_ratios, loss_train, sub_losses_train, (cr_pred_train, cr_loss_train))

            # VALIDATION (based on iteration) ====================
            if ((itr-1) % val_period_itr) == (val_period_itr-1):
                iterations_val = get_batch_iterations(len(inputs_val), batch_size)
                logger.log_val_init(iterations_val)
                iterations_per_epoch_val = get_batch_iterations(len(inputs_val), batch_size)
                for step, batch_val in tqdm(enumerate(generator(inputs_val, targets_val)), desc='[Val] Iterations (epoch {})'.format(epoch), initial=0, total=iterations_per_epoch_val):
                    # Parse the batch.
                    inputs, targets, seq_lengths, mask = batch_val
                    # Forward propagation.
                    pred_val, states_val, h_seqs_val, loss_val, sub_losses_val = model.test_step(inputs, targets, mask)
                    # Prepare a batch for the coorector.
                    cr_inputs_val, cr_targets_val, cr_mask_val = dl.generate_dataset_for_corrector(pred_val, states_val, h_seqs_val, inputs, targets, mask)
                    cr_pred_val, cr_loss_val = model.corrector.test_step(cr_inputs_val, cr_targets_val, cr_mask_val)
                    # Accumulate validation evaluation for logging.
                    logger.log_val_eval(dl.transform_set_a2v(batch_val), pred_val, loss_val, sub_losses_val, (cr_pred_val, cr_loss_val))
                # Log validation for this epoch.
                logger.log_val(itr)

            # TEST (based on iteration) ====================
            if ((itr-1) % test_period_itr) == (test_period_itr-1):
            #if ((itr-1) % test_period_itr) == (test_period_itr-1) or (itr < 100):
                logger.log_test_synth(itr, epoch_float, model, dl.transform_set_a2v(test_set))
                if logger.save_model_now():
                    logger.save_model(model, epoch_float, itr)

            # Stop iteration
            if itr == hparams['stop_itr']:
                exit()



        #print('model.loss_fn.cnt_sp', model.loss_fn.cnt_sp)

        ## HERE: Generate a random training image.
        if (epoch+1) % train_synth_period == 0:
            logger.log_train_synth(itr, epoch_float, dl.transform_set_a2v(batch_train), pred_train, loss_train, sub_losses_train)
        ## END: Generate a random training image.

        # VALIDATION ====================
        if ((epoch+1) % val_period_epoch) == 0:
            iterations_val = get_batch_iterations(len(inputs_val), batch_size)
            logger.log_val_init(iterations_val)
            iterations_per_epoch_val = get_batch_iterations(len(inputs_val), batch_size)
            for step, batch_val in tqdm(enumerate(generator(inputs_val, targets_val)), desc='[Val] Iterations (epoch {})'.format(epoch), initial=0, total=iterations_per_epoch_val):
                # Parse the batch.
                inputs, targets, seq_lengths, mask = batch_val
                # Forward propagation.
                pred_val, states_val, h_seqs_val, smoothing_ratios, loss_val, sub_losses_val = model.test_step(inputs, targets, mask)
                # Prepare a batch for the coorector.
                cr_inputs_val, cr_targets_val, cr_mask_val = dl.generate_dataset_for_corrector(pred_val, states_val, h_seqs_val, inputs, targets, mask)
                cr_pred_val, cr_loss_val = model.corrector.test_step(cr_inputs_val, cr_targets_val, cr_mask_val)
                # Accumulate validation evaluation for logging.
                logger.log_val_eval(dl.transform_set_a2v(batch_val), pred_val, smoothing_ratios, loss_val, sub_losses_val, (cr_pred_val, cr_loss_val))
            # Log validation for this epoch.
            logger.log_val(itr)

        ## HERE: Generate a random validation image by teacher-forcing.
        if (epoch+1) % val_period_epoch == 0:
            logger.log_val_synth(itr, epoch_float, dl.transform_set_a2v(batch_val), pred_val, loss_val, sub_losses_val, model)
        ## END: Generate a random validation image by teacher-forcing.

        # HERE: Generate a random validation image without teacher-forcing (with self-feedback (SF)).
        #if ((epoch+1) % synth_period_SF == 0) and (epoch != 0):
        if (epoch+1) % test_period_epoch == 0:
            logger.log_test_synth(itr, epoch_float, model, dl.transform_set_a2v(test_set))
            if logger.save_model_now():
                logger.save_model(model, epoch_float, itr)
        if epoch % period_to_save_model == (period_to_save_model - 1):
            # Save the current model.
            logger.save_model(model, epoch_float, itr)
        ## END: Generate a random validation image without teacher-forcing.


if __name__ == '__main__':
    '''
    [Example command]
    python train.py
        --dir_vector_data "dir_vector_data"
        --dir_velocity_data "dir_velocity_data"
        --dt 20
    '''
    parser = argparse.ArgumentParser(
        description='Train a recurrent neural network to generate single handwritten characters.')
    parser.add_argument('-p', '--project', type=str, required=True,
        help='Specify a project name that groups running sessions.')
    parser.add_argument('-r', '--run', type=str, default='run',
        help='Specify a run name that identifies the current running execution.')
    parser.add_argument('--run_resume', type=str, default='',
        help='Specify the WandB run ID where you want to resume to log.')
    parser.add_argument('--checkpoint', type=str, default='',
        help='Specify a checkpoint path where the model import model parameters.')
    parser.add_argument('--hparams', type=str, default='hparams.yaml',
        help='Specify an YAML path that contains hyperparameters of this training.')
    parser.add_argument('--off_wandb', action='store_true',
        help='Turn off WandB logging.')
    parser.add_argument('--debug', action='store_true',
        help='Turn on the test mode where the size of splitted datasets is the batch size.')
    parser.add_argument('--use_gpu', type=bool, default=True,
        help='Use GPUs instead of CPUs')
    parser.add_argument('-g', '--gpus', type=str, default='0',
        help='Specify GPU numbers, e.g., "1" for a single GPU and "0,2" for mulitple GPUs.')

    args = parser.parse_args()

    # Import hyperparamters from an YAML file.
    hparams = import_hparams(args.hparams, v_type='cartesian')

    # Set the run ID.
    str_datetime = datetime.now().strftime("%y%m%d%H%M%S")
    run_id = '{}-{}'.format(args.run, str_datetime)
    print('- Project name:', args.project)
    print('- Run name:', run_id)
    print('- Target motor command: {}'.format(hparams['target']))
    print('- Input feeback: {}'.format(hparams['input_kine']))
    if args.run_resume != '':
        run_id = args.run_resume

    # Set the random seed for reproducibility.
    set_random_seed(hparams['seed'])

    # GPU Settings
    if args.use_gpu:
        set_gpu_util(args.gpus)

    # Construct a logger.
    logger = Logger(args.project, run_id, hparams, args.off_wandb)
    for path in [args.hparams] + glob('*.py'):
        logger.archive_file(path)

    if args.debug:
        hparams['train_used_ratio'] = hparams['batch_size']
        hparams['val_used_ratio'] = hparams['batch_size_eval']
        hparams['test_used_ratio'] = hparams['batch_size_eval']
        hparams['batch_size_test'] = 1

    print('hparam and python files are saved!')

    # Train the network.
    train(hparams, logger, args.checkpoint)
