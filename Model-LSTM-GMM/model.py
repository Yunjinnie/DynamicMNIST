import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras.layers import LSTMCell, SimpleRNNCell, RNN, Dense, Concatenate, Dropout
from data_utils import get_character_vector
from utils import sample_gmm_batch, sample_gmm_batch_tf, parse_pred
from loss import Loss, CorrectorLoss

class Model(tf.keras.Model):
    def __init__(self, hparams, data_loader):
        super(Model, self).__init__()

        self.hp = hparams
        self.dl = data_loader

        self.epochs = tf.Variable(0, trainable=False)
        self.iterations = tf.Variable(0, trainable=False)
        self.dx_norm = tf.Variable(1.0, trainable=False)
        self.dy_norm = tf.Variable(1.0, trainable=False)
        self.d2x_norm = tf.Variable(1.0, trainable=False)
        self.d2y_norm = tf.Variable(1.0, trainable=False)
        self.set_norms(self.dl.dx_norm, self.dl.dy_norm, self.dl.d2x_norm, self.dl.d2y_norm)

        # Data parameters ==================================================================
        self.v_direction_dim = hparams['v_direction_dim']
        self.v_speed_dim = hparams['v_speed_dim']
        self.v_hover_dim = hparams['v_hover_dim']
        self.v_eod_dim = hparams['v_eod_dim']
        self.v_character_dim = hparams['v_character_dim']
        self.character_set = list(hparams['character_set'])
        # Model Hparams ==================================================================
        self.target = hparams['target']
        self.input_kine = hparams['input_kine']
        self.lstm_dim = hparams['lstm_dim']
        self.n_g_mixtures = hparams['n_g_mixtures']
        # Training Hparams ==================================================================
        self.l2_reg_coef = hparams['l2_reg_coef']
        self.max_grad_norm = hparams['max_grad_norm']
        self.learning_rate = hparams['learning_rate']
        # Running Hparams ==================================================================
        self.max_steps = hparams['max_steps']
        self.delayed_steps = hparams['delayed_steps']
        self.delayed_steps_ctrl = hparams['delayed_steps_ctrl']
        self.delayed_steps_ctrl_test = hparams['delayed_steps_ctrl_test']
        self.smooth_dxdy_ratio = hparams['smooth_dxdy_ratio']
        # Inferred hyperparamters ==================================================================
        self.input_dim = hparams['input_dim']
        self.g_mixtures_dim = hparams['g_mixtures_dim'] # These 17 2D Gaussians predict dx and dy at each step.
        self.output_dim = hparams['output_dim']

        if self.hp['p_spatial_error'] == 0:
            self.use_spatial_error = False
        else:
            self.use_spatial_error = True

        # Initialize the model ==================================================================
        self.l2_reg = keras.regularizers.l2(self.l2_reg_coef)

        self.use_lstm = hparams['use_lstm']
        self.use_layernorm = hparams['use_layernorm']

        if self.use_lstm:
            if self.use_layernorm:
                rnn_cell = tfa.rnn.LayerNormLSTMCell
            else:
                rnn_cell = LSTMCell
        else:
            if self.use_layernorm:
                rnn_cell = tfa.rnn.LayerNormSimpleRNNCell
            else:
                rnn_cell = SimpleRNNCell

        self.lstmLayer1 = RNN(
            rnn_cell(
                self.lstm_dim, kernel_regularizer=self.l2_reg,
                dropout=hparams['lstm_dropout'], recurrent_dropout=hparams['lstm_r_dropout']),
            return_sequences=True, return_state=True)
        self.lstmLayer2 = RNN(
            rnn_cell(
                self.lstm_dim, kernel_regularizer=self.l2_reg,
                dropout=hparams['lstm_dropout'], recurrent_dropout=hparams['lstm_r_dropout']),
            return_sequences=True, return_state=True)

        #self.dense0 = Dense(256, kernel_regularizer=self.l2_reg)
        #self.dense_do0 = Dropout(0.2)

        self.dense1 = Dense(self.output_dim)
        #self.dense1 = Dense(self.output_dim, kernel_regularizer=self.l2_reg)

        self.smoothing_net = SmoothingNet(hparams)
        self.corrector = Corrector(hparams)

        # Initialize the loss ==================================================================
        self.loss_fn = Loss(hparams)

        # Initialize the optimizer ==================================================================
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate,
            decay_steps=1000,
            #decay_rate=0.9, # default: 0.1
            #decay_rate=0.1, # default: 0.1
            decay_rate=1.0, # 1 means no decaying. default: 0.1
            staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, epsilon=hparams['epsilon'])


    def set_norms(self, dx_norm, dy_norm, d2x_norm, d2y_norm):
        self.dx_norm.assign(dx_norm)
        self.dy_norm.assign(dy_norm)
        self.d2x_norm.assign(d2x_norm)
        self.d2y_norm.assign(d2y_norm)

    def call(self, inputs, states=None, one_step=False, bias=0):
        '''
        Call arguments:
        - inputs: A 2D tensor, with shape of `[batch, sequence_length, feature]`.
        - states: List of 2 tensors that corresponding to the cell's units. Both of
            them have shape `[batch, units]`, the first tensor is the memory state
            from previous time step, the second tensor is the carry state from
            previous time step. For timestep 0, the initial state provided by user
            will be feed to cell.
            - states = (states1, states2)
            - states1 = (h_lstmLayer1, c_lstmLayer1)
            - states2 = (h_lstmLayer2, c_lstmLayer2)
        '''
        if (not one_step) or (states is None):
            states1, states2 = None, None
        else:
            states1, states2 = states

        character_seq = inputs[...,4:]

        #inputs = self.bn(inputs)
        if self.use_lstm:
            h_seqs, h_lstmLayer1, c_lstmLayer1 = self.lstmLayer1(inputs, initial_state=states1)
            #h_seqs = Concatenate(axis=-1)([h_seqs, character_seq])
            h_seqs, h_lstmLayer2, c_lstmLayer2 = self.lstmLayer2(h_seqs, initial_state=states2)
            #h_seqs = Concatenate(axis=-1)([h_seqs, character_seq])
        else:
            if states1 is not None:
                states1 = states1[0]; states2 = states2[0];
            h_seqs, h_lstmLayer1 = self.lstmLayer1(inputs, initial_state=states1)
            c_lstmLayer1 = h_lstmLayer1
            h_seqs, h_lstmLayer2 = self.lstmLayer2(h_seqs, initial_state=states2)
            c_lstmLayer2 = h_lstmLayer2
        gmm_logits = self.dense1(h_seqs)
        #gmm_logits = self.dense1(self.dense_do0(self.dense0(h_seqs)))
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod = parse_pred(gmm_logits, self.hp)
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = self.gmm_layer(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, bias)
        z_hov = tf.sigmoid(z_hov)
        z_eod = tf.sigmoid(z_eod)

        outputs = Concatenate(axis=-1)([z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod])
        states1 = (h_lstmLayer1, c_lstmLayer1)
        states2 = (h_lstmLayer2, c_lstmLayer2)
        states = (states1, states2)

        smoothing_ratios = self.smoothing_net(inputs)

        return outputs, states, h_seqs, smoothing_ratios

    def gmm_layer(self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, bias=0):
        # A mixture of Gaussians
        # The weights for each Gaussian: pi's. Softmax all the pi's:
        max_pi = tf.reduce_max( z_pi, -1, keepdims=True)
        z_pi = tf.subtract( z_pi, max_pi ) #EdJ: subtract max pi for numerical stabilization
        z_pi = tf.exp( z_pi * (1+bias) ) #eq 19
        normalize_pi = tf.math.reciprocal( tf.reduce_sum( z_pi, -1, keepdims=True))
        z_pi = tf.multiply( normalize_pi, z_pi ) #19

        z_sigma1 = tf.exp( z_sigma1 * (1+bias) ) #eq 21
        z_sigma2 = tf.exp( z_sigma2 * (1+bias) )
        z_corr = tf.tanh( z_corr ) #eq 22
        z_corr = .95 * z_corr #avoid -1 and 1

        return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, mask):
        with tf.GradientTape() as tape:
            y_pred, states, h_seqs, smoothing_ratios = self(x, training=True)
            loss, sub_losses = self.loss_fn(y_pred, y, mask, smoothing_ratios, use_spatial_error=self.use_spatial_error)
        grads = tape.gradient(loss, self.trainable_weights)
        grads = [tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm) for grad in grads if grad is not None]
        #grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return y_pred, states, h_seqs, smoothing_ratios, loss, sub_losses

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, x, y, mask):
        y_pred, states, h_seqs, smoothing_ratios = self(x, training=False)
        loss, sub_losses = self.loss_fn(y_pred, y, mask, smoothing_ratios)

        return y_pred, states, h_seqs, smoothing_ratios, loss, sub_losses

    #@tf.function(experimental_relax_shapes=True) # Not fast.
    def one_step(self, inputs, states, bias=0):
        outputs, states, h_seqs, smooth_ratios = self.call(inputs, states, one_step=True, bias=bias)
        return outputs, states, h_seqs, smooth_ratios

    def infer_batch(self, character, batch_size, eod_stop=False, eod_slice=False, bias=0, tf_sampling=False, return_preds=False,
            use_corrector=True, null_correction=False):
        '''Inferenece. Feedback the current prediction to the next input.

        character : str. The character to write.
        '''

        i_step = 0
        states = None # Zero vectors for initial hidden and cell states.
        cr_states = None # The states of Corrector

        character_vector = get_character_vector(character, self.character_set)
        character_vector_batch = np.repeat(np.expand_dims(character_vector, 0), batch_size, axis=0)

        first_in_vector = np.concatenate(
            (np.zeros(self.input_dim-self.v_character_dim),
             character_vector))
        first_in_vector = np.expand_dims(np.expand_dims(first_in_vector,0),0)
        first_in_vector = np.repeat(first_in_vector, batch_size, axis=0)
        inputs = first_in_vector # zeros for previous velocity, hovering label, and character ending label; plus, one-hot character vector.
        delayed_inputs = [first_in_vector for _ in range(self.delayed_steps_ctrl_test)] # If delayed_steps == 0, then this list will be an empty list.

        dx_seq = list()
        dy_seq = list()
        hov_seq = list()
        eod_seq = list()

        seq_lengths = np.full(batch_size, self.max_steps)
        eod_marker = np.zeros(batch_size, dtype=np.int64)

        y_preds = list()

        if use_corrector:
            cr_sensory_input_dim = 3
            cr_dx_seq = list()
            cr_dy_seq = list()
            rnn_dx_seq = list()
            rnn_dy_seq = list()
            empty_cr_inputs = np.concatenate(
                (np.zeros(cr_sensory_input_dim), character_vector))
            empty_cr_inputs = np.expand_dims(np.expand_dims(empty_cr_inputs,0),0)
            empty_cr_inputs = np.repeat(empty_cr_inputs, batch_size, axis=0)
            delayed_cr_inputs = [empty_cr_inputs for _ in range(self.delayed_steps+1)]
            #delayed_cr_inputs = []

        while i_step < self.max_steps:

            y_pred, states, h_seqs, smoothing_ratios = self.one_step(inputs, states, bias)

            y_pred = tf.squeeze(y_pred, axis=[1]) # Squeeze the time dimension.
            if tf_sampling:
                next_var_x, next_var_y, hov, eod = sample_gmm_batch_tf(*parse_pred(y_pred, self.hp))
            else:
                next_var_x, next_var_y, hov, eod = sample_gmm_batch(*parse_pred(y_pred, self.hp))

            # Collect RNN's outputs without correction.
            if use_corrector:
                next_x, next_y, next_dx, next_dy, next_d2x, next_d2y = self.get_kinematics(next_var_x, next_var_y, i_step)
                rnn_dx_seq.append(next_dx)
                rnn_dy_seq.append(next_dy)

            if use_corrector and \
                self.hp['correction_output'] and (
                    self.hp['use_corrector_input_gmm_params'] \
                    or ((not self.hp['use_corrector_input_gmm_params']) and i_step >= self.delayed_steps + 1)
                ):
                # Take out the earliest element in delayed_cr_inputs.
                cr_inputs = delayed_cr_inputs.pop(0)
                if self.hp['use_corrector_input_gmm_params']:
                    cr_inputs = np.concatenate([cr_inputs, h_seqs], axis=-1, dtype=np.float32)
                var_xy = np.expand_dims(np.stack([next_var_x, next_var_y], axis=-1), axis=-2)
                #var_xy = cr_inputs[...,:2]
                cr_inputs = np.concatenate([var_xy, cr_inputs], axis=-1, dtype=np.float32)

                # Corrector post-processes the model's sampled output.
                (cr_next_var_x, cr_next_var_y), (cr_var_x, cr_var_y), cr_states = self.corrector.one_step(cr_inputs, cr_states)
                if not null_correction:
                    next_var_x, next_var_y = cr_next_var_x, cr_next_var_y

            # Test smoothing_ratios
            if i_step > 0:
                smoothing_ratios = tf.squeeze(smoothing_ratios)
                next_var_x = smoothing_ratios * next_var_x + (1 - smoothing_ratios) * prev_var_x
                next_var_y = smoothing_ratios * next_var_y + (1 - smoothing_ratios) * prev_var_y

            # [START] SENSORY DOMAIN ==========
            # Stack the results at this step.
            next_x, next_y, next_dx, next_dy, next_d2x, next_d2y = self.get_kinematics(next_var_x, next_var_y, i_step)
            dx_seq.append(next_dx)
            dy_seq.append(next_dy)
            hov_seq.append(hov)
            eod_seq.append(eod)
            y_preds.append(y_pred)
            # [END] SENSORY DOMAIN ==========

            # Chnage the RNN input using Corrector.
            if use_corrector and \
                self.hp['correction_input'] and (
                self.hp['use_corrector_input_gmm_params'] \
                or ((not self.hp['use_corrector_input_gmm_params']) and i_step >= self.delayed_steps + 1)
            ):
                # Take out the earliest element in delayed_cr_inputs.
                cr_inputs = delayed_cr_inputs.pop(0)
                if self.hp['use_corrector_input_gmm_params']:
                    cr_inputs = np.concatenate([cr_inputs, h_seqs], axis=-1, dtype=np.float32)
                # Corrector post-processes the model's sampled output.
                (cr_next_var_x, cr_next_var_y), (cr_var_x, cr_var_y), cr_states = self.corrector.one_step(cr_inputs, cr_states)
                if not null_correction:
                    next_var_x, next_var_y = cr_next_var_x, cr_next_var_y

            # Get kinematics to construct the RNN input.
            next_x, next_y, next_dx, next_dy, next_d2x, next_d2y = self.get_kinematics(next_var_x, next_var_y, i_step)
            if use_corrector:
                cr_x, cr_y, cr_dx, cr_dy, cr_d2x, cr_d2y = self.get_kinematics(cr_var_x, cr_var_y, i_step)
                cr_dx_seq.append(cr_dx)
                cr_dy_seq.append(cr_dy)

            # Input masking
            input_dxdy_mask = np.random.binomial(1, 1-self.hp['p_mask_input_dxdy_test'], next_dx.shape)
            input_next_x = next_x * input_dxdy_mask
            input_next_y = next_y * input_dxdy_mask
            input_next_dx = next_dx * input_dxdy_mask
            input_next_dy = next_dy * input_dxdy_mask
            input_next_d2x = next_d2x * input_dxdy_mask
            input_next_d2y = next_d2y * input_dxdy_mask

            # Set the next input
            inputs = list()
            if 'p' in self.input_kine:
                inputs += [input_next_x, input_next_y]
            if 'v' in self.input_kine:
                inputs += [input_next_dx, input_next_dy]
            if 'a' in self.input_kine:
                inputs += [input_next_d2x, input_next_d2y]
            inputs += [hov, eod]
            inputs = np.stack(inputs, axis=1)
            inputs = np.hstack((inputs, character_vector_batch))
            #inputs = np.expand_dims(np.expand_dims(inputs,0),0)
            inputs = np.expand_dims(inputs,1) # Expand at the time dimension.

            # Get delayed inputs.
            delayed_inputs.append(inputs) # Append the last prediction into the end of `delayed_inputs`.
            inputs = delayed_inputs.pop(0) # Take off the first element from `delayed_inputs`.

            # Set the next input of Corrector
            if use_corrector:
                # Stack cr_inputs in a queue delayed_cr_inputs.
                input_dxdy_mask_cr = input_dxdy_mask
                #input_dxdy_mask_cr = np.random.binomial(1, 1-self.hp['p_mask_input_dxdy_test'], next_var_x.shape)
                #input_dxdy_mask_cr = np.random.binomial(1, 0, next_var_x.shape)
                next_var_x_masked = next_var_x * input_dxdy_mask_cr
                next_var_y_masked = next_var_y * input_dxdy_mask_cr
                cr_inputs = np.concatenate([np.stack([next_var_x_masked, next_var_y_masked, hov], axis=-1), character_vector_batch], axis=-1, dtype=np.float32) # var_x, var_y, hov, char
                cr_inputs = np.expand_dims(cr_inputs, axis=-2)
                delayed_cr_inputs.append(cr_inputs)

            # Set prev var x and y for SmoothNet
            prev_var_x = next_var_x
            prev_var_y = next_var_y

            # Record sequence lengths according to eod.
            for i_batch, eod_sample in enumerate(eod):
                if (eod_marker[i_batch] == 0) and (eod_sample == 1):
                    seq_lengths[i_batch] = i_step + 1
                    eod_marker[i_batch] = 1

            # End of the character drawing if EOD is 1. (EOD is either 0 or 1.)
            if (np.sum(eod_marker) == batch_size) and eod_stop:
                break

            # End of this step.
            i_step += 1

        # Each sequence matrix has the shape of [batch_size, seq_lengths.max()]
        dx_seq = np.stack(dx_seq, axis=1)
        dy_seq = np.stack(dy_seq, axis=1)
        hov_seq = np.stack(hov_seq, axis=1)
        eod_seq = np.stack(eod_seq, axis=1)

        # Slice each seuqence by its sequence length.
        results = list()
        for i_batch in range(batch_size):
            L = seq_lengths[i_batch] if eod_slice else i_step
            result = [
                dx_seq[i_batch,:L],
                dy_seq[i_batch,:L],
                hov_seq[i_batch,:L],
                eod_seq[i_batch,:L],
            ]
            result = np.stack(result, axis=-1)
            results.append(result)

        if eod_slice:
            results = np.asarray(results, dtype=object)
        else:
            results = np.stack(results, axis=0)

        if use_corrector:
            cr_dx_seq = np.stack(cr_dx_seq, axis=1)
            cr_dy_seq = np.stack(cr_dy_seq, axis=1)
            rnn_dx_seq = np.stack(rnn_dx_seq, axis=1)
            rnn_dy_seq = np.stack(rnn_dy_seq, axis=1)

            corrections = list()
            rnn_results = list()
            for i_batch in range(batch_size):
                if self.hp['use_corrector_input_gmm_params']:
                    L = seq_lengths[i_batch] if eod_slice else i_step
                else:
                    L = seq_lengths[i_batch] - (self.delayed_steps + 1) if eod_slice else i_step
                correction = [
                    cr_dx_seq[i_batch,:L],
                    cr_dy_seq[i_batch,:L],
                    #hov_seq[i_batch,:L],
                    #eod_seq[i_batch,:L],
                ]
                correction = np.stack(correction, axis=-1)
                corrections.append(correction)

                L = seq_lengths[i_batch] if eod_slice else i_step
                rnn_result = [
                    rnn_dx_seq[i_batch,:L],
                    rnn_dy_seq[i_batch,:L],
                ]
                rnn_result = np.stack(rnn_result, axis=-1)
                rnn_results.append(rnn_result)

            if return_preds:
                y_preds = tf.stack(y_preds)
                y_preds = tf.transpose(y_preds, perm=[1,0,2])

                return results, y_preds, corrections, rnn_results

            return results, corrections, rnn_results

        if return_preds:
            y_preds = tf.stack(y_preds)
            y_preds = tf.transpose(y_preds, perm=[1,0,2])

            return results, y_preds

        return results

    def get_kinematics(self, var_x, var_y, i_step):
        if i_step == 0:
            self.dx_prev = np.zeros_like(var_x)
            self.dy_prev = np.zeros_like(var_y)
            self.dx = np.zeros_like(var_x)
            self.dy = np.zeros_like(var_y)
            self.x = np.zeros_like(var_x)
            self.y = np.zeros_like(var_y)
        if self.target == 'v':
            self.dx = var_x * self.dx_norm
            self.dy = var_y * self.dy_norm
            self.x += self.dx
            self.y += self.dy
            self.d2x = self.dx - self.dx_prev
            self.d2y = self.dy - self.dy_prev
            self.dx_prev = self.dx
            self.dy_prev = self.dy
        elif self.target == 'a':
            self.d2x = var_x * self.d2x_norm
            self.d2y = var_y * self.d2y_norm
            self.dx += self.d2x
            self.dy += self.d2y
            self.x += self.dx
            self.y += self.dy
        else:
            exit()

        d2x = self.d2x / self.d2x_norm
        d2y = self.d2y / self.d2y_norm
        dx = self.dx / self.dx_norm
        dy = self.dy / self.dy_norm
        x, y = self.dl.normalize_position(self.x, self.y)

        return x, y, dx, dy, d2x, d2y


class SmoothingNet(tf.keras.Model):
    def __init__(self, hparams):
        super(SmoothingNet, self).__init__()
        self.hp = hparams
        self.smooth_dxdy_ratio = hparams['smooth_dxdy_ratio']
        self.use_trainable_smooth_ratio = hparams['use_trainable_smooth_ratio']
        self.use_static_trainable_smooth_ratio = hparams['use_static_trainable_smooth_ratio']
        self.input_dim = hparams['output_dim']
        self.smooth_net_dim = hparams['smooth_net_dim']
        self.l2_reg_coef = hparams['l2_reg_coef']
        self.output_dim = 1

        if not self.use_trainable_smooth_ratio:
            self.static_smooth_ratio_variable = tf.Variable(self.smooth_dxdy_ratio, trainable=False)
        elif self.use_static_trainable_smooth_ratio:
            self.static_smooth_ratio_variable = tf.Variable(3.0, trainable=True)
        else:
            self.l2_reg = keras.regularizers.l2(self.l2_reg_coef)

            self.rnn = RNN(
                tfa.rnn.LayerNormLSTMCell(
                    self.smooth_net_dim, kernel_regularizer=self.l2_reg,
                    dropout=hparams['lstm_dropout'], recurrent_dropout=hparams['lstm_r_dropout']),
                return_sequences=True, return_state=True)

            self.dense = Dense(self.output_dim)

    def call(self, inputs):
        '''
        INPUTS
        - inputs: tf.Tensor
          - shape: [batch_size, max_seq_length, input_dim]
          - components: concat([character_vector, dx, dy, hover, eod])
        '''
        if  not self.use_trainable_smooth_ratio:
            smooth_ratio = self.static_smooth_ratio_variable
        elif self.use_static_trainable_smooth_ratio:
            smooth_ratio = tf.sigmoid(self.static_smooth_ratio_variable)
        else:
            inputs = tf.stop_gradient(inputs)
            # Why don't the RNN take state variables?
            h_seq, h_last, c_last = self.rnn(inputs)
            smooth_ratio = tf.sigmoid(self.dense(h_seq))

        return smooth_ratio

class Corrector(tf.keras.Model):
    def __init__(self, hparams):
        super(Corrector, self).__init__()
        self.hp = hparams
        self.use_rnn = hparams['use_rnn_corrector'] # RNN OR FFN (Feed Forward Network)
        self.dim_rnn = 32
        self.dim_dense1 = self.dim_rnn
        if self.hp['use_corrector_output_gmm']:
            self.dim_output = hparams['g_mixtures_dim']
        else:
            self.dim_output = 2 # var_x, var_y: Follow RNN's output type: either v or a.
        self.learning_rate = hparams['learning_rate']
        self.l2_reg_coef = hparams['l2_reg_coef']
        self.max_grad_norm = hparams['max_grad_norm']
        self.l2_reg = keras.regularizers.l2(self.l2_reg_coef)

        # Initialize layers ==================================================================
        if self.use_rnn:
            self.rnn = RNN(
                tfa.rnn.LayerNormLSTMCell(
                    self.dim_rnn, kernel_regularizer=self.l2_reg,
                    dropout=hparams['lstm_dropout'], recurrent_dropout=hparams['lstm_r_dropout']),
                return_sequences=True, return_state=True)
        else:
            self.ffn = Dense(4 * self.dim_rnn)
            self.do0 = Dropout(0.5)
        self.dense1 = Dense(self.dim_dense1)
        self.do1 = Dropout(0.5)
        self.dense2 = Dense(self.dim_output)

        # Initialize the loss ==================================================================
        self.loss_fn = CorrectorLoss(hparams)

        # Initialize the optimizer ==================================================================
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate,
            decay_steps=1000,
            #decay_rate=0.9, # default: 0.1
            #decay_rate=0.1, # default: 0.1
            decay_rate=1.0, # 1 means no decaying. default: 0.1
            staircase=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=hparams['epsilon'])

    def call(self, inputs, states=None, one_step=False):
        #var_x_seqs, var_y_seqs, inputs = self.parse_inputs(inputs)
        var_x_seqs, var_y_seqs, _ = self.parse_inputs(inputs)
        #inputs = inputs[...,5:] # dx, dy, dx_delay, dy_delay, hov_delay # Making no sensory FB
        var_xy_seqs = tf.concat([var_x_seqs, var_y_seqs], axis=-1)
        inputs = tf.stop_gradient(inputs)
        if not one_step:
            states = None # default: zero inputs.
        if self.use_rnn:
            h_seq, h_last, c_last = self.rnn(inputs, initial_state=states)
            states = (h_last, c_last)
        else:
            h_seq = self.ffn(inputs)
            #h_seq = self.do0(h_seq, training=True)
            h_seq = self.do0(h_seq)
        h_seq2 = tf.nn.relu(self.dense1(h_seq))
        #h_seq2 = self.do1(h_seq2, training=True)
        h_seq2 = self.do1(h_seq2)
        corrections = self.dense2(h_seq2)

        if self.hp['use_corrector_output_gmm']:
            corrections = self.gmm_layer(corrections)

        #outputs = var_xy_seqs + corrections
        outputs = corrections

        return outputs, corrections, states, (var_x_seqs, var_y_seqs)

    def gmm_layer(self, corrections, bias=0):
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = parse_pred(corrections, self.hp, only_gmm=True)

        # A mixture of Gaussians
        # The weights for each Gaussian: pi's. Softmax all the pi's:
        max_pi = tf.reduce_max( z_pi, -1, keepdims=True)
        z_pi = tf.subtract( z_pi, max_pi ) #EdJ: subtract max pi for numerical stabilization
        z_pi = tf.exp( z_pi * (1+bias) ) #eq 19
        normalize_pi = tf.math.reciprocal( tf.reduce_sum( z_pi, -1, keepdims=True))
        z_pi = tf.multiply( normalize_pi, z_pi ) #19

        z_sigma1 = tf.exp( z_sigma1 * (1+bias) ) #eq 21
        z_sigma2 = tf.exp( z_sigma2 * (1+bias) )
        z_corr = tf.tanh( z_corr ) #eq 22
        z_corr = .95 * z_corr #avoid -1 and 1

        corrections = Concatenate(axis=-1)([z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr])

        return corrections

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, mask):
        with tf.GradientTape() as tape:
            y_pred, corrections, states, _ = self(x, training=True)
            #loss = self.loss_fn(y_pred, y, mask)
            loss = self.loss_fn(corrections, y, mask)
        grads = tape.gradient(loss, self.trainable_weights)
        grads = [tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm) for grad in grads if grad is not None]
        #grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return y_pred, loss

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, x, y, mask):
        y_pred, corrections, states, _ = self(x, training=False)
        #loss = self.loss_fn(y_pred, y, mask)
        loss = self.loss_fn(corrections, y, mask)


        return y_pred, loss

    #@tf.function(experimental_relax_shapes=True) # Not fast.
    def one_step(self, inputs, states, parse_outputs=True):
        outputs, corrections, states, (var_x_seqs, var_y_seqs) = self.call(inputs, states, one_step=True)
        if parse_outputs:
            #var_x_seqs, var_y_seqs = self.parse_outputs(outputs)
            var_x_seqs = var_x_seqs[:,0,0]
            var_y_seqs = var_y_seqs[:,0,0]
            if self.hp['use_corrector_output_gmm']:
                var_x_corrections, var_y_corrections = sample_gmm_batch_tf(*parse_pred(corrections, self.hp), only_gmm=True)
                var_x_corrections = var_x_corrections[:,0]
                var_y_corrections = var_y_corrections[:,0]
            else:
                var_x_corrections, var_y_corrections = self.parse_outputs(corrections)
                var_x_corrections = var_x_corrections[:,0,0]
                var_y_corrections = var_y_corrections[:,0,0]
            var_x_seqs += var_x_corrections
            var_y_seqs += var_y_corrections
            return (var_x_seqs, var_y_seqs), (var_x_corrections, var_y_corrections), states
        else:
            return outputs, corrections, states

    def parse_inputs(self, inputs):
        var_x_seqs = inputs[...,0:1] # shape == [batch_size, max_seq_len, 1]
        var_y_seqs = inputs[...,1:2]
        cr_inputs = inputs[...,2:]
        return var_x_seqs, var_y_seqs, cr_inputs

    def parse_outputs(self, outputs):
        var_x_seqs = outputs[...,0:1] # shape == [batch_size, max_seq_len, 1]
        var_y_seqs = outputs[...,1:2]
        return var_x_seqs, var_y_seqs