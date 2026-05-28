import numpy as np
import tensorflow as tf
#tf.executing_eagerly()
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils import tf_2d_normal, parse_pred, parse_target, sample_gmm_batch_tf


class Loss(tf.keras.losses.Loss):
    def __init__(self, hparams):
        super(Loss, self).__init__()
        self.hp = hparams
        self.cnt_sp = tf.Variable(0.0)

    def __call__(self, pred, trgt, mask, smoothing_ratios, use_spatial_error=False):
        # From https://github.com/edwin-de-jong/incremental-sequence-learning/blob/ab1cd9ef815094fcd0f272f1b4fc6d2f841ea2a5/model.py#L125
        '''
        1 for direction and 2 for speeds
        Predictions: z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod
        Targets: x1_data, x2_data, hov_data, eod_data
        '''
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod = parse_pred(pred, self.hp)
        x1_data, x2_data, hov_data, eod_data = parse_target(trgt)

        if use_spatial_error:
            dx_trgt_spatial, dy_trgt_spatial = self.get_dxdy_targets_from_spatial_error(pred, trgt, self.hp)
            dx_trgt_spatial, dy_trgt_spatial = tf.expand_dims(dx_trgt_spatial, axis=-1), tf.expand_dims(dy_trgt_spatial, axis=-1)
            x1_data, x2_data = self.blend_two_dxdy(x1_data, x2_data, dx_trgt_spatial, dy_trgt_spatial, self.hp['p_spatial_error'])
            self.cnt_sp.assign_add(1.0)
        loss0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr) #batch_nrpoints x seq_len x nrmixtures

        # implementing eq # 26 of http://arxiv.org/abs/1308.0850
        epsilon = 1e-10
        loss1 = tf.multiply(loss0, z_pi)
        loss1 = tf.reduce_sum(loss1, -1) #batch_nrpoints x seq_len
        loss1 = -tf.math.log(loss1 + epsilon) # atf the beginning, some errors are exactly zero.
        loss_gmm = tf.math.multiply_no_nan(loss1, mask) # Remove losses out of the sequence length. # loss1 *= mask
        loss_gmm = tf.reduce_sum(loss_gmm) / tf.reduce_sum(mask)


        z_hov = tf.squeeze(z_hov)
        hov_data =  tf.squeeze(hov_data)
        loss2 = tf.multiply(z_hov, hov_data) + tf.multiply(1 - z_hov, 1 - hov_data) #eq 26 rightmost part
        loss2 = -tf.math.log(loss2 + epsilon)
        loss_hov = tf.math.multiply_no_nan(loss2, mask) # loss2 *= mask
        loss_hov = tf.reduce_sum(loss_hov) / tf.reduce_sum(mask)

        z_eod =  tf.squeeze(z_eod)
        eod_data =  tf.squeeze(eod_data)
        loss3 = tf.multiply(z_eod, eod_data) + tf.multiply(1 - z_eod, 1 - eod_data) #analogous for eod
        loss3 = -tf.math.log(loss3 + epsilon)
        loss_eod = tf.math.multiply_no_nan(loss3, mask) # loss3 *= mask
        loss_eod = tf.reduce_sum(loss_eod) / tf.reduce_sum(mask)

        loss_smoothing = self.get_smoothing_loss(z_mu1, z_mu2, x1_data, x2_data, smoothing_ratios, mask)

        loss_total = loss_gmm + loss_hov + loss_eod + loss_smoothing

        return loss_total, (loss_gmm, loss_hov, loss_eod, loss_smoothing)

    def get_smoothing_loss(self, z_mu1, z_mu2, x1_data, x2_data, smoothing_ratios, mask):
        '''
        z_mu1.shape == [batch_size, max_seq_length, n_gmm]
        z_mu2.shape == [batch_size, max_seq_length, n_gmm]
        smoothing_ratios.shape == [batch_size, max_seq_length, 1]
        '''
        n_gmm = tf.shape(z_mu1)[-1]
        max_seq_length = tf.shape(z_mu1)[-2]

        # Broadcasting
        smoothing_ratios = tf.broadcast_to(smoothing_ratios, tf.shape(z_mu1))
        x1_data = tf.broadcast_to(x1_data, tf.shape(z_mu1))
        x2_data = tf.broadcast_to(x2_data, tf.shape(z_mu1))
        mask = tf.broadcast_to(tf.expand_dims(mask, axis=-1), tf.shape(z_mu1))

        # Collector
        z_mu1_smooth = tf.TensorArray(tf.float32, size=max_seq_length, dynamic_size=False)
        z_mu2_smooth = tf.TensorArray(tf.float32, size=max_seq_length, dynamic_size=False)
        z_mu1_smooth = z_mu1_smooth.write(0, z_mu1[:,0,:])
        z_mu2_smooth = z_mu2_smooth.write(0, z_mu2[:,0,:])

        # [START] Smoothing process
        for t in tf.range(1, max_seq_length):
            mu1 = smoothing_ratios[:,t,:] * z_mu1[:,t,:] + (1 - smoothing_ratios[:,t,:]) * z_mu1[:,t-1,:]
            mu2 = smoothing_ratios[:,t,:] * z_mu2[:,t,:] + (1 - smoothing_ratios[:,t,:]) * z_mu2[:,t-1,:]
            z_mu1_smooth = z_mu1_smooth.write(t, mu1)
            z_mu2_smooth = z_mu2_smooth.write(t, mu2)
        z_mu1_smooth = tf.transpose(z_mu1_smooth.stack(), perm=[1,0,2])
        z_mu2_smooth = tf.transpose(z_mu2_smooth.stack(), perm=[1,0,2])
        # [END] Smoothing process

        # Compute the loss.
        loss1 = tf.square(z_mu1_smooth - x1_data)
        loss1 = tf.math.multiply_no_nan(loss1, mask) # Remove losses out of the sequence length. # loss1 *= mask
        loss1 = tf.reduce_sum(loss1) / tf.reduce_sum(mask)
        loss2 = tf.square(z_mu2_smooth - x2_data)
        loss2 = tf.math.multiply_no_nan(loss2, mask) # Remove losses out of the sequence length. # loss1 *= mask
        loss2 = tf.reduce_sum(loss2) / tf.reduce_sum(mask)
        loss = loss1 + loss2

        return loss

    def get_dxdy_targets_from_spatial_error(self, pred, trgt, hparams):
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod = parse_pred(pred, hparams)

        batch_size = tf.shape(z_pi)[0]
        max_seq_length = tf.shape(z_pi)[1]

        z_pi = tf.reshape(z_pi, [-1, tf.shape(z_pi)[2]])
        z_mu1 = tf.reshape(z_mu1, [-1, tf.shape(z_mu1)[2]])
        z_mu2 = tf.reshape(z_mu2, [-1, tf.shape(z_mu2)[2]])
        z_sigma1 = tf.reshape(z_sigma1, [-1, tf.shape(z_sigma1)[2]])
        z_sigma2 = tf.reshape(z_sigma2, [-1, tf.shape(z_sigma2)[2]])
        z_corr = tf.reshape(z_corr, [-1, tf.shape(z_corr)[2]])

        dx_trgt, dy_trgt, hov_trgt, eod_trgt = parse_target(trgt)
        dx_pred, dy_pred, hov_pred, eod_pred = sample_gmm_batch_tf(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod, numpy_output=False)

        dx_trgt = tf.squeeze(dx_trgt, axis=-1)
        dy_trgt = tf.squeeze(dy_trgt, axis=-1)
        dx_pred = tf.reshape(dx_pred, [batch_size, max_seq_length])
        dy_pred = tf.reshape(dy_pred, [batch_size, max_seq_length])

        x_trgt = tf.zeros((batch_size,max_seq_length))
        y_trgt = tf.zeros((batch_size,max_seq_length))
        x_pred = tf.zeros((batch_size,max_seq_length))
        y_pred = tf.zeros((batch_size,max_seq_length))

        #sum_mask_np = np.ones((batch_size,max_seq_length))
        #sum_mask = tf.ones((batch_size,max_seq_length))

        for i in range(max_seq_length):
            #sum_mask = tf.convert_to_tensor(sum_mask_np, dtype=dx_trgt.dtype)
            sum_mask = tf.concat([tf.zeros((batch_size,i)), tf.ones((batch_size,max_seq_length-i))], axis=1)
            x_trgt += sum_mask * tf.tile(dx_trgt[:,i:i+1], [1,max_seq_length])
            y_trgt += sum_mask * tf.tile(dy_trgt[:,i:i+1], [1,max_seq_length])
            x_pred += sum_mask * tf.tile(dx_pred[:,i:i+1], [1,max_seq_length])
            y_pred += sum_mask * tf.tile(dy_pred[:,i:i+1], [1,max_seq_length])
            #sum_mask_np[:,i] = 0
            #sum_mask[:,i].assign(tf.zeros(batch_size))

        dx_trgt_spatial = tf.concat([x_trgt[:,:1], x_trgt[:,1:] - x_pred[:,:-1]], axis=1)
        dy_trgt_spatial = tf.concat([y_trgt[:,:1], y_trgt[:,1:] - y_pred[:,:-1]], axis=1)

        dx_trgt_spatial = tf.stop_gradient(dx_trgt_spatial)
        dy_trgt_spatial = tf.stop_gradient(dy_trgt_spatial)

        return dx_trgt_spatial, dy_trgt_spatial

    def blend_two_dxdy(self, dx_m, dy_m, dx_s, dy_s, p_mask_m):
        if p_mask_m == 0:
            return dx_m, dy_m

        s_selector = tfd.Bernoulli(probs=p_mask_m, dtype=dx_m.dtype).sample(tf.shape(dx_m))
        m_selector = 1 - s_selector
        dx = dx_m * m_selector + dx_s * s_selector
        dy = dy_m * m_selector + dy_s * s_selector

        return dx, dy

class CorrectorLoss(tf.keras.losses.Loss):
    def __init__(self, hparams):
        super(CorrectorLoss, self).__init__()
        self.hp = hparams

    def __call__(self, pred, trgt, mask):
        '''
        - inputs_pred : Inputs of the corrector. The GMM parameters. The outputs of the controller.
        '''

        if self.hp['use_corrector_output_gmm']:
            # GMM Loss
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = parse_pred(pred, self.hp, only_gmm=True)
            x1_data, x2_data, _, _ = parse_target(trgt)

            loss0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr) #batch_nrpoints x seq_len x nrmixtures

            # implementing eq # 26 of http://arxiv.org/abs/1308.0850
            epsilon = 1e-10
            loss1 = tf.multiply(loss0, z_pi)
            loss1 = tf.reduce_sum(loss1, -1) #batch_nrpoints x seq_len
            loss1 = -tf.math.log(loss1 + epsilon) # atf the beginning, some errors are exactly zero.
            loss_gmm = tf.math.multiply_no_nan(loss1, mask) # Remove losses out of the sequence length. # loss1 *= mask
            loss_gmm = tf.reduce_sum(loss_gmm) / tf.reduce_sum(mask)
            loss = loss_gmm

        else:
            loss = tf.reduce_sum(tf.square(pred - trgt), axis=-1)
            loss = tf.math.multiply_no_nan(loss, mask) # Remove losses out of the sequence length. # loss1 *= mask
            loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

        return loss