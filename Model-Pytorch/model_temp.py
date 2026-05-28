import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from data_utils import get_character_vector  # 토치 텐서를 반환하도록 수정
from utils import sample_gmm_batch, sample_gmm_batch_pt, parse_pred
from loss import Loss, CorrectorLoss

class Model(nn.Module):
    def __init__(self, hparams, data_loader):
        super(Model, self).__init__()

        self.hp = hparams
        self.dl = data_loader

        # Buffers for non-trainable constants (replaces tf.Variable(trainable=False))
        # 학습되지 않는 고정/상태 변수는 register_buffer로 등록
        self.register_buffer('epochs', torch.tensor(0))
        self.register_buffer('iterations', torch.tensor(0))
        self.register_buffer('dx_norm', torch.tensor(1.0))
        self.register_buffer('dy_norm', torch.tensor(1.0))
        self.register_buffer('d2x_norm', torch.tensor(1.0))
        self.register_buffer('d2y_norm', torch.tensor(1.0))
        
        # 내부 디바이스 추적용 (infer_batch 등에서 사용)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.g_mixtures_dim = hparams['g_mixtures_dim']
        self.output_dim = hparams['output_dim']
        self.use_spatial_error = self.hp['p_spatial_error'] != 0

        # Initialize the model ==================================================================
        self.use_lstm = hparams['use_lstm']
        self.use_layernorm = hparams['use_layernorm']

        # Note: TF의 LayerNormLSTMCell과 완벽히 동일한 동작을 원한다면 커스텀 Cell이 필요함
        # 여기서는 논문 베이스라인 구성을 위해 표준 LSTM/RNN을 사용
        if self.use_lstm:
            self.rnn1 = nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_dim, batch_first=True, dropout=hparams['lstm_r_dropout'] if hparams['lstm_r_dropout'] > 0 else 0)
            self.rnn2 = nn.LSTM(input_size=self.lstm_dim, hidden_size=self.lstm_dim, batch_first=True, dropout=hparams['lstm_r_dropout'] if hparams['lstm_r_dropout'] > 0 else 0)
        else:
            self.rnn1 = nn.RNN(input_size=self.input_dim, hidden_size=self.lstm_dim, batch_first=True, dropout=hparams['lstm_r_dropout'] if hparams['lstm_r_dropout'] > 0 else 0)
            self.rnn2 = nn.RNN(input_size=self.lstm_dim, hidden_size=self.lstm_dim, batch_first=True, dropout=hparams['lstm_r_dropout'] if hparams['lstm_r_dropout'] > 0 else 0)

        # PyTorch에서는 Weight Decay로 L2 Reg를 처리하므로 레이어 선언 시 L2가 들어가지 않음
        self.dense1 = nn.Linear(self.lstm_dim, self.output_dim)

        self.smoothing_net = SmoothingNet(hparams)
        self.corrector = Corrector(hparams)

        # Initialize the loss ==================================================================
        self.loss_fn = Loss(hparams)

        # Initialize the optimizer ==================================================================
        # L2 정규화는 optimizer의 weight_decay 파라미터로 처리합니다.
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, eps=hparams['epsilon'], weight_decay=self.l2_reg_coef)
        # default: Adam
        self.lr_schedule = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0) # decay_rate 1.0 (No decay), default: 0.1

        self.set_norms(self.dl.dx_norm, self.dl.dy_norm, self.dl.d2x_norm, self.dl.d2y_norm)


    def set_norms(self, dx_norm, dy_norm, d2x_norm, d2y_norm):
        # register_buffer로 등록된 텐서 업데이트
        self.dx_norm.fill_(dx_norm)
        self.dy_norm.fill_(dy_norm)
        self.d2x_norm.fill_(d2x_norm)
        self.d2y_norm.fill_(d2y_norm)

    def forward(self, inputs, states=None, one_step=False, bias=0.0):
        if (not one_step) or (states is None):
            states1, states2 = None, None
        else:
            states1, states2 = states

        # inputs: [batch_size, seq_len, input_dim]
        if self.use_lstm:
            h_seqs, states1 = self.rnn1(inputs, states1)
            h_seqs, states2 = self.rnn2(h_seqs, states2)
        else:
            h_seqs, states1 = self.rnn1(inputs, states1)
            h_seqs, states2 = self.rnn2(h_seqs, states2)

        gmm_logits = self.dense1(h_seqs)
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod = parse_pred(gmm_logits, self.hp)
        
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = self.gmm_layer(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, bias)
        z_hov = torch.sigmoid(z_hov)
        z_eod = torch.sigmoid(z_eod)

        outputs = torch.cat([z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod], dim=-1)
        
        states = (states1, states2)
        smoothing_ratios = self.smoothing_net(inputs)

        return outputs, states, h_seqs, smoothing_ratios
    
    def gmm_layer(self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, bias=0):
        # Numerical stabilization for pi
        max_pi, _ = torch.max(z_pi, dim=-1, keepdim=True)
        z_pi = z_pi - max_pi
        z_pi = torch.exp(z_pi * (1 + bias))
        normalize_pi = 1.0 / torch.sum(z_pi, dim=-1, keepdim=True)
        z_pi = normalize_pi * z_pi

        z_sigma1 = torch.exp(z_sigma1 * (1 + bias))
        z_sigma2 = torch.exp(z_sigma2 * (1 + bias))
        z_corr = torch.tanh(z_corr)
        z_corr = 0.95 * z_corr

        return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr

    def train_step(self, x, y, mask):
        self.train()
        self.optimizer.zero_grad()
        
        y_pred, states, h_seqs, smoothing_ratios = self(x)
        loss, sub_losses = self.loss_fn(y_pred, y, mask, smoothing_ratios, use_spatial_error=self.use_spatial_error)
        
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=self.max_grad_norm)
        self.optimizer.step()
        # self.lr_schedule.step() # 필요시 활성화

        return y_pred, states, h_seqs, smoothing_ratios, loss, sub_losses

    def test_step(self, x, y, mask):
        self.eval()
        with torch.no_grad():
            y_pred, states, h_seqs, smoothing_ratios = self(x)
            loss, sub_losses = self.loss_fn(y_pred, y, mask, smoothing_ratios)

        return y_pred, states, h_seqs, smoothing_ratios, loss, sub_losses

    def one_step(self, inputs, states, bias=0):
        outputs, states, h_seqs, smooth_ratios = self(inputs, states, one_step=True, bias=bias)
        return outputs, states, h_seqs, smooth_ratios
    
    # inference logic (추론 파이프라인 텐서화 & 최적화)
    # @torch.no_grad()
    def infer_batch(self, character, batch_size, eod_stop = False, eod_slice=False, bias = 0, tf_sampling=False, return_preds=False, use_corrector=True, null_correction=False):
        # device='cuda'?
        """
        NumPy와 혼재되었던 코드를 PyTorch 텐서 연산으로 치환하여 병목 제거
        Queue 방식은 collections.deque를 사용하여 O(1) 복잡도로 만듦
        """
        self.eval()
        i_step = 0
        # RNN 및 Corrector State
        states = None # Zero vectors for initial hidden and cell states.
        cr_states = None # Zero vectors for initial hidden and cell states.

        # 1. 초기 변수 설정
        # (data_utils.get_character_vector가 PyTorch 텐서를 반환한다고 가정)
        character_vector = get_character_vector(character, self.character_set)
        character_vector_batch = np.repeat(np.expand_dims(character_vector, 0), batch_size, axis=0)

        first_in_vector = np.concatenate((np.zeros(self.input_dim - self.v_character_dim), character_vector))
        # expand_dims
        first_in_vector = np.repeat(first_in_vector[None, None, :], batch_size, axis=0) # [batch_size, char_dim]
        
        # 입력 텐서 구성: [dx, dy, hov, eod, ... char_vec ...] 
        # zeros for previous velocity, hovering label, and character ending label; plus, one-hot character vector.
        inputs = torch.tensor(first_in_vector, dtype=torch.float32, device=self.device)
        # If delayed_steps == 0, then this list will be an empty list.
        delayed_inputs = [torch.tensor(first_in_vector, dtype=torch.float32, device=self.device) for _ in range(self.delayed_steps_ctrl_test)]

        dx_seq, dy_seq, hov_seq, eod_seq, y_preds = [], [], [], [], []
        seq_lengths = np.full(batch_size, self.max_steps)
        eod_marker = np.zeros(batch_size, dtype=np.int64)

        if use_corrector:
            cr_sensory_input_dim = 3 # [var_x, var_y, hov]
            cr_dx_seq, cr_dy_seq, rnn_dx_seq, rnn_dy_seq = [], [], [], []
            empty_cr_inputs = np.concatenate((np.zeros(cr_sensory_input_dim), character_vector))
            empty_cr_inputs = np.repeat(empty_cr_inputs[None, None, :], batch_size, axis=0)
            delayed_cr_inputs = [torch.tensor(empty_cr_inputs, dtype=torch.float32, device=self.device) for _ in range(self.delayed_steps + 1)]

        '''
        # Kinematics State Dictionary (인스턴스 오염 방지)
        kine_states = {
            'x': torch.zeros(batch_size, device=device),
            'y': torch.zeros(batch_size, device=device),
            'dx': torch.zeros(batch_size, device=device),
            'dy': torch.zeros(batch_size, device=device),
            'dx_prev': torch.zeros(batch_size, device=device),
            'dy_prev': torch.zeros(batch_size, device=device)
        }
        norms = (self.dx_norm, self.dy_norm, self.d2x_norm, self.d2y_norm)
        
        
        # 결과 저장용 리스트
        dx_seq, dy_seq, hov_seq, eod_seq = [], [], [], []
        seq_lengths = torch.full((batch_size,), self.hp['max_steps'], dtype=torch.long, device=device)
        eod_marker = torch.zeros(batch_size, dtype=torch.long, device=device)

        prev_var_x, prev_var_y = torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device)
        '''

        # 2. Autoregressive 생성 루프 시작
        with torch.no_grad():
            while i_step < self.max_steps:
                # RNN 1-Step 추론
                y_pred, states, h_seqs, smoothing_ratios = self.one_step(inputs, states, bias)
                # Squeeze the time dimension.
                y_pred = torch.squeeze(y_pred, dim=1) # [batch_size, out_dim]

                # TF/Numpy 대응 함수로 분기
                # 예측 파싱 및 GMM 샘플링
                if tf_sampling:
                    next_var_x, next_var_y, hov, eod = sample_gmm_batch_pt(*parse_pred(y_pred, self.hp))
                else:
                    # Numpy 기반 함수라면 CPU로 내려야 할 수 있음... utils 구현에 따라 다름
                    y_pred_np = y_pred.cpu().numpy()
                    next_var_x, next_var_y, hov, eod = sample_gmm_batch(*parse_pred(y_pred_np, self.hp))
            
                # Kinematics 연산 등 원본 코드 로직과 동일
                # 원본 논문의 Corrector 및 입력 마스킹, Sensory Feedback 로직이 동일하게 위치함

                # --- Corrector 로직 ---
                # Collect RNN's outputs without correction.
                if use_corrector:
                    next_x, next_y, next_dx, next_dy, next_d2x, next_d2y = self.get_kinematics(next_var_x, next_var_y, i_step)
                    rnn_dx_seq.append(next_dx)
                    rnn_dy_seq.append(next_dy)
                    
                if use_corrector and self.hp['correction_output'] and i_step >= self.hp['delayed_steps'] + 1:
                    cr_in = delayed_cr_inputs.popleft()
                    if self.hp['use_corrector_input_gmm_params']:
                        cr_in = torch.cat([cr_in, h_seqs], dim=-1)
                    
                    var_xy = torch.stack([next_var_x, next_var_y], dim=-1).unsqueeze(1)
                    cr_in_full = torch.cat([var_xy, cr_in], dim=-1)
                    
                    cr_corrections, cr_states = self.corrector(cr_in_full, cr_states)
                    
                    # 보정치 더하기 (GMM 여부에 따른 처리 간략화)
                    if not self.hp['use_corrector_output_gmm']:
                        cr_dx, cr_dy = cr_corrections[..., 0].squeeze(), cr_corrections[..., 1].squeeze()
                        next_var_x = next_var_x + cr_dx
                        next_var_y = next_var_y + cr_dy

                # --- Smoothing 로직 ---
                if i_step > 0:
                    smooth_ratios = smooth_ratios.squeeze()
                    next_var_x = smooth_ratios * next_var_x + (1 - smooth_ratios) * prev_var_x
                    next_var_y = smooth_ratios * next_var_y + (1 - smooth_ratios) * prev_var_y

                # --- Kinematics 계산 (상태 저장 안함) ---
                nxt_x, nxt_y, nxt_dx, nxt_dy, nxt_d2x, nxt_d2y, kine_states = self.get_kinematics_pt(
                    self.target, next_var_x, next_var_y, kine_states, norms
                )
                # End of this Step
                i_step +=1

            # 결과 기록
            dx_seq.append(nxt_dx)
            dy_seq.append(nxt_dy)
            hov_seq.append(hov)
            eod_seq.append(eod)

            # --- Masking 및 다음 Input 준비 ---
            p_mask = self.hp['p_mask_input_dxdy_test']
            mask = torch.bernoulli(torch.full_like(nxt_dx, 1 - p_mask)).unsqueeze(-1)
            
            input_kine_list = []
            if 'p' in self.input_kine: input_kine_list.extend([nxt_x * mask.squeeze(), nxt_y * mask.squeeze()])
            if 'v' in self.input_kine: input_kine_list.extend([nxt_dx * mask.squeeze(), nxt_dy * mask.squeeze()])
            if 'a' in self.input_kine: input_kine_list.extend([nxt_d2x * mask.squeeze(), nxt_d2y * mask.squeeze()])
            input_kine_list.extend([hov, eod])
            
            next_kine = torch.stack(input_kine_list, dim=-1) # [batch_size, kine_dim]
            next_input = torch.cat([next_kine, char_vec_batch], dim=-1).unsqueeze(1)
            
            delayed_inputs.append(next_input)
            inputs = delayed_inputs.popleft()

            if use_corrector:
                nxt_var_masked = torch.stack([next_var_x * mask.squeeze(), next_var_y * mask.squeeze(), hov], dim=-1)
                next_cr_input = torch.cat([nxt_var_masked, char_vec_batch], dim=-1).unsqueeze(1)
                delayed_cr_inputs.append(next_cr_input)

            prev_var_x, prev_var_y = next_var_x, next_var_y

            # --- EOD 처리 (조기 종료 로직) ---
            eod_detected = (eod == 1.0) & (eod_marker == 0)
            seq_lengths[eod_detected] = i_step + 1
            eod_marker[eod_detected] = 1
            
            if eod_stop and eod_marker.sum() == batch_size:
                break

        # 3. 리스트를 텐서로 통합 [batch_size, seq_len]
        dx_seq = torch.stack(dx_seq, dim=1)
        dy_seq = torch.stack(dy_seq, dim=1)
        hov_seq = torch.stack(hov_seq, dim=1)
        eod_seq = torch.stack(eod_seq, dim=1)

        results = torch.stack([dx_seq, dy_seq, hov_seq, eod_seq], dim=-1)

        return results

    def get_kinematics(self, target, var_x, var_y, i_step): #, states, norms):
        """
        클래스 내부에 값을 저장하지 않고 상태 비의존성(Stateless)을 보장하는 순수 함수 버전의 Kinematics 계산 로직
        - states: 궤적의 현재 상태를 담은 딕셔너리 (x, y, dx_prev, dy_prev, dx, dy 등)
        원본 코드: 모델 클래스(self) 내부에 궤적의 현재 상태를 직접 저장하고 업데이트
        -> 미니배치 처리를 하거나 모델을 병렬로 돌릴 때, 내부 변수(self.x)가 꼬이면서 다른 배치의 데이터가 섞여버림
        """
        dx_norm, dy_norm, d2x_norm, d2y_norm = norms

        if target == 'v':
            dx = var_x * dx_norm
            dy = var_y * dy_norm
            x = states['x'] + dx
            y = states['y'] + dy
            d2x = dx - states['dx_prev']
            d2y = dy - states['dy_prev']
            
            states['dx_prev'] = dx
            states['dy_prev'] = dy
        elif target == 'a':
            d2x = var_x * d2x_norm
            d2y = var_y * d2y_norm
            dx = states['dx'] + d2x
            dy = states['dy'] + d2y
            x = states['x'] + dx
            y = states['y'] + dy
            
            states['dx'] = dx
            states['dy'] = dy
        else:
            raise ValueError("Target must be 'v' or 'a'")

        states['x'] = x
        states['y'] = y

        # 정규화된 값 반환
        norm_d2x = d2x / d2x_norm
        norm_d2y = d2y / d2y_norm
        norm_dx = dx / dx_norm
        norm_dy = dy / dy_norm
        
        # dl.normalize_position에 해당하는 로직 적용 필요 (여기서는 생략하고 x, y 반환)
        norm_x, norm_y = x, y 

        return norm_x, norm_y, norm_dx, norm_dy, norm_d2x, norm_d2y, states


class SmoothingNet(nn.Module):
    def __init__(self, hparams):
        super(SmoothingNet, self).__init__()

        self.hp = hparams
        self.use_trainable_smooth_ratio = hparams['use_trainable_smooth_ratio']
        self.use_static_trainable_smooth_ratio = hparams['use_static_trainable_smooth_ratio']
        self.smooth_net_dim = hparams['smooth_net_dim']
        self.output_dim = 1
        
        if not self.use_trainable_smooth_ratio:
            self.register_buffer('static_smooth_ratio', torch.tensor(hparams['smooth_dxdy_ratio'], dtype=torch.float32))
        elif self.use_static_trainable_smooth_ratio:
            self.static_smooth_ratio = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        else:
            self.rnn = nn.LSTM(input_size=hparams['output_dim'], hidden_size=self.smooth_net_dim, batch_first=True)
            self.dense = nn.Linear(self.smooth_net_dim, self.output_dim)

    def forward(self, inputs):
        if not self.use_trainable_smooth_ratio:
            smooth_ratio = self.static_smooth_ratio_variable
        elif self.use_static_trainable_smooth_ratio:
            smooth_ratio = torch.sigmoid(self.static_smooth_ratio_variable)
        else:
            inputs = inputs.detach() # Stop gradient via detach()
            h_seq, _ = self.rnn(inputs)
            smooth_ratio = torch.sigmoid(self.dense(h_seq))
        return smooth_ratio

class Corrector(nn.Module):
    def __init__(self, hparams):
        super(Corrector, self).__init__()
        self.hp = hparams
        self.use_rnn = hparams['use_rnn_corrector']
        self.dim_rnn = 32
        self.dim_dense1 = self.dim_rnn
        self.dim_output = hparams['g_mixtures_dim'] if self.hp['use_corrector_output_gmm'] else 2

        # RNN or FFN
        # Input dim needs to match Corrector input
        if self.use_rnn:
            # 입력 차원은 Corrector의 parse_inputs 구성에 따라 결정 (예: 2 + character_dim 등)
            input_dim = 3 + hparams['v_character_dim'] # 추정치, 실제 데이터 형태에 맞춰 수정 필요
            self.rnn = nn.LSTM(input_dim, self.dim_rnn, batch_first=True)
        else:
            input_dim = 3 + hparams['v_character_dim']
            self.ffn = nn.Linear(input_dim, 4 * self.dim_rnn)
            self.do0 = nn.Dropout(0.5)
            
        self.dense1 = nn.Linear(self.dim_rnn if self.use_rnn else 4 * self.dim_rnn, self.dim_dense1)
        self.do1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(self.dim_dense1, self.dim_output)

        self.loss_fn = CorrectorLoss(hparams)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hparams['learning_rate'], eps=hparams['epsilon'], weight_decay=hparams['l2_reg_coef'])

    def forward(self, inputs, states=None, one_step=False):
        var_x_seqs, var_y_seqs, _ = self.parse_inputs(inputs)
        inputs = inputs.detach() # tf.stop_gradient

        if not one_step:
            states = None

        if self.use_rnn:
            h_seq, states = self.rnn(inputs, states)
        else:
            h_seq = self.do0(self.ffn(inputs))

        h_seq2 = self.do1(F.relu(self.dense1(h_seq)))
        corrections = self.dense2(h_seq2)

        if self.hp['use_corrector_output_gmm']:
            corrections = self.gmm_layer(corrections)

        return corrections, corrections, states, (var_x_seqs, var_y_seqs)

    def gmm_layer(self, corrections, bias=0):
        # Model 클래스의 로직과 동일하므로 생략 없이 PyTorch 문법 적용
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, _, _ = parse_pred(corrections, self.hp, only_gmm=True)
        # GMM layer 계산 (Model.gmm_layer와 동일)
        max_pi, _ = torch.max(z_pi, dim=-1, keepdim=True)
        z_pi = z_pi - max_pi
        z_pi = torch.exp(z_pi * (1 + bias))
        z_pi = (1.0 / torch.sum(z_pi, dim=-1, keepdim=True)) * z_pi

        z_sigma1 = torch.exp(z_sigma1 * (1 + bias))
        z_sigma2 = torch.exp(z_sigma2 * (1 + bias))
        z_corr = 0.95 * torch.tanh(z_corr)

        return torch.cat([z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr], dim=-1)

    def train_step(self, x, y, mask):
        self.train()
        self.optimizer.zero_grad()
        
        y_pred, corrections, states, _ = self(x)
        loss = self.loss_fn(corrections, y, mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=self.hp['max_grad_norm'])
        self.optimizer.step()

        return y_pred, loss

    def test_step(self, x, y, mask):
        self.eval()
        with torch.no_grad():
            y_pred, corrections, states, _ = self(x)
            loss = self.loss_fn(corrections, y, mask)
        return y_pred, loss

    def parse_inputs(self, inputs):
        var_x_seqs = inputs[..., 0:1]
        var_y_seqs = inputs[..., 1:2]
        cr_inputs = inputs[..., 2:]
        return var_x_seqs, var_y_seqs, cr_inputs