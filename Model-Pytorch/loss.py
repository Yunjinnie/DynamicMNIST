import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import math

# utils.py에 포팅되어야 할 헬퍼 함수들
from utils import pt_2d_normal, parse_pred, parse_target_pt, sample_gmm_batch_pt

class Loss(nn.Module):
    def __init__(self, hparams):
        super(Loss, self).__init__()
        self.hp = hparams
        self.cnt_sp = 0.0 # 학습 시 단순 카운터이므로 tensor일 필요 없음

    def forward(self, pred, trgt, mask, smoothing_ratios, use_spatial_error=False):
        # 1. Prediction & Target 파싱
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod = parse_pred(pred, self.hp)
        x1_data, x2_data, hov_data, eod_data = parse_target_pt(trgt)

        # 2. Spatial Error 타겟 보정 (선택)
        if use_spatial_error:
            dx_trgt_sp, dy_trgt_sp = self.get_dxdy_targets_from_spatial_error(pred, trgt)
            dx_trgt_sp, dy_trgt_sp = dx_trgt_sp.unsqueeze(-1), dy_trgt_sp.unsqueeze(-1)
            x1_data, x2_data = self.blend_two_dxdy(x1_data, x2_data, dx_trgt_sp, dy_trgt_sp, self.hp['p_spatial_error'])
            self.cnt_sp += 1.0

        mask_sum = torch.sum(mask) + 1e-8 # Division by zero 방지

        # 3. GMM Loss (Negative Log-Likelihood)
        prob_density = pt_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        gmm_prob = torch.sum(prob_density * z_pi, dim=-1) # [batch, seq_len]
        loss1 = -torch.log(gmm_prob + 1e-10) 
        
        # Masking: mask.bool()을 사용하여 NaN을 0으로 안전하게 치환
        loss_gmm = torch.where(mask.bool(), loss1, torch.zeros_like(loss1))
        loss_gmm = torch.sum(loss_gmm) / mask_sum

        # 4. Hover & EOD Loss (수동 구현 제거 -> 안정적인 PyTorch 내장 BCE 사용)
        z_hov = z_hov.squeeze(-1)
        hov_data = hov_data.squeeze(-1)
        hov_data = torch.clamp(hov_data, 0.0, 1.0)
        loss_hov_raw = F.binary_cross_entropy(z_hov, hov_data, reduction='none') ###
        loss_hov = torch.where(mask.squeeze(-1).bool(), loss_hov_raw, torch.zeros_like(loss_hov_raw))
        loss_hov = torch.sum(loss_hov) / mask_sum

        z_eod = z_eod.squeeze(-1)
        eod_data = eod_data.squeeze(-1)
        eod_data = torch.clamp(eod_data, 0.0, 1.0)
        loss_eod_raw = F.binary_cross_entropy(z_eod, eod_data, reduction='none') ###
        loss_eod = torch.where(mask.squeeze(-1).bool(), loss_eod_raw, torch.zeros_like(loss_eod_raw))
        loss_eod = torch.sum(loss_eod) / mask_sum

        # 5. Smoothing Loss
        loss_smoothing = self.get_smoothing_loss(z_mu1, z_mu2, x1_data, x2_data, smoothing_ratios, mask, mask_sum)

        loss_total = loss_gmm + loss_hov + loss_eod + loss_smoothing

        return loss_total, (loss_gmm, loss_hov, loss_eod, loss_smoothing)

    def get_smoothing_loss(self, z_mu1, z_mu2, x1_data, x2_data, smoothing_ratios, mask, mask_sum):
        """ TensorArray 연산을 파이썬 리스트와 torch.stack으로 교체하여 성능 극대화 """
        max_seq_length = z_mu1.size(1)

        mu1_smooth = [z_mu1[:, 0, :]]
        mu2_smooth = [z_mu2[:, 0, :]]

        # 순차적 스무딩
        for t in range(1, max_seq_length):
            #mu1_t = smoothing_ratios[:, t, :] * z_mu1[:, t, :] + (1 - smoothing_ratios[:, t, :]) * z_mu1[:, t-1, :]
            # smoothing_ratios가 1차원(고정값)일 때와 3차원(동적)일 때를 모두 지원하게 바꿈
            if smoothing_ratios.dim() == 1:
                sr = smoothing_ratios  # 1차원이면 자르지 않고 브로드캐스팅(자동 확장) 시킴
            else:
                sr = smoothing_ratios[:, t, :] # 3차원이면 원래대로 시간(t) 축 자름

            # mu1_t = smoothing_ratios[:, t, :] * z_mu1[:, t, :] + (1 - smoothing_ratios[:, t, :]) * z_mu1[:, t-1, :]
            mu1_t = sr * z_mu1[:, t, :] + (1 - sr) * z_mu1[:, t-1, :]
            #mu2_t = smoothing_ratios[:, t, :] * z_mu2[:, t, :] + (1 - smoothing_ratios[:, t, :]) * z_mu2[:, t-1, :]
            mu2_t = sr * z_mu2[:, t, :] + (1 - sr) * z_mu2[:, t-1, :]
            mu1_smooth.append(mu1_t)
            mu2_smooth.append(mu2_t)

        z_mu1_smooth = torch.stack(mu1_smooth, dim=1) # [batch, seq_len, n_gmm]
        z_mu2_smooth = torch.stack(mu2_smooth, dim=1)

        loss1 = (z_mu1_smooth - x1_data)**2
        loss2 = (z_mu2_smooth - x2_data)**2
        
        # mask를 boolean 형태로 먼저 변환하고
        mask_b = mask.bool()
        # print(mask_b.dim()) # 2
        # print(loss1.size(-1)) # 17
        # print(mask_b.size(-1)) # 51

        # mask의 마지막 차원 크기가 loss1과 다르므로, 마지막 차원을 1로 잘라내서 자동 확장이 되게 만들어야 함
        if mask_b.dim() == 2: # and mask_b.size(-1) != loss1.size(-1):
            # 2차원(Batch, 51)이므로, 끝에 차원을 하나 추가해 3차원(Batch, 51, 1)으로 만듦
            mask_b = mask_b.unsqueeze(-1) 
            # (Batch, Time, 17) -> (Batch, Time, 1)로 변환 xxx 

        # loss1 = torch.where(mask.bool(), loss1, torch.zeros_like(loss1))
        # 다듬어진 mask_b 적용 -> loss1 (Batch, 51, 17)과 mask_b (Batch, 51, 1)의 아다리 완벽하게 맞음
        loss1 = torch.where(mask_b, loss1, torch.zeros_like(loss1))

        # loss2도 mask.bool()을 사용하고 있다면 거기도 전부 mask.bool() 대신 방금 만든 mask_b 로 바꿔야함!
        loss2 = torch.where(mask_b, loss2, torch.zeros_like(loss2))
        #loss2 = torch.where(mask.bool(), loss2, torch.zeros_like(loss2))

        return (torch.sum(loss1) + torch.sum(loss2)) / mask_sum

    def get_dxdy_targets_from_spatial_error(self, pred, trgt):
        """ 비효율적인 for-loop를 O(1) 텐서 CumSum으로 완벽히 치환 (제거 명시 부분) """
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod = parse_pred(pred, self.hp)
        dx_trgt, dy_trgt, _, _ = parse_target_pt(trgt)
        
        dx_pred, dy_pred, _, _ = sample_gmm_batch_pt(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_hov, z_eod)

        dx_trgt = dx_trgt.squeeze(-1)
        dy_trgt = dy_trgt.squeeze(-1)
        
        # 핵심 최적화: torch.cumsum을 이용해 궤적의 절대 좌표(x, y)를 단번에 계산
        x_trgt = torch.cumsum(dx_trgt, dim=1)
        y_trgt = torch.cumsum(dy_trgt, dim=1)
        x_pred = torch.cumsum(dx_pred, dim=1)
        y_pred = torch.cumsum(dy_pred, dim=1)

        # Spatial target 보정식: t=0일 때는 원래 dx_trgt, t>0일 때는 타겟 누적치에서 모델 누적 예측치를 뺌
        dx_trgt_spatial = torch.cat([x_trgt[:, :1], x_trgt[:, 1:] - x_pred[:, :-1]], dim=1)
        dy_trgt_spatial = torch.cat([y_trgt[:, :1], y_trgt[:, 1:] - y_pred[:, :-1]], dim=1)

        return dx_trgt_spatial.detach(), dy_trgt_spatial.detach()

    def blend_two_dxdy(self, dx_m, dy_m, dx_s, dy_s, p_mask_m):
        if p_mask_m == 0:
            return dx_m, dy_m

        # Bernoulli 마스크 샘플링
        s_selector = torch.bernoulli(torch.full_like(dx_m, p_mask_m))
        m_selector = 1.0 - s_selector
        
        dx = dx_m * m_selector + dx_s * s_selector
        dy = dy_m * m_selector + dy_s * s_selector

        return dx, dy

class CorrectorLoss(nn.Module):
    def __init__(self, hparams):
        super(CorrectorLoss, self).__init__()
        self.hp = hparams

    def forward(self, pred, trgt, mask):
        mask_sum = torch.sum(mask) + 1e-8

        if self.hp['use_corrector_output_gmm']:
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, _, _ = parse_pred(pred, self.hp)
            x1_data, x2_data, _, _ = parse_target_pt(trgt)

            prob_density = pt_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
            gmm_prob = torch.sum(prob_density * z_pi, dim=-1)
            loss_raw = -torch.log(gmm_prob + 1e-10)
            
            loss = torch.where(mask.squeeze(-1).bool(), loss_raw, torch.zeros_like(loss_raw))
            return torch.sum(loss) / mask_sum

        else:
            # MSE Loss for Direct Coordinate Prediction
            loss_raw = torch.sum((pred - trgt)**2, dim=-1)
            loss = torch.where(mask.squeeze(-1).bool(), loss_raw, torch.zeros_like(loss_raw))
            return torch.sum(loss) / mask_sum