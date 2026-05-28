import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import get_dot_seq_from_dx_dy
from data_utils_transfer_revised import rasterize
#from onnx2torch import convert
'''
Keras(TF) 이미지 형태: [Batch, Height, Width, Channel] (예: 28x28 흑백이면 [B, 28, 28, 1])
PyTorch 이미지 형태: [Batch, Channel, Height, Width] (예: [B, 1, 28, 28])
'''

class CNNClassifier:
    def __init__(self, hparams, norms, device=None):
        self.hp = hparams
        self.character_set = hparams['character_set']
        self.batch_size = hparams['batch_size_test']
        
        # 기본 디바이스 설정 (GPU가 있으면 GPU 사용)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # 주의: Keras의 .h5 대신 PyTorch의 .pt 또는 .pth 파일을 로드해야 합니다.
        # TorchScript 형태로 저장된 모델이라고 가정합니다.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, 'KerasMNIST', 'cnn.pt')
        #self.model_path = os.path.join('../KerasMNIST', 'cnn.pt') # KerasMNIST/cnn.pt
        #print(self.model_path) /home/yunjinna/DynamicMNIST/Model-Pytorch/KerasMNIST/cnn.pt
        
        try:
            #self.model = torch.jit.load(self.model_path).to(self.device)
            #self.model = convert(self.model_path).to(device)
            # ONNX convert 대신, PyTorch 기본 함수로 읽어옴 + 안전한 파일이니까, 구조체까지 다 읽어오라고 허락
            self.model = torch.load(self.model_path, weights_only=False).to(device)
            self.model.eval()
        except Exception as e:
            print(f"[경고] PyTorch CNN 모델 로드 실패: {e}")
            print("Keras의 cnn.h5를 PyTorch 모델로 변환하거나 새로 학습한 cnn.pt 파일이 필요합니다.")
            self.model = None

        self.w_img = hparams['w_img']
        self.h_img = hparams['h_img']
        self.dt = hparams['dt']
        self.stroke_width = hparams['stroke_width']

        self.dx_norm, self.dy_norm, self.d2x_norm, self.d2y_norm = norms

    def summary(self):
        if self.model:
            print(self.model)
        else:
            print("로드된 모델이 없습니다.")

    def predict(self, inputs):
        if self.model is None:
            return None
            
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            
        with torch.no_grad():
            return self.model(inputs).cpu().numpy()

    def classify(self, results, characters):
        if self.model is None:
            print("[에러] 분류 모델이 없어 평가를 건너뜁니다.")
            return 0.0, {}, {}, {}

        np_images = list()
        char_ids = list()
        for result, character in zip(results, characters):
            dx_seq, dy_seq, hov_seq, eod_seq = result[:,0], result[:,1], result[:,2], result[:,3]

            df_v_synt = pd.DataFrame({'dx':dx_seq*self.dx_norm, 'dy':dy_seq*self.dy_norm, 'hover':hov_seq, 'eod':eod_seq})
            df_dots_synt = get_dot_seq_from_dx_dy(df_v_synt, self.w_img, self.h_img, self.dt)

            np_images_drawing = rasterize(df_dots_synt, self.w_img, self.h_img, self.dt, self.stroke_width, mnist=True)
            np_image_final = np_images_drawing[-1] # 완성된 최종 이미지

            np_images.append(np_image_final)
            char_ids.append(self.character_set.index(character))

        x_input = np.stack(np_images)
        y_trgt = np.asarray(char_ids)

        # PyTorch Dataset & DataLoader 생성
        tensor_x = torch.tensor(x_input, dtype=torch.float32)
        tensor_y = torch.tensor(y_trgt, dtype=torch.long)
        
        # [참고] Keras 기반 이미지가 (Batch, H, W, Channel) 구조였다면, 
        # PyTorch는 (Batch, Channel, H, W) 구조를 쓰므로 차원 변경이 필요할 수 있습니다.
        # 만약 에러가 난다면 아래 주석 해제
        # if len(tensor_x.shape) == 4 and tensor_x.shape[-1] in [1, 3]:
        #     tensor_x = tensor_x.permute(0, 3, 1, 2)

        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        y_pred_all = list()
        
        with torch.no_grad(): # 추론 모드 (그래디언트 계산 비활성화)
            for x, _ in dataloader:
                x = x.to(self.device)
                # 모델을 x가 있는 디바이스(GPU)로 확실하게 보내줘야 함
                self.model = self.model.to(x.device)
                y_out = self.model(x)
                y_pred = torch.argmax(y_out, dim=-1).cpu().numpy()
                y_pred_all.extend(y_pred.tolist())
                
        y_pred = y_pred_all

        accuracy = accuracy_score(y_trgt, y_pred)
        # zero_division=0 을 추가하여 특정 클래스가 배치에 없을 때 발생하는 경고를 방지합니다.
        precision, recall, f1score, _ = precision_recall_fscore_support(y_trgt, y_pred, beta=1, zero_division=0)

        precisions = dict(); recalls = dict(); f1scores = dict()
        for char in self.character_set:
            idx = self.character_set.index(char)
            # scikit-learn 결과 배열의 길이 방어 로직 추가
            if idx < len(precision):
                precisions[char] = precision[idx]
                recalls[char] = recall[idx]
                f1scores[char] = f1score[idx]
            else:
                precisions[char], recalls[char], f1scores[char] = 0.0, 0.0, 0.0

        return accuracy, precisions, recalls, f1scores