import os
# Keras 2 모드로 강제 실행
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
# GPU를 완전히 비활성화하고 CPU만 사용하도록 강제!
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tf2onnx
import onnx
from onnx2torch import convert
import torch

# 1. Keras model load
keras_model_path = '/home/yunjinna/DynamicMNIST/Model-LSTM-GMM/KerasMNIST/cnn.h5'
model = tf.keras.models.load_model(keras_model_path, compile=False)
#print('model input size:', model.input_shape)
# model input size: (None, 28, 28, 1)

# 2. Convert into ONNX format (tensorflow -> 범용 format)
onnx_model_path = '/home/yunjinna/DynamicMNIST/Model-Pytorch/KerasMNIST/cnn.onnx'
# 앞에 / 빼먹으면 현재 위치 아래 폴더 생김...
spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),) # (None, 28, 28, 1) 이미지 크기(28x28)에 맞게 조정 필요
output_path = onnx_model_path
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
print(f"ONNX model Saved: {onnx_model_path}")

# 3. Convert ONNX model into Pytorch model
pytorch_model = convert(onnx_model_path)

# 4. Pytorch model save (.pt)
torch.save(pytorch_model, '/home/yunjinna/DynamicMNIST/Model-Pytorch/KerasMNIST/cnn.pt')
print("Convert into Pytorch model & Saved")

'''
self.dim_names = {0:'dx',1:'dy',2:'x',3:'y', 4:'hover'}
self.dims = list(self.dim_names.values())
self.n_dims = len(self.dim_names)
'''

'''
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1775118339.598059  222489 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
I0000 00:00:1775118339.645065  222489 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1775118340.673189  222489 cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
E0000 00:00:1775118342.960447  222489 cuda_platform.cc:52] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
I0000 00:00:1775118342.960484  222489 cuda_diagnostics.cc:160] env: CUDA_VISIBLE_DEVICES="-1"
I0000 00:00:1775118342.960493  222489 cuda_diagnostics.cc:163] CUDA_VISIBLE_DEVICES is set to -1 - this hides all GPUs from CUDA
I0000 00:00:1775118342.960502  222489 cuda_diagnostics.cc:171] verbose logging is disabled. Rerun with verbose logging (usually --v=1 or --vmodule=cuda_diagnostics=1) to get more diagnostic output from this module
I0000 00:00:1775118342.960507  222489 cuda_diagnostics.cc:176] retrieving CUDA diagnostic information for host: tacolab-calc-amd
I0000 00:00:1775118342.960513  222489 cuda_diagnostics.cc:183] hostname: tacolab-calc-amd
I0000 00:00:1775118342.960578  222489 cuda_diagnostics.cc:190] libcuda reported version is: 580.126.9
I0000 00:00:1775118342.960595  222489 cuda_diagnostics.cc:194] kernel reported version is: 580.126.9
I0000 00:00:1775118342.960599  222489 cuda_diagnostics.cc:284] kernel version seems to match DSO: 580.126.9
model input size: (None, 28, 28, 1)
I0000 00:00:1775118343.159058  222489 devices.cc:67] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
I0000 00:00:1775118343.159220  222489 single_machine.cc:376] Starting new session
I0000 00:00:1775118343.294557  222489 devices.cc:67] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
I0000 00:00:1775118343.294744  222489 single_machine.cc:376] Starting new session
ONNX model Saved: home/yunjinna/DynamicMNIST/Model-Pytorch/KerasMNIST/cnn.onnx
Convert into Pytorch model & Saved
'''