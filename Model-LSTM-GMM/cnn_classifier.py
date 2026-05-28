import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import get_dot_seq_from_dx_dy
from data_utils import rasterize

class CNNClassifier:
    def __init__(self, hparams, norms):
        self.hp = hparams
        self.character_set = hparams['character_set']
        self.batch_size = hparams['batch_size_test']
        self.model_path = os.path.join('KerasMNIST', 'cnn.h5')
        self.model = load_model(self.model_path)
        self.w_img = hparams['w_img']
        self.h_img = hparams['h_img']
        self.dt = hparams['dt']
        self.stroke_width = hparams['stroke_width']

        self.dx_norm, self.dy_norm, self.d2x_norm, self.d2y_norm = norms

    def summary(self):
        self.model.summary()

    def predict(self, inputs):
        return self.model.predict(inputs)

    def classify(self, results, characters):
        np_images = list()
        char_ids = list()
        for result, character in zip(results, characters):
            dx_seq, dy_seq, hov_seq, eod_seq = result[:,0], result[:,1], result[:,2], result[:,3]

            df_v_synt = pd.DataFrame({'dx':dx_seq*self.dx_norm, 'dy':dy_seq*self.dy_norm, 'hover':hov_seq, 'eod':eod_seq})
            df_dots_synt = get_dot_seq_from_dx_dy(df_v_synt, self.w_img, self.h_img, self.dt)

            np_images_drawing = rasterize(df_dots_synt, self.w_img, self.h_img, self.dt, self.stroke_width, mnist=True)
            np_image_final = np_images_drawing[-1]

            np_images.append(np_image_final)
            char_ids.append(self.character_set.index(character))

        x_input = np.stack(np_images)
        y_trgt = np.asarray(char_ids)

        dataset = tf.data.Dataset.from_tensor_slices((x_input, y_trgt))

        y_pred_all = list()
        for x, y in dataset.batch(self.batch_size):
            y_out = self.model.predict(x, batch_size=self.batch_size, verbose=0)
            y_pred = np.argmax(y_out, axis=-1)
            y_pred_all += y_pred.tolist()
        y_pred = y_pred_all

        accuracy = accuracy_score(y_trgt, y_pred)
        precision, recall, f1score, _ = precision_recall_fscore_support(y_trgt, y_pred, beta=1)

        precisions = dict(); recalls = dict(); f1scores = dict()
        for char in self.character_set:
            precisions[char] = precision[self.character_set.index(char)]
            recalls[char] = recall[self.character_set.index(char)]
            f1scores[char] = f1score[self.character_set.index(char)]

        return accuracy, precisions, recalls, f1scores
