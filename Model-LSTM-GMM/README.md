# Handwriting generation model based on LSTM and GMM

This repository contains a model that generates handwritten characters.
The model mainly consists of long short-term memory (LSTM) and Guassian mixture models (GMMs).

## Training a model

Use `train.py` to train a model that generates handwritten characters.
You can understand how to set arguments at `ArgumentParser` in `train.py`.
Plus, you can control hyperparamters of training using `hparams.yaml`.

## Using pre-trained models

You can use pre-trained models in `ckpt`.
You can download pre-trained models [here](http://gofile.me/6YZVo/mZFwNKLCC).
You can grasp how to import pre-trained models and generate handwritten characters in `import-trained-model.ipynb`.
