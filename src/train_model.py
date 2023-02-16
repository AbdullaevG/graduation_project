import os
import sys
import logging
import glob
import matplotlib.pyplot as plt
import time
import copy
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from models.utils import get_emb_extractor, get_model, do_mixup, interpolate, pad_framewise_output
import torchaudio
from models.models import MyModel
from models.models import  Cnn10
from entity.train_pipeline_params import read_training_pipeline_params
import click
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    """train pipeline"""
    logger.info("Read configs for train pipeline")
    all_params = read_training_pipeline_params(config_path)
    logger.info("All parameters for model was loaded!!!")
    model_params = vars(all_params.modelparams)
    embedding_model_params = vars(all_params.embeddingmodelparams)

    logger.info("Load embedding extractor model...")
    emb_extractor = get_emb_extractor(Cnn10, **embedding_model_params)
    logger.info("Embedding extractor model was loaded!!!")

    logger.info("Try to load model...")
    model = get_model(MyModel, emb_extractor, **model_params)
    logger.info("Model was loaded!!!")


@click.command(name='train_pipeline')
@click.argument('config_path', default='../configs/train_model_config.yml')
def train_pipeline_command(config_path: str):
    """ Make start for terminal """
    train_pipeline(config_path)


if __name__ == '__main__':
    train_pipeline_command()