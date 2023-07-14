# -*- coding: utf-8 -*-
from config import ConfigArgs as args
from utils import load_spectrogram
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import codecs
import data

NUM_JOBS = 8

def f(fpath):
    """
    Loads spectrograms from file paths and save as numpy

    :param f_args: spectrogram file paths and texts
    """
    mel, mag = load_spectrogram(os.path.join(args.data_path, 'wavs', f"{fpath}.wav"))
    fname = os.path.basename(fpath).replace('wav', 'npy')
    np.save(os.path.join(args.data_path, args.mel_dir, fname), mel)
    return None

def prepro(seed):
    """
    Preprocess meta data and splits them for train and test set
    """
    print('Preprocessing meta')
    meta = data.read_meta(os.path.join(args.data_path, args.meta))
    # Creates folders
    if not os.path.exists(os.path.join(args.data_path, args.mel_dir)):
        os.mkdir(os.path.join(args.data_path, args.mel_dir))

    # Creates pool
    p = Pool(NUM_JOBS)

    total_files = len(meta)
    fpaths = meta.id.values
    with tqdm(total=total_files) as pbar:
        for _ in tqdm(p.imap_unordered(f, fpaths)):
            pbar.update()
    print('Complete')

if __name__ == '__main__':
    seed = 0
    prepro(seed)
