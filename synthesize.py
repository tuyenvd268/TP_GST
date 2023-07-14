from config import ConfigArgs as args
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from hifigan.models import Generator
from hifigan.env import AttrDict
import json

import numpy as np
import pandas as pd
from model import TPGST
from data import TextDataset, text_collate_fn, load_vocab, SpeechDataset, collate_fn
import utils
import glob
from scipy.io.wavfile import write
from utils import plot_data

DEVICE = None
MAX_WAV_VALUE = 32768.0

def synthesize(model, vocoder, data_loader, batch_size=100):
    """
    To synthesize with text samples 

    :param model: nn module object
    :param data_loader: data loader
    :param batch_size: Scalar

    """
    with torch.no_grad():
        print('*'*15, ' Synthesize ', '*'*15)
        for step, (texts, _, _) in enumerate(data_loader):
            texts = texts.to(DEVICE)
            GO_frames = torch.zeros([texts.shape[0], 1, args.n_mels*args.r]).to(DEVICE)            
            mels_hat, mags_hat, A, _, _, se, _ = model(texts, GO_frames, synth=True)
            plot_data((mels_hat[0].transpose(0, 1).detach().cpu(), mags_hat[0].transpose(0, 1).detach().cpu()), -1, path=os.path.join(args.logdir, type(model).__name__, 'A', 'train'))
            
            y_g_hat = vocoder(mels_hat.transpose(1, 2))
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = 'test_generated_e2e.wav'
            write(output_file, 22050, audio)
            print(f'saved: {output_file}')
    return None

def main():
    """
    main function

    :param load_model: String. {best, latest, <model_path>}
    :param synth_mode: {'test', 'synthesize'}

    """
    assert os.path.exists(args.testset), 'Test sentence path is wrong.'

    model = TPGST().to(DEVICE)
    
    vocoder_config = AttrDict(json.load(open("hifigan/config.json")))
    vocoder = Generator(vocoder_config)
    vocoder.load_state_dict(torch.load(args.vocoder_ckpt, map_location=torch.device('cpu'))['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    
    testset = TextDataset(args.testset)
    test_loader = DataLoader(dataset=testset, batch_size=args.test_batch, drop_last=False,
                            shuffle=False, collate_fn=text_collate_fn, pin_memory=True)
    
    state = torch.load(args.infer_ckpt, map_location="cpu")
    model.load_state_dict(state['model'])
    args.global_step = state['global_step']

    print('The model is loaded. Step: {}'.format(args.global_step))

    model.eval()
    
    if not os.path.exists(os.path.join(args.sampledir, 'A')):
        os.makedirs(os.path.join(args.sampledir, 'A'))

    synthesize(model, vocoder, test_loader, args.test_batch)

if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
