import os
import time
import glob
from tqdm import tqdm

import torch
import argparse
from scipy.io.wavfile import write
from melgan.model.generator import Generator
from melgan.utils.hparams import HParam, load_hparam_str
from for_melgan import infer

import random

MAX_WAV_VALUE = 32768.0

def main(args):
    torch.cuda.manual_seed(13524532)

    print("... Load trained models ...\n")
    print("     Loding checkpoint of document-level TTS model: {}".format(tts_ckpt))
    print("     Loding checkpoint of MelGAN TTS model: {}".format(args.mel_ckpt))
    start = time.time()

    mel_ckpt = torch.load(args.mel_ckpt)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(mel_ckpt['hp_str'])

    model = Generator(hp.audio.n_mel_channels).cuda()
    model.load_state_dict(mel_ckpt['model_g'])
    model.eval(inference=False)
    mel_time = time.time() - start

    print('\n... Generate waveform ...\n')
    with torch.no_grad():
        num_of_iter = args.iteration
        texts = []
        with open(args.script_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line):
                    texts.append(line)

        print("   * input text\n    {} \n".format(texts[0]))

        for i in range(num_of_iter):
            start = time.time()
            mel, length, alignments = infer(args.tts_ckpt, texts[0])

            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()

            audio = model.inference(mel)
            audio = audio.cpu().detach().numpy()
            save_path = os.path.join(args.out_dir, str(i) + '_audio.wav')
            write(save_path, hp.audio.sampling_rate, audio)
            audio_length = len(audio)/hp.audio.sampling_rate

            print("    {}. ".format(i+1))
            print("     - Path of generated audio file: {}".format(save_path))
            print("     - Length of generated audio file: {}s".format(audio_length))
            print("     - Time taken from text loading to generate spectrogram: : {}s".format(time.time() - start))
            print("     - Time taken to generate waveform: : {}s\n".format(time.time() - start + mel_time))
        print("finished generation")

if __name__ == '__main__':
    mel_ckpt = './ckpt/ckpt_melgan_sktDB_2175.pt'
    tts_ckpt = './ckpt/ckpt_tts_sktDB_69000'
    script_path = './test/1.txt'
    out_dir = './samples'

    parser = argparse.ArgumentParser()


    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-m', '--mel_ckpt', type=str, required=False, default=mel_ckpt,
                        help="path of MelGAN checkpoint pt file for evaluation")
    parser.add_argument('-t', '--tts_ckpt', type=str, required=False, default=tts_ckpt,
                        help="path of TTS checkpoint pt file for evaluation")
    parser.add_argument('-s', '--script_path', type=str, required=False, default=script_path,
                        help="path of script file for evaluation")
    parser.add_argument('-o', '--out_dir', type=str, required=False, default=out_dir,
                        help="output directory")
    parser.add_argument('-i', '--iteration', type=str, required=False, default=5,
                        help="output directory")

    args = parser.parse_args()

    main(args)
