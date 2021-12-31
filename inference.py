import os
import time

import torch
import argparse
from scipy.io.wavfile import write
from melgan.model.generator import Generator
from melgan.utils.hparams import HParam, load_hparam_str
from inferref.for_melgan import infer
from inferref.utils import load_wav_to_torch
from inferref.layers import TacotronSTFT
MAX_WAV_VALUE = 32768.0

GPU_NUM = 1  # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

def load_mel(path, hparams):
    stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)

    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.to(device)
    return melspec


def main(args, ref_audio):
    torch.cuda.manual_seed(13524532)

    print("... Load trained models ...\n")
    print("     Loding checkpoint of document-level TTS model: {}".format(tts_ckpt))
    print("     Loding checkpoint of MelGAN TTS model: {}".format(args.mel_ckpt))
    start = time.time()

    ''' mel ckpt '''
    mel_ckpt = torch.load(args.mel_ckpt)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(mel_ckpt['hp_str'])

    hp.max_decoder_steps = 8000
    model = Generator(hp.audio.n_mel_channels).to(device)
    model.load_state_dict(mel_ckpt['model_g'])
    model.eval(inference=False)
    mel_time = time.time() - start

    print('\n... Generate waveform ...\n')
    with torch.no_grad():
        num_of_iter = 10
        texts = []
        with open(args.script_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line):
                    texts.append(line)

        print("   * input text\n    {} \n".format(texts[0]))

        ''' tts ckpt '''
        for i in range(num_of_iter):
            start = time.time()
            mel, length, alignments = infer(args.tts_ckpt, texts[0], ref_audio)

            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.to(device)

            audio = model.inference(mel)
            audio = audio.cpu().detach().numpy()
            save_path = os.path.join(args.out_dir, str(i) + '_audio.wav')
            os.makedirs(args.out_dir, exist_ok=True)
            write(save_path, hp.audio.sampling_rate, audio)
            audio_length = len(audio)/hp.audio.sampling_rate

            print("    {}. ".format(i+1))
            print("     - Path of generated audio file: {}".format(save_path))
            print("     - Length of generated audio file: {}s".format(audio_length))
            print("     - Time taken from text loading to generate spectrogram: : {}s".format(time.time() - start))
            print("     - Time taken to generate waveform: : {}s\n".format(time.time() - start + mel_time))
        print("finished generation")

if __name__ == '__main__':
    mel_ckpt = './output/melgan.pt'
    tts_ckpt = './output/ckpt'
    script_path = 'test/1.txt'
    out_dir = 'samples'
    ref = './inferref/ref.npy'

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for configs. will use hp_str from checkpoint if not given.")
    parser.add_argument('-m', '--mel_ckpt', type=str, required=False, default=mel_ckpt,
                        help="path of MelGAN checkpoint pt file for evaluation")
    parser.add_argument('-t', '--tts_ckpt', type=str, required=False, default=tts_ckpt,
                        help="path of TTS checkpoint pt file for evaluation")
    parser.add_argument('-s', '--script_path', type=str, required=False, default=script_path,
                        help="path of script file for evaluation")
    parser.add_argument('-o', '--out_dir', type=str, required=False, default=out_dir,
                        help="output directory")
    parser.add_argument('-i', '--iteration', type=str, required=False, default=10,
                        help="output directory")
    parser.add_argument('-r', '--ref_audio', type=str, required=False, default=ref,
                        help="reference audio")


    args = parser.parse_args()

    main(args, ref)
