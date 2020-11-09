import os
import time
import datetime
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
from melgan.model.generator import Generator
from melgan.utils.hparams import HParam, load_hparam_str
from for_melgan import infer

MAX_WAV_VALUE = 32768.0

from plotting_utils import plot_alignment_to_numpy
import matplotlib.pylab as plt

def main(args):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    text_path = "/media/sh/Workspace/긴문장합성/kor_Document-level_Neural_TTS_length/test/1.txt"
    doc_ckpt_kor = '/media/sh/Workspace/긴문장합성/kor_Document-level_Neural_TTS_length/outdir/checkpoint_29000'
    save_folder = '/media/sh/Workspace/긴문장합성/samples'
    today = datetime.datetime.today()
    time = str(today.month) + str(today.day) + str(today.hour) + str(today.minute) + str(today.second)
    # time = str(time)
    save_name = 'kor_audio_length_regul_' + doc_ckpt_kor.split('_')[-1] +'_' + time+ '.wav'
    save_path = os.path.join(save_folder, save_name)

    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=False)

    with torch.no_grad():
        # for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.mel'))):
        #     mel = torch.load(melpath)
        texts = []
        with open(text_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line):
                    texts.append(line)

        mel, length, alignments = infer(doc_ckpt_kor, texts[0])

        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()

        audio = model.inference(mel)
        audio = audio.cpu().detach().numpy()
        # out_path = melpath.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
        write(save_path, hp.audio.sampling_rate, audio)
        print('합성 끝')
    # print(length.size(1), mel.size(2))
    #
    #
    # plt.figure()
    # plt.imshow(mel[0].cpu().detach().numpy(), extent=[0,400,0,80])
    # plt.savefig('mel.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=False, default='/media/sh/Workspace/taco2melgan_kor/melgan/melgan.pt',
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, required=False, default='./',
                        help="directory of mel-spectrograms to invert into raw audio. ")
    args = parser.parse_args()

    import time
    start = time.time()  # 시작 시간 저장
    main(args)
    print("time :", time.time() - start)
