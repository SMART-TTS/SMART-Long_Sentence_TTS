import os
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
            with open("/media/qw/data/Experiment/Encoder_selfAtt/test/1.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if len(line):
                        texts.append(line)
            for i in range(10):
                mel, length, alignments = infer('/media/qw/data/Experiment/Encoder_selfAtt/tacotron2_statedict.pt', texts[0])
                # mel, length, alignments = infer('/media/qw/data/Experiment/Encoder_selfAtt/result/3sentence', texts[0])
                # print('/'*i, '.'*(50-i))
                # plt.figure()
                # plt.imshow(alignments[0].T.cpu())
                # plt.savefig('./align/alignment{}.png'.format(i), dpi=300)
    # mel, length, alignments = infer('/media/qw/data/Experiment/Encoder_selfAtt/tacotron2_statedict.pt', 'Emil Sinclair is the protagonist of the novel. hello my name is sung woong hwang.')

                if len(mel.shape) == 2:
                    mel = mel.unsqueeze(0)
                mel = mel.cuda()

                audio = model.inference(mel)
                audio = audio.cpu().detach().numpy()
                # out_path = melpath.replace('.mel', '_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
                write('/media/qw/data/Experiment/Encoder_selfAtt/audio.wav', hp.audio.sampling_rate, audio)
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
    parser.add_argument('-p', '--checkpoint_path', type=str, required=False, default='./melgan.pt',
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, required=False, default='./',
                        help="directory of mel-spectrograms to invert into raw audio. ")
    args = parser.parse_args()

    main(args)
