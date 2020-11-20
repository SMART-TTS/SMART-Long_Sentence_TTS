import numpy as np
import torch
from hparams import create_hparams
from train import load_model
from text import text_to_sequence


hparam = create_hparams()
cleaner_names = hparam.text_cleaners
def infer(checkpoint_path, text):
    hparams = create_hparams()
    # hparams.sampling_rate = 16000

    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()#.half()

    sequence = np.array(text_to_sequence(text,cleaner_names))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments, length, duration = model.inference(sequence)
    return mel_outputs_postnet, length, alignments