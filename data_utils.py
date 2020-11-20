import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from torch import nn

import os

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        duration_path = os.path.join("/media/sh/DB/fsdata/alignments2", audiopath_and_text[0].split('/')[-2],
                        audiopath_and_text[0].split('/')[-1].split('.')[0]+".npy")
        D = np.load(duration_path)
        D = torch.from_numpy(D)
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text + " ")
        mel = self.get_mel(audiopath)
        return (text, mel, D)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        # For D (duration으로 변형된 alignment)
        max_alignment_len = max_input_len
        alignment_padded = torch.LongTensor(len(batch), max_alignment_len)
        alignment_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            alignment = batch[ids_sorted_decreasing[i]][2]
            if alignment_padded[i].size(0) < alignment.size(0):
                alignment_padded = None
                print('duration error')
                break
            else:
                alignment_padded[i, :alignment.size(0)] = alignment

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, alignment_padded


class TextMelCollate2():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        s_token = torch.tensor([0]).int()
        m_token = torch.tensor(torch.mul(torch.ones(80, 40), -10))
        d_token = torch.LongTensor([40])
        new_batch = []  # (text, mel, duration, attention)
        for i in range(len(batch) // 2):
            new_batch.append(tuple((torch.cat((batch[i][0], s_token, batch[i + 1][0]), dim=-1),
                                    torch.cat((batch[i][1], m_token, batch[i + 1][1]), dim=-1),
                                    torch.cat((batch[i][2], d_token, batch[i + 1][2]), dim=-1))))

        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # For new batch
        input_lengths2, ids_sorted_decreasing2 = torch.sort(
            torch.LongTensor([len(x[0]) for x in new_batch]),
            dim=0, descending=True)
        max_input_len = input_lengths2[0]

        max_alignment_len = max_input_len
        alignment_padded = torch.LongTensor(len(new_batch), max_alignment_len)
        alignment_padded.zero_()
        for i in range(len(ids_sorted_decreasing2)):
            alignment = new_batch[ids_sorted_decreasing2[i]][2]
            if alignment_padded[i].size(0) < alignment.size(0):
                alignment_padded = None
                print('alignment error')
                break
            else:
                alignment_padded[i, :alignment.size(0)] = alignment

        text_padded2 = torch.LongTensor(len(new_batch), max_input_len)
        text_padded2.zero_()
        for i in range(len(ids_sorted_decreasing2)):
            text = new_batch[ids_sorted_decreasing2[i]][0]
            text_padded2[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = new_batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in new_batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded

        mel_padded2 = torch.FloatTensor(len(new_batch), num_mels, max_target_len)
        mel_padded2.zero_()
        gate_padded2 = torch.FloatTensor(len(new_batch), max_target_len)
        gate_padded2.zero_()

        output_lengths2 = torch.LongTensor(len(new_batch))
        for i in range(len(ids_sorted_decreasing2)):
            mel = new_batch[ids_sorted_decreasing2[i]][1]
            mel_padded2[i, :, :mel.size(1)] = mel
            gate_padded2[i, mel.size(1) - 1:] = 1
            output_lengths2[i] = mel.size(1)
        teacher_attention = None
        return text_padded2, input_lengths2, mel_padded2, gate_padded2, output_lengths2, alignment_padded


class TextMelCollate3():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        s_token = torch.tensor([0]).int()
        m_token = torch.tensor(torch.mul(torch.ones(80, 40), -10))
        d_token = torch.LongTensor([40])
        new_batch = []
        for i in range(len(batch) // 3):
            new_batch.append(
                tuple((torch.cat((batch[i][0], s_token, batch[i + 1][0], s_token, batch[i + 2][0]), dim=-1),
                       torch.cat((batch[i][1], m_token, batch[i + 1][1], m_token, batch[i + 2][1]), dim=-1),
                       torch.cat((batch[i][2], d_token, batch[i + 1][2], d_token, batch[i + 2][2]), dim=-1))))

        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # For new batch
        input_lengths2, ids_sorted_decreasing2 = torch.sort(
            torch.LongTensor([len(x[0]) for x in new_batch]),
            dim=0, descending=True)
        max_input_len = input_lengths2[0]

        max_alignment_len = max_input_len
        alignment_padded = torch.LongTensor(len(new_batch), max_alignment_len)
        alignment_padded.zero_()
        for i in range(len(ids_sorted_decreasing2)):
            alignment = new_batch[ids_sorted_decreasing2[i]][2]
            if alignment_padded[i].size(0) < alignment.size(0):
                alignment_padded = None
                print('alignment error')
                break
            else:
                alignment_padded[i, :alignment.size(0)] = alignment

        text_padded2 = torch.LongTensor(len(new_batch), max_input_len)
        text_padded2.zero_()
        for i in range(len(ids_sorted_decreasing2)):
            text = new_batch[ids_sorted_decreasing2[i]][0]
            text_padded2[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = new_batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in new_batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded

        mel_padded2 = torch.FloatTensor(len(new_batch), num_mels, max_target_len)
        mel_padded2.zero_()
        gate_padded2 = torch.FloatTensor(len(new_batch), max_target_len)
        gate_padded2.zero_()

        output_lengths2 = torch.LongTensor(len(new_batch))
        for i in range(len(ids_sorted_decreasing2)):
            mel = new_batch[ids_sorted_decreasing2[i]][1]
            mel_padded2[i, :, :mel.size(1)] = mel
            gate_padded2[i, mel.size(1) - 1:] = 1
            output_lengths2[i] = mel.size(1)
        teacher_attention = None
        return text_padded2, input_lengths2, mel_padded2, gate_padded2, output_lengths2, alignment_padded
