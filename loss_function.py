from torch import nn
import torch
import random

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target, alignment_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, attention, duration_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        #Duration loss
        if duration_out is not None:
            alignment_target.requires_grad = False
            duration_loss = nn.L1Loss()(duration_out, alignment_target.float().cuda())
        else:
            duration_loss = mel_loss
            # print('no duration loss')
        return mel_loss + gate_loss + duration_loss, duration_loss