import os
import glob
import torch
import tqdm
import numpy as np
npy_files = glob.glob(os.path.join('/media/qw/data/taco_mel', '*.npy'), recursive=True)
with torch.no_grad():
    for melpath in tqdm.tqdm(npy_files):
        # mel = torch.load(melpath)
        mel = np.load(melpath)
        mel = mel.T
        mel = torch.from_numpy(mel).unsqueeze(0)
        newpath = melpath.replace('.npy', '.mel')
        torch.save(mel, newpath)