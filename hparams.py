import os
import argparse
import json

from text.symbols import kor_symbols

class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default='outdir',
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, default='logdir',
                        help='directory to save tensorboard logs')
    parser.add_argument('-ck', '--checkpoint_path', type=str, default=  './output/ckpt_tts',
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true', default=False,
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                        help='JSON file for configuration')

    args = parser.parse_args()
    model_dir = args.output_directory

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()

    config = json.loads(data)

    hparams = HParams(**config)

    hparams.n_symbols = len(kor_symbols)

    hparams.training_files = '/media/sh/Workspace/SKT_DB/script.txt'
    hparams.validation_files = '/media/sh/Workspace/SKT_DB/script_val.txt'
    hparams.dur_path= '/media/sh/Workspace/SKT_DB/2020/large/FSNR0/wav_16000_alig'

    hparams.output_directory = args.output_directory
    hparams.log_directory = args.log_directory
    hparams.checkpoint_path = args.checkpoint_path

    hparams.warm_start = args.warm_start
    hparams.n_gpus = args.n_gpus
    hparams.rank = args.rank
    hparams.group_name = args.group_name

    return hparams



