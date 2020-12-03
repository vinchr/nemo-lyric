#
# Taken from https://towardsdatascience.com/train-conversational-ai-in-3-lines-of-code-with-nemo-and-lightning-a6088988ae37
#
# Quartznet model architecture:
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/quartznet_15x5.yaml
#

import argparse
import json

from ruamel.yaml import YAML

import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig

import os.path

NEMO_REPO_PATH = "../NeMo"

QUARTZNET_PATH = os.path.join(
                              NEMO_REPO_PATH,
                              'examples/asr/conf/quartznet_15x5.yaml')

def read_model_cfg(config_path, train_manifest, test_manifest):
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = test_manifest
    return params


def train_asr(params):
    #trainer = pl.Trainer(gpus=1, max_epochs=50)
    trainer = pl.Trainer(gpus=0, max_epochs=1)
    quartznet_model = nemo_asr.models.EncDecCTCModel(
        cfg=DictConfig(params['model']), trainer=trainer)

    trainer.fit(quartznet_model)
    return quartznet_model
    #trainer.test(test_manifest)


def validate_asr(asr_model, val_ds):
    val_set = None
    with open(val_ds) as F:
        val_set = json.load(val_ds)
    val_files = [t["audio_filepath"] for t in val_set[0:4]]
    transcription = asr_model.transcribe(val_files)
    print(transcription)


def main():
    parser = argparse.ArgumentParser(description="Example ASR Trainer")
    parser.add_argument("--model", required=False, default=QUARTZNET_PATH, type=str)
    parser.add_argument("--train-ds", required=True, default=None, type=str)
    parser.add_argument("--val-ds", required=False, default=None, type=str)
    parser.add_argument("--save", required=False, default=None, type=str)
    args = parser.parse_args()

    params = read_model_cfg(args.model, args.train_ds, args.val_ds)
    asr_model = train_asr(params)
    if args.save:
        asr_model.save_to(args.save)
    if args.val_ds:
        validate_asr(asr_model)


if __name__ == '__main__':
  main()
