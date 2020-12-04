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
    trainer = pl.Trainer(gpus=1, max_epochs=500)
    #trainer = pl.Trainer(gpus=1, max_epochs=5)
    #trainer = pl.Trainer(gpus=0, max_epochs=1)
    quartznet_model = nemo_asr.models.EncDecCTCModel(
        cfg=DictConfig(params['model']), trainer=trainer)

    trainer.fit(quartznet_model)
    return quartznet_model
    #trainer.test(test_manifest)


def restore_asr(restore_path):
    quartznet_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path)
    return quartznet_model


def validate_asr(asr_model, val_ds, num_to_validate):
    val_set = []
    with open(val_ds) as F:
        for line in F:
            val = json.loads(line)
            val_set.append(val)
    val_files = [t["audio_filepath"] for t in val_set[0:num_to_validate]]
    #print(val_files)
    transcription = asr_model.transcribe(val_files, batch_size=32, logprobs=False)
    print(transcription)


def main():
    parser = argparse.ArgumentParser(description="Example ASR Trainer")
    parser.add_argument("--model", required=False, default=QUARTZNET_PATH, type=str)
    parser.add_argument("--train-ds", required=False, default=None, type=str)
    parser.add_argument("--restore", required=False, default=None, type=str)
    parser.add_argument("--pretrained", required=False, default=None, type=str)
    parser.add_argument("--val-ds", required=False, default=None, type=str)
    parser.add_argument("--num-to-validate", required=False, default=10, type=int)
    parser.add_argument("--save", required=False, default=None, type=str)
    args = parser.parse_args()

    assert args.train_ds or args.restore or args.pretrained
    assert not (args.train_ds and args.restore)
    assert not (args.restore and args.save)

    params = read_model_cfg(args.model, args.train_ds, args.val_ds)

    asr_pretrained = None
    asr_model = None

    if args.pretrained:
        # Example pretrained: QuartzNet15x5Base-En
        asr_pretrained = nemo_asr.models.EncDecCTCModel.from_pretrained(
            model_name=args.pretrained)

    if args.restore:
        asr_model = restore_asr(args.restore)
    elif args.train_ds:
        asr_model = train_asr(params)
        if args.save:
            asr_model.save_to(args.save)

    if args.val_ds:
        if asr_pretrained:
            validate_asr(asr_pretrained, args.val_ds, args.num_to_validate)
        if asr_model:
            validate_asr(asr_model, args.val_ds, args.num_to_validate)


if __name__ == '__main__':
  main()
