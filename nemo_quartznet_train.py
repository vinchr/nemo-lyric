#
# Taken from https://towardsdatascience.com/train-conversational-ai-in-3-lines-of-code-with-nemo-and-lightning-a6088988ae37
#
# Quartznet model architecture:
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/quartznet_15x5.yaml
#
import re
import argparse
import json
import os.path

from ruamel.yaml import YAML

from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr

from nemo.utils import logging

from ctc_segmentation import ctc_segmentation, CtcSegmentationParameters, prepare_text, determine_utterance_segments

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield



def read_model_cfg(config_path, args):
    train_manifest = args.train_ds
    validation_manifest = args.val_ds
    sample_rate = args.sample_rate
    preprocessor_sample_rate = args.preprocessor_sample_rate
    if not preprocessor_sample_rate:
        preprocessor_sample_rate = sample_rate
    batch_size = args.batch_size
    gradient_batch_size = args.gradient_batch_size

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = validation_manifest
    if sample_rate:
        params['model']['sample_rate'] = sample_rate
        params['model']['preprocessor']['params']['sample_rate'] = preprocessor_sample_rate
        params['model']['train_ds']['sample_rate'] = sample_rate
        params['model']['validation_ds']['sample_rate'] = sample_rate
    if batch_size:
        params['model']['train_ds']['batch_size'] = batch_size
        params['model']['validation_ds']['batch_size'] = batch_size
    if gradient_batch_size:
        accumulate_grad_batches = gradient_batch_size // params['model']['train_ds']['batch_size']
        accumulate_grad_batches = max(accumulate_grad_batches, 1)
        assert 'accumulate_grad_batches' in params['trainer']
        params['trainer']['accumulate_grad_batches'] = accumulate_grad_batches
    return params


def train_asr(params, resume_from_checkpoint, gpus, epochs, do_ddp):
    accelerator = 'ddp' if do_ddp else None
    # fast_dev_run = 5000
    precision = 16 if gpus > 0 else 32
    precision = 32
    trainer = pl.Trainer(resume_from_checkpoint=resume_from_checkpoint,
                         gpus=gpus,
                         max_epochs=epochs,
                         accelerator=accelerator,
                         precision=precision)
    quartznet_model = nemo_asr.models.EncDecCTCModel(
        cfg=DictConfig(params['model']), trainer=trainer)

    trainer.fit(quartznet_model)
    return quartznet_model


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
    test_cfg = asr_model.cfg['validation_ds']
    test_cfg['manifest_filepath'] = val_ds
    asr_model.setup_test_data(test_cfg)
    # trainer = pl.Trainer()
    # trainer.test(asr_model.test_dataloader())
    calc_wer(asr_model)
    #asr_model.preprocessor._sample_rate = 22050.
    asr_model.preprocessor._sample_rate = test_cfg['sample_rate']
    print("batch size: ", test_cfg['batch_size'],
          "preprocessor sample_rate: ", asr_model.preprocessor._sample_rate)
    transcription = asr_model.transcribe(
        val_files, batch_size=test_cfg['batch_size'], logprobs=False)
    print(transcription)

def validate_asr_with_alignment(asr_model,val_ds,num_to_validate):
    
    val_set = []
    with open(val_ds) as F:
        for line in F:
            val = json.loads(line)
            val_set.append(val)
    val_files = [t["audio_filepath"] for t in val_set[0:num_to_validate]]
    val_text = [t["text"] for t in val_set[0:num_to_validate]]
    test_cfg = asr_model.cfg['validation_ds']
    test_cfg['manifest_filepath'] = val_ds
    asr_model.setup_test_data(test_cfg)  #TODO: what is this doing?
    calc_wer(asr_model)
    asr_model.preprocessor._sample_rate = test_cfg['sample_rate']
    print("batch size: ", test_cfg['batch_size'],
          "preprocessor sample_rate: ", asr_model.preprocessor._sample_rate)
    
    logprobs_list = asr_model.transcribe(val_files, batch_size=test_cfg['batch_size'], logprobs=True)
    nlogprobs = len(logprobs_list)
    alphabet  = [t for t in asr_model.cfg['labels']] + ['%'] # converting to list and adding blank character.

    # adapted example from here:
    # https://github.com/lumaku/ctc-segmentation
    config = CtcSegmentationParameters()
    config.frame_duration_ms = 20  #frame duration is the window of the predictions (i.e. logprobs prediction window) 
    config.blank = len(alphabet)-1 #index for character that is intended for 'blank' - in our case, we specify the last character in alphabet.

    for ii in range(nlogprobs):
        transcript = val_text[ii]

        ground_truth_mat, utt_begin_indices = prepare_text(config,transcript,alphabet)

        timings, char_probs, state_list     = ctc_segmentation(config,logprobs_list[ii].cpu().numpy(),ground_truth_mat)
        
        # Obtain list of utterances with time intervals and confidence score
        segments                            = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcript)
        
        quartznet_transcript = asr_model.transcribe([val_files[ii]])

        print('Ground Truth Transcript:',transcript)
        print('Quartznet Transcript:',quartznet_transcript[0])
        print('CTC Segmentation Dense Sequnce:\n',''.join(state_list))

        #save onset per word.
        print('Saving timing prediction.')
        fname = open(val_files[ii][:-4]+'_align.csv','w') #jamendolyrics convention
        for i in transcript.split():
           # re.search performs regular expression operations.
           # .format inserts characters into {}.  
           # r'<string>' is considered a raw string.
           # char.start() gives you the start index of the starting character of the word (i) in transcript string
           # char.end() gives you the last index of the ending character** of the word (i) in transcript string
           # **the ending character is offset by one for the regex command, so a -1 is required to get the right 
           # index
           char = re.search(r'\b({})\b'.format(i),transcript)
           #       segments[index of character][start time of char=0]
           onset = segments[char.start()][0]
           #       segments[index of character][end time of char=1]
           term  = segments[char.end()-1][1]
           fname.write(str(onset)+','+str(term)+'\n')
        fname.close()

# Adapted from an example in NeMo repo
def calc_wer(asr_model):
    asr_model.eval()
    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])
    wer = nemo_asr.metrics.wer.WER(vocabulary=asr_model.decoder.vocabulary)
    hypotheses = []
    references = []
    for test_batch in asr_model.test_dataloader():
        if torch.cuda.is_available():
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
        hypotheses += wer.ctc_decoder_predictions_tensor(greedy_predictions)
        for batch_ind in range(greedy_predictions.shape[0]):
            reference = ''.join([labels_map[c] for c in test_batch[2][batch_ind].cpu().detach().numpy()])
            references.append(reference)
        del test_batch
    logging.info(hypotheses)
    logging.info(references)
    wer_value = nemo_asr.metrics.wer.word_error_rate(hypotheses=hypotheses, references=references)
    # if wer_value > args.wer_tolerance:
    #     raise ValueError(f"Got WER of {wer_value}. It was higher than {args.wer_tolerance}")
    # logging.info(f'Got WER of {wer_value}. Tolerance was {args.wer_tolerance}')
    logging.info(f'Got WER of {wer_value}.')


def main():
    parser = argparse.ArgumentParser(description="Example ASR Trainer")
    parser.add_argument("--model", required=False, default=None, type=str)
    parser.add_argument("--resume-from-checkpoint", required=False, default=None, type=str)
    parser.add_argument("--train-ds", required=False, default=None, type=str)
    parser.add_argument("--gpus", required=False, default=torch.cuda.device_count(), type=int)
    parser.add_argument("--ddp", required=False, dest='ddp', action='store_true')
    parser.add_argument("--no-ddp", required=False, dest='ddp', action='store_false')
    parser.set_defaults(ddp=True if torch.cuda.is_available() else False)
    parser.add_argument("--epochs", required=False, default=1, type=int)
    parser.add_argument("--sample-rate", required=False, default=None, type=int)
    parser.add_argument("--preprocessor-sample-rate", required=False, default=None, type=int)
    parser.add_argument("--batch-size", required=False, default=None, type=int)
    parser.add_argument("--gradient-batch-size", required=False, default=None, type=int)
    parser.add_argument("--restore", required=False, default=None, type=str)
    parser.add_argument("--pretrained", required=False, default=None, type=str)
    parser.add_argument("--val-ds", required=False, default=None, type=str)
    parser.add_argument("--num-to-validate", required=False, default=10, type=int)
    parser.add_argument("--save", required=False, default=None, type=str)
    args = parser.parse_args()

    assert args.train_ds or args.restore or args.pretrained
    assert not (args.train_ds and args.restore)
    assert not (args.restore and args.save)

    asr_pretrained = None
    asr_model = None

    if args.pretrained:
        # Example pretrained: QuartzNet15x5Base-En
        asr_pretrained = nemo_asr.models.EncDecCTCModel.from_pretrained(
            model_name=args.pretrained)

    if args.restore:
        asr_model = restore_asr(args.restore)
    elif args.train_ds:
        params = read_model_cfg(args.model, args)
        asr_model = train_asr(params, args.resume_from_checkpoint,
                              args.gpus, args.epochs, args.ddp)
        if args.save:
            asr_model.save_to(args.save)

    if args.val_ds and not args.train_ds:
        if asr_pretrained:
            validate_asr(asr_pretrained, args.val_ds, args.num_to_validate)
        if asr_model:
            validate_asr_with_alignment(asr_model, args.val_ds, args.num_to_validate)


if __name__ == '__main__':
  main()
