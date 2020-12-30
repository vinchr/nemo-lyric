#
# Taken from https://towardsdatascience.com/train-conversational-ai-in-3-lines-of-code-with-nemo-and-lightning-a6088988ae37
#
# Quartznet model architecture:
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/quartznet_15x5.yaml
#
import re,time
import argparse
import json
import os.path
import numpy as np

from ruamel.yaml import YAML

from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr

from nemo.utils import logging

from ctc_segmentation import ctc_segmentation, CtcSegmentationParameters, prepare_text, determine_utterance_segments



from configparser import ConfigParser

import sys
sys.path.append('../dali-dataset-tools')
import dali_helpers

import Evaluate

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def restore_asr(restore_path):
    quartznet_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path)
    return quartznet_model


def prediction_save_logprobs(asr_model,filename,transcript,prediction_dir='./predictions/'):
    

    asr_model.preprocessor._sample_rate = 22050
    logprobs_list = asr_model.transcribe([filename], logprobs=True)[0].cpu().numpy()

    print('Saving logprobs for song.')
    if not os.path.isdir(prediction_dir):
        print('prediction directory not found, trying to create it.')
        os.makedirs(prediction_dir)
        if not os.path.isabs(prediction_dir):
            #make string absolute path
            prediction_dir = os.path.abspath(prediction_dir)
    audiofile = strip_path(filename)
    pred_fname = prediction_dir+'/'+audiofile[:-4]+'_logprobs.npy'
    #fname = open(pred_fname,'w') #jamendolyrics convention
    np.save(pred_fname,logprobs_list)

def prediction_with_alignment(asr_model,filename,transcript,prediction_dir='./predictions/'):
    

    asr_model.preprocessor._sample_rate = 22050
    
    logprobs_list = asr_model.transcribe([filename], logprobs=True)
    alphabet  = [t for t in asr_model.cfg['labels']] + ['%'] # converting to list and adding blank character.

    # adapted example from here:
    # https://github.com/lumaku/ctc-segmentation
    config = CtcSegmentationParameters()
    config.frame_duration_ms = 20  #frame duration is the window of the predictions (i.e. logprobs prediction window) 
    config.blank = len(alphabet)-1 #index for character that is intended for 'blank' - in our case, we specify the last character in alphabet.


    ground_truth_mat, utt_begin_indices = prepare_text(config,transcript,alphabet)

    timings, char_probs, state_list     = ctc_segmentation(config,logprobs_list[0].cpu().numpy(),ground_truth_mat)
    
    # Obtain list of utterances with time intervals and confidence score
    segments                            = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcript)

    print('Ground Truth Transcript:',transcript)
    print('CTC Segmentation Dense Sequnce:\n',''.join(state_list))

    #save onset per word.
    print('Saving timing prediction.')
    if not os.path.isdir(prediction_dir):
        print('prediction directory not found, trying to create it.')
        os.makedirs(prediction_dir)
        if not os.path.isabs(prediction_dir):
            #make string absolute path
            prediction_dir = os.path.abspath(prediction_dir)
    audiofile = strip_path(filename)
    pred_fname = prediction_dir+'/'+audiofile[:-4]+'_align.csv'
    fname = open(pred_fname,'w') #jamendolyrics convention
    for i in transcript.split():
       #
       # taking each word, and writing out the word timings from segments variable
       #
       # re.search performs regular expression operations.
       # .format inserts characters into {}.  
       # r'<string>' is considered a raw string.
       # char.start() gives you the start index of the starting character of the word (i) in transcript string
       # char.end() gives you the last index of the ending character** of the word (i) in transcript string
       # **the ending character is offset by one for the regex command, so a -1 is required to get the right 
       # index
       char = re.search(r'{}'.format(i),transcript)
       #       segments[index of character][start time of char=0]
       onset = segments[char.start()][0]
       #       segments[index of character][end time of char=1]
       term  = segments[char.end()-1][1]
       fname.write(str(onset)+','+str(term)+'\n')
    fname.close()


def read_manifest(filename):

    line_list = []
    with open(filename) as F:
        for line in F:
            val = json.loads(line)
            line_list.append(val)
    files = [t["audio_filepath"] for t in line_list]
    transcripts = [t["text"] for t in line_list]
    return files,transcripts

def strip_path(filename):
    return filename.split('/')[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Audio files / Transcripts (from nemo audio manifest)  through a known model for lyric alignment predictions, and save to file.")
    parser.add_argument('-c','--config', required=False, default='test/sample.cfg', type=str,help='config file with model, audio, prediction setup information.')
    args = parser.parse_args()

    config = ConfigParser(inline_comment_prefixes=["#"])
    config.read(args.config)

    audio_manifest_path = config.get('main','AUDIO_MANIFEST')
    
    exit_flag = False
    if not os.path.exists(audio_manifest_path):
        print(audio_manifest_path,'not found.  Exiting.')
        exit_flag = True
    
    model_filename      = config.get('main','MODEL')
    if not os.path.exists(model_filename):
        print(model_filename,'not found.  Exiting.')
        exit_flag = True
    
    prediction_path     = config.get('main','PREDICTION_PATH')
    if not os.path.exists(prediction_path):
        print(prediction_path,'not found.  It should be fine.')
        #TODO: make directory if doesn't exist...

    if exit_flag: sys.exit()

    files,transcripts = read_manifest(audio_manifest_path)
    
    #load model 
    asr_model = restore_asr(model_filename)

    print('Testing',len(files),'files.')
    ptime = []
    for i in range(len(files)):
        start = time.time()
        print('Testing',strip_path(files[i]))
        prediction_save_logprobs(asr_model, files[i], transcripts[i], prediction_path)
        term = time.time()
        ptime.append(term-start)

    print(np.mean(ptime),'to run prediction on 10s file.')

    results = Evaluate.compute_metrics(config)
    Evaluate.print_results(results)
