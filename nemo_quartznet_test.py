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


def prediction_with_alignment(asr_model,filename,transcript,prediction_dir='./predictions/'):
    

    asr_model.preprocessor._sample_rate = 22050
    print("batch size: ", 
          "preprocessor sample_rate: ", asr_model.preprocessor._sample_rate)
    
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
    
    quartznet_transcript = asr_model.transcribe([filename])

    print('Ground Truth Transcript:',transcript)
    print('Quartznet Transcript:',quartznet_transcript[0])
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
       char = re.search(r'\b({})\b'.format(i),transcript)
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

def validate_files(filename_list,audio_path):
    if not os.path.exists(filename_list[0]):
        print(filename_list[0],' dne, attempting to create in expected path')

def strip_path(filename):
    return filename.split('/')[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example ASR Trainer")
    #parser.add_argument("--model-yaml", required=False, default=None, type=str,help='the nemo model reference yaml.')
    #parser.add_argument("--nemo-manifest", required=False, default=None, type=str,help='the nemo model reference yaml.')
    #parser.add_argument('-n',"--num-to-evaluate", required=False, default=4, type=int)
    #args = parser.parse_args()


    #config = ConfigParser(inline_comment_prefixes=["#"])
    #config.read(args.config_path)

    #paths = get_params(config)

    #location of audio manifest
    audio_manifest_path = '10_songs_dali_train_04/dali_test_74ab294511ff4c2fa1b1e27be26b2834.json'
    model_path          = '10_songs_dali_train_04/version_58/quartznet_15x5_22k_adam_cosine.01.nemo'
    prediction_path     = '10_songs_dali_train_04/version_58/predictions'

    files,transcripts = read_manifest(audio_manifest_path)
    
    #load model 
    asr_model = restore_asr(model_path)

    print('Testing',len(files),'files.')
    ptime = []
    for i in range(len(files)):
        start = time.time()
        print('Testing',strip_path(files[i]))
        prediction_with_alignment(asr_model, files[i], transcripts[i], prediction_path)
        term = time.time()
        ptime.append(term-start)

    print(np.mean(ptime),'to run prediction on 10s file.')
