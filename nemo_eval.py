#
# Taken from https://towardsdatascience.com/train-conversational-ai-in-3-lines-of-code-with-nemo-and-lightning-a6088988ae37
#
# Quartznet model architecture:
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/quartznet_15x5.yaml
#
import re,time,argparse,json,sys,glob,librosa
import os.path
import numpy as np

import soundfile as sf
from ruamel.yaml import YAML

from omegaconf import OmegaConf
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
import ctc_segmentation as ctc 

from nemo.utils import logging

from configparser import ConfigParser

import sys
import Evaluate

tmp_dir = 'tmp' #make tmp for temporary files that are cropped to run through the model.
logprobs_ext = '_logprobs.npy' #temporary file for CTC decoding
sample_rate = 22050 #sampling rate (could be in the config file)
window_size_samples = 225500 #window for the predictions based on the model

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

def normalize_data(data):
    '''
    adjust the data from 1 to -1.
    '''
    xmin = np.min(data)
    xmax = np.max(data)
    return (2*(data - xmin) / (xmax-xmin+1e-10)-1)

def save_samples_wav(song_id, audio_path, id_num, x, window_samples, sr):
    '''
    Inputs:
    song_id       (string) DALI song ID, used as basename
    audio_path    (string) folder to save to
    id_num        (int) used to append to the name of the file.  can be something specific like line number
                         or just an arbitrary counter.
    x             (numpy array) song samples of entire song
    window_samples  (tuple) (start,end) window of samples to save to disk


    Return:
    filename with no path      (string)
    absolute path and filename (string)
    '''
    n = x.shape[0]
    filename = generate_wav_file_name(song_id,audio_path,id_num)
    
    start = window_samples[0]
    term = window_samples[1]
    
    #print('Writing:', filename,',',window_samples)
    sf.write(filename, x[start:term], sr)
    return filename

def calc_window_for_song(total_length,win_samples):
    '''
    calculate the start / end index for training windows
    slide window over song every (win_samples / 2) samples.

    Input:
    total_length (int) - total samples in a song
    win_samples (int) -  window size

    Return:
    start_ndx (m,) numpy array, the start of each window relative to the total samples of the song
    end_ndx (m,) numpy array, the end of each window relative to the total samples of song
    '''
    n   = np.arange(total_length)  # counter from 0 to max samples of x
    div = np.floor(total_length / win_samples).astype(int)
    rem = total_length % win_samples
    ndx = np.reshape(n[:-rem],(div,win_samples))
    start_ndx = np.reshape(ndx[:,0],(ndx[:,0].shape[0],1))
    end_ndx   = np.reshape(ndx[:,-1],(ndx[:,-1].shape[0],1))

    return np.concatenate((start_ndx, end_ndx),axis=1)

def generate_wav_file_name(song_id, audio_path, id_num):
    '''
    generating a filename for .wav files for training data

    Input:
    song_id       (string) DALI song ID, used as basename
    audio_path    (string) folder to save to
    id_num        (int) used to append to the name of the file.  can be something specific like line number
                         or just an arbitrary counter.

    Return:
    filename
    '''

    id_str = str(id_num).zfill(3)
    if id_num >= 1000:
        print('YIKES.  save_samples_wav - more than 1000 segments?')
        sys.exit()

    return audio_path + '/' + song_id + '_' + id_str + '.wav'

def crop_song(audio_filename, audio_path, win_samples, sample_rate):
    '''
    crop_song - takes a DALI song and crops it m times into win_length samples.

    Inputs: 
       song_id    - DALI song id
       entry      - DALI data entry
       audio_path - absolute path where audio files are stored (read/write)
       win_samples - number of samples for each crop
    Return:
       song_ndx   - (m,start_sample,stop_sample) indices for the m crops
       filename_list - absolute path for filenames saved with save_samples_wav

    1. load song with librosa
    2. calculate indices (i.e. sample index starting at 0) for windows of chunks. 
       a. 'win_rate' is win_length/2
       b. do not keep parts of the song less than win_length
    3. crop according to indices and save to audio_path in the form
        audio/<song_id>_##.wav 
        where ## is the number of chunks in the song.
   '''
    xin, sr = librosa.load(audio_filename, sr=sample_rate)
    x = normalize_data(xin)

    song_ndx = calc_window_for_song(x.shape[0],win_samples)

    l = song_ndx.shape[0]
    basename = audio_filename.split('/')[-1][:-4]

    filename_list = []
    for i in range(l):
        filename_list.append( save_samples_wav(basename, audio_path, i, x, (song_ndx[i,0],song_ndx[i,1]), sr) )

    return song_ndx, filename_list

def restore_asr(restore_path):
    quartznet_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path)
    return quartznet_model


def prediction_save_logprobs(asr_model,filename,lp_ext='_logprobs.npy',prediction_dir='tmp'):
    

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
    pred_fname = prediction_dir+'/'+audiofile[:-4]+lp_ext
    np.save(pred_fname,logprobs_list)

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

def prediction_one_song(model,audio_filename,transcript,lp_dir='tmp',lp_ext='_logprobs.py',word_dir='../lyrics',word_ext='.words.txt',prediction_dir='metadata',prediction_ext='_align.csv'):
    '''
    model  - nemo model object
    lp_dir - path with logprobabilities
    audio_filename - file name of audio song that is being proceesed

    '''
    basename = audio_filename[:-4] #crop extension (mp3 or wav)
    alphabet  = [t for t in model.cfg['labels']] + ['%'] # converting to list and adding blank character.

    # adapted example from here:
    # https://github.com/lumaku/ctc-segmentation
    config = ctc.CtcSegmentationParameters()
    config.frame_duration_ms = 20  #frame duration is the window of the predictions (i.e. logprobs prediction window) 
    config.blank = len(alphabet)-1 #index for character that is intended for 'blank' - in our case, we specify the last character in alphabet.
    logprobs_filenames      = glob.glob(os.path.join(lp_dir,basename+'*'+lp_ext))
    logprobs_filenames.sort()

    logprobs_list = []
    for f in logprobs_filenames:
        logprobs_list.append(np.load(f))
    
    logprobs = logprobs_list[0]
    for i in range(1,len(logprobs_list)):
        logprobs = np.concatenate((logprobs,logprobs_list[i]))

    print('Prepare Text.',flush=True)
    ground_truth_mat, utt_begin_indices = ctc.prepare_text(config,transcript,alphabet)

    print('Segmentation.',flush=True)
    timings, char_probs, state_list     = ctc.ctc_segmentation(config,logprobs,ground_truth_mat)
    
    print('Get time intervals.',flush=True)
    # Obtain list of utterances with time intervals and confidence score
    segments                            = ctc.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcript)
    tend = time.time()
    pred_fname = prediction_dir+'/'+basename+'_align.csv' #jamendolyrics convention
    fname = open(pred_fname,'w') 
    offset = 0  #offset is used to compensate for the re.search command which only finds the first 
                # match in the string.  so the transcript is iteratively cropped to ensure that the 
                # previous words in the transcript are not found again.
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
       char = re.search(r'{}'.format(i),transcript[offset:])
       #       segments[index of character][start time of char=0]
       onset = segments[char.start()+offset][0]
       #       segments[index of character][end time of char=1]
       term  = segments[char.end()-1+offset][1]
       offset += char.end()
       fname.write(str(onset)+','+str(term)+'\n')
    fname.close()


if __name__ == '__main__':
    '''
    For a single song...
    1. read in nemo audio manifest of data to evaluate 
        Input: manifest filename
        Output: song file name and transcripts per song
    2. chop it up song to fit the model. 
        Input: song filename
        Output: .wav's
    3. predict logprobs for each segment in #2 
        Input: .wav's
        Output: .npy's
    4. concatenate the logprobs and run ctc_segmentation (_align.csv)
        Input: .npy's
        Output: _align.csv (one)
    6. run Jamendo Evaluate on results
        Input: _align.csv and .wordonset.txt
        Output: <printed alignment error>
    '''
    parser = argparse.ArgumentParser(description="Run Audio file(s) / Transcripts (from nemo audio manifest)  through a known model for lyric alignment predictions, and save to file.")
    parser.add_argument('-c','--config', required=False, default='nemo.cfg', type=str,help='config file with model, audio, prediction setup information.')
    parser.add_argument('--eval-only', required=False, default=False, action='store_true',help='skips preprocessing and goes straight to alignment.')
    args = parser.parse_args()

    print('Using: ',args.config)

    config = ConfigParser(inline_comment_prefixes=["#"])
    config.read(args.config)
    
    if not args.eval_only:

        audio_manifest_path = config.get('main','AUDIO_MANIFEST')
        print('Using: ',audio_manifest_path)

        exit_flag = False
        if not os.path.exists(audio_manifest_path):
            print(audio_manifest_path,'not found.  Exiting.')
            exit_flag = True
        
        model_filename      = config.get('main','MODEL')
        print('Using: ',model_filename)

        if not os.path.exists(model_filename):
            print(model_filename,'not found.  Exiting.')
            exit_flag = True
        
        prediction_path     = config.get('main','PREDICTION_PATH')
        print('Using: ',prediction_path)
        if not os.path.exists(prediction_path):
            print('prediction directory not found, trying to create it.')
            os.makedirs(prediction_path)
            if not os.path.isabs(prediction_path):
                #make string absolute path
                prediction_dir = os.path.abspath(prediction_path)

        if not os.path.exists(tmp_dir):
            print('WARNING: Creating directory:',tmp_dir)
            os.makedirs(tmp_dir)

        if exit_flag: sys.exit()

        files,transcripts = read_manifest(audio_manifest_path)
        #load model 
        asr_model = restore_asr(model_filename)

        for i in range(len(files)):
            song_fname = files[i]
            transcript = transcripts[i]
            print('Cropping songs...')
            #225500 - 10.23 seconds is the time chosen for the size of model...
            #22050  - sample rate used for training the model
            _ , song_cnames_list = crop_song(song_fname,tmp_dir,window_size_samples,sample_rate)

            print('Testing',len(song_cnames_list),'files.')
            ptime = []
            for j in range(len(song_cnames_list)):
                print('Testing',strip_path(song_cnames_list[j]))
                prediction_save_logprobs(asr_model, song_cnames_list[j], logprobs_ext,tmp_dir)

            prediction_one_song(asr_model,strip_path(song_fname),transcript,lp_dir=tmp_dir,lp_ext=logprobs_ext,prediction_dir=prediction_path,prediction_ext='_align.csv')

    #delay = np.arange(0,25.0,1.0)
    delay = [0]
    ae_list = []
    results_list = []
    preds_list = []
    for i in delay:
        config['main']['DELAY'] = str(i)
        results, preds = Evaluate.compute_metrics(config)
        #Evaluate.print_results(results)
        ae_list.append(results['mean_AE'][0])
        results_list.append(results)
        preds_list.append(preds)

    ndx = np.argmin(ae_list)
    print('Lowest Error was:',delay[ndx])
    print( Evaluate.print_results( results_list[ndx]) )
    print(ae_list)
