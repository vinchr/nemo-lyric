#
# Taken from https://towardsdatascience.com/train-conversational-ai-in-3-lines-of-code-with-nemo-and-lightning-a6088988ae37
#
# Quartznet model architecture:
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/quartznet_15x5.yaml
#

import numpy as np
from string import ascii_lowercase
import librosa,re
import nemo.collections.asr as nemo_asr

import os.path
from ctc_segmentation import ctc_segmentation, CtcSegmentationParameters, prepare_text, determine_utterance_segments

'''
ctc_segmentation libraries are from 
https://github.com/lumaku/ctc-segmentation

super easy configuration...
pip install ctc_segmentation
'''

def predict_labels_greedy(alphabet,probs):
    '''
    model - nemo model
    filename - audio to perform prediction
    '''

    alphabet_sequence = ''
    # look at each time and select the highest probability item for that 
    # time window.
    for t in probs:
        alphabet_sequence += alphabet[np.argmax(t)]
    
    #optionally this could be returned to debug what's happening
    numeric_sequence = np.argmax(probs,axis=1)

    return alphabet_sequence


if __name__ == '__main__':
    print("Using pretrained Quartznet Model.")
    filename = './sample.wav'
    
    annotation_timing = [
           0.543,#mister
           0.833,#quilter
           1.361,#is
           1.443,#the
           1.659,#apostle
           2.143,#of
           2.262,#the
           2.418,#middle
           2.659,#classes
           3.333,#and
           3.489,#we
           3.608,#are
           3.638,#glad
           4.092,#to
           4.174,#welcome
           4.635,#his
           4.895, #gospel
    ]

    print('Saving timing annotations.')
    fname = open('sample_annotation.txt','w')
    for j in annotation_timing:
        fname.write(str(j)+'\n')
    fname.close()

    transcript = 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'.lower()

    #build typical alphabet
    alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                   "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'",'%']

    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    
    logprobs = quartznet.transcribe([filename],logprobs=True)
    
    greedy_transcript = predict_labels_greedy(alphabet,logprobs[0].cpu().numpy())

    # adapted example from here:
    # https://github.com/lumaku/ctc-segmentation
    config                              = CtcSegmentationParameters()

    #frame duration is the window of the predictions (i.e. logprobs prediction window)
    config.frame_duration_ms = 20
    #character that is intended for 'blank' - in our case, we specify the last character in alphabet.
    config.blank = len(alphabet)-1
    ground_truth_mat, utt_begin_indices = prepare_text(config,transcript,alphabet)
    timings, char_probs, state_list     = ctc_segmentation(config,logprobs[0].cpu().numpy(),ground_truth_mat)
    # Obtain list of utterances with time intervals and confidence score
    segments                            = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcript)
    
    quartznet_transcript = quartznet.transcribe([filename])

    print('Ground Truth Transcript:',transcript)
    print('Quartznet Transcript:',quartznet_transcript[0])
    print('Quartznet Dense Sequnce (greedy search):\n',greedy_transcript)
    print('CTC Segmentation Dense Sequnce:\n',''.join(state_list))

    #save onset per word.
    print('Saving timing prediction.')
    fname = open('sample_prediction.txt','w')
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
 


