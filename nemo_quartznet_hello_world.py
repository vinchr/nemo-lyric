#
# Taken from https://towardsdatascience.com/train-conversational-ai-in-3-lines-of-code-with-nemo-and-lightning-a6088988ae37
#
# Quartznet model architecture:
# https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/quartznet_15x5.yaml
#

import numpy as np
from string import ascii_lowercase
import librosa
import nemo.collections.asr as nemo_asr

import os.path
from ctc_segmentation import ctc_segmentation, CtcSegmentationParameters, prepare_text, determine_utterance_segments

def predict_labels_greedy(alphabet,ctc_probs):
    '''
    model - nemo model
    filename - audio to perform prediction
    '''

    probs = ctc_probs[0].numpy()

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

    transcript = 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'.lower()

    #build typical alphabet
    alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                   "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'",'%']

    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

    logprobs = quartznet.transcribe([filename],logprobs=True)

    greedy_transcript = predict_labels_greedy(alphabet,logprobs)

    # fork lifted the example from here:
    # https://github.com/lumaku/ctc-segmentation
    #
    config                              = CtcSegmentationParameters()
    config.subsampling_factor =1
    config.blank = 28
    ground_truth_mat, utt_begin_indices = prepare_text(config,transcript,alphabet)
    timings, char_probs, state_list     = ctc_segmentation(config,logprobs[0].numpy(),ground_truth_mat)
    # Obtain list of utterances with time intervals and confidence score
    segments                            = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcript)
    
    quartznet_transcript = quartznet.transcribe([filename])

    print('Ground Truth Transcript:',transcript)
    print('Quartznet Transcript:',quartznet_transcript[0])
    print('Quartznet Dense Sequnce (greedy search):\n',greedy_transcript)
    print('CTC Segmentation Dense Sequnce:\n',''.join(state_list))
