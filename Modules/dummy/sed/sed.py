import soundfile as sound
import librosa
import numpy as np
import keras
import sys


def feature(file):
    FREQ_BINS = 40
    FFT_POINTS = 321
    HOP_LENGTH = int(FFT_POINTS/2)

    y, sr = sound.read(file)
    feat = librosa.feature.melspectrogram(y, 
                                          sr=sr, 
                                          n_fft=FFT_POINTS,
                                          hop_length=HOP_LENGTH,
                                          n_mels=FREQ_BINS,
                                          fmin=0.0, 
                                          fmax=sr/2, 
                                          htk=True, 
                                          norm=None)
    feat_log = np.log(feat+1e-8)
    feat_log = np.expand_dims(feat_log, axis=0)
    feat_log = np.expand_dims(feat_log, axis=3)
    return feat_log

def process(idx, thres, feat_log, model):
    classes = [['baby'],['bicycle'],['boiling'],['car'],['carpassing'],
           ['clock'],['dog'],['door'],['fire'],['glass'],['jackhammer'],
           ['kettle'],['scream'],['siren'],['speech'],['unknown'],['whistle']] 
    unknown_flag = False

    softmax = model.predict(feat_log)
    result = np.argmax(softmax,axis=1)

    if float(softmax[0][int(result)]) > thres:
        unknown_flag = False
        out_softmax = softmax
        out_result = result
        out_classes = classes
    else:
        unknown_flag = True

    if unknown_flag == True:
        label_dict = {'score': 0.0, 'description': 'unknown'}
        times_stamp = str(round(idx*0.1,1))+':'+str(round(idx*0.1+0.1,1))
        out_dict = {'label': label_dict, 'timestamp': times_stamp}
        out.append(out_dict)
    else:
        label_dict = {'score': out_softmax[0][int(out_result)], 'description': out_classes[int(out_result)][0]}
        times_stamp = str(round(idx*0.1,1))+':'+str(round(idx*0.1+0.1,1))
        out_dict = {'label': label_dict, 'timestamp': times_stamp}

    return out_dict

