#!/usr/bin/env python
#title           : featureExtract.py
#description     : main feature exctraction module  for AARIP feature extraction and classification
#author          : Patrik Lechner <ptrk.lechner@gmail.com>
#date            : Mar 2019
#version         : 0.1
#usage           :
#notes           :
#python_version  : 3.6.3
#=======================================================================

import matplotlib.pyplot as plt
import librosa
import pandas as pd
import numpy as np
from scipy.io import wavfile
import customFeatures



def columns(featureDict):
    '''Helper function for creating pandas.Multiindex dataframe
    
    Arguments:
        featureDict {dictionary} -- a dictionary conataining names and dimensions of features
    
    Returns:
        pandas.Multiindex -- correctly dimensioned and named pandas multiindex dataframe
    '''

    cols = []
    for i, it in enumerate(featureDict.items()):
        size = it[1]
        name = it[0]
        for j in range(size):
            colName = name
            cols.append((name, j))
    return pd.MultiIndex.from_tuples(cols)


def extractFeatures(x,sr, label):
    '''Extracts Audio Features and returns them as a pandas.Dataframe
    
    Arguments:
        x {numpy.array} -- time domain mono signal
        sr {int} -- sampling rate
        label {int} -- a label to be used for classification
    
    Returns:
        pandas.Dataframe -- extracted features combined with label.
    '''
    nyq = sr/2.
    nMfcc = 20
    dropMfcc = 5
    finNMfcc = nMfcc-dropMfcc
    stft = np.abs(librosa.stft(x, n_fft=2**14, hop_length=2**12))

    melSpectrogram = librosa.feature.melspectrogram(S=stft)

    mfccs = librosa.feature.mfcc(S=melSpectrogram, n_mfcc=nMfcc)[dropMfcc:]/20

    rmse = librosa.feature.rmse(S=stft)

    spectralCentroid = librosa.feature.spectral_centroid(S=stft)/nyq
    spectralBandwidth = librosa.feature.spectral_bandwidth(S=stft)/nyq
    # spectralContrast = librosa.feature.spectral_contrast(S=stft)
    # spectralFlatness = librosa.feature.spectral_flatness(S=stft)
    spectralRolloff = librosa.feature.spectral_rolloff(S=stft)/nyq
    #  fcpe = customFeatures.filteredCepstrumPeakEnergy(x,plot=False)

    featureDict = dict(mfcc=finNMfcc, s_centroid=1, rmse=1, s_bw = 1, s_roll=1)#, fcpe = 1)


    feats = np.append(mfccs, spectralCentroid, axis=0)
    feats = np.append(feats, rmse, axis=0)
    feats = np.append(feats, spectralBandwidth, axis=0)
    # feats = np.append(feats, spectralContrast, axis=0)
    # feats = np.append(feats, spectralFlatness, axis=0)
    feats = np.append(feats, spectralRolloff, axis=0)
    # feats = np.append(feats, fcpe, axis=0)

    featdf = pd.DataFrame(columns=columns(featureDict),
                          dtype=np.float32, data=feats.T)

    nFrames = len(featdf)
    labels = np.ones(nFrames)*label

    featdf[('label', '0')] = labels
    return featdf
