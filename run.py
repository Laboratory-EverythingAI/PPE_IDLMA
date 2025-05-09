#!/usr/bin/env python
#title           : helpers.py
#description     : helper functions for AARIP feature extraction and classification
#author          : Patrik Lechner <ptrk.lechner@gmail.com>
#date            : Mar 2019
#version         : 0.1
#usage           :
#notes           :
#python_version  : 3.6.3
#=======================================================================
import scipy.signal as sig
import numpy as np


def smooLin(x, N=21):
    return sig.savgol_filter(x, N, 2)


def freqToBin(freqs, sr, frameSize):
    "given the frequencies(array), the sample rate and the frame size; calculates Bin numbers."
    if type(freqs) == list:
        freqs = np.array(freqs)

    nyq = sr / 2
    bins = (np.round((freqs / nyq) * frameSize / 2.))
    bins = bins.astype(int)
    return bins


def binToFreq(bins, sr, frameSize):
    "given the bins as a list, the sample rate and the frame size; calculates Frequency array."
    if type(bins) == int or type(bins) == float:
        f = (bins/float(frameSize/2.))*sr/2
        freqs = f
    else:
        freqs = []
        for i in range(len(bins)):
            f = (bins[i] / float(frameSize / 2.)) * sr / 2
            freqs.append(f)

    return freqs
