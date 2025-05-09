#!/usr/bin/env python
#title           : customFeatures.py
#description     : custom audio features for AARIP feature extraction and classification
#author          : Patrik Lechner <ptrk.lechner@gmail.com>
#date            : Mar 2019
#version         : 0.1
#usage           :
#notes           :
#python_version  : 3.6.3
#=======================================================================


import librosa
import matplotlib.pyplot as plt
from helpers import * 
import numpy as np



def filteredEnergyfeature(timeDomian, sr = 48000, plot=False, frameSize=2**14, trim=[5900, 8800]):
    stft = librosa.core.stft(timeDomian, n_fft=frameSize)
    binTrim = (freqToBin(trim[0], sr, frameSize),
               freqToBin(trim[1], sr, frameSize))

    trimmed = abs(stft[binTrim[0]:binTrim[1], :])
    energy = np.average(trimmed, axis=1)

    filtered = smooLin(energy)
    hzAxis = np.linspace(trim[0], trim[1], len(filtered))
    if plot:
        plt.plot(hzAxis, filtered)
        plt.show()

    return hzAxis, librosa.core.amplitude_to_db(filtered)


def filteredCepstrumPeakEnergy(x, plot=True, label='No label', frameSize=2**14, sr=48000):
    '''Takes a time domain signal, applies the stft, trims it to acertain frequency spectrum and calculates the pseudo-cepstrum via another fft and trims that again. after al that the average is taken
    
    Arguments:
        x {np.array} -- time domain signal
    
    Keyword Arguments:
        plot {bool} -- Wether to plot also (default: {True})
        label {str} -- a label used for plotting (default: {'No label'})
        frameSize {int} -- stft frame size (default: {2**14})
        sr {int} -- sampling rate (default: {48000})
    
    Returns:
        [float] -- a signle value describing the energy at the trimmed down fft and trimmed down cepstrum
    '''


    freqTrim = np.array([17000, 22000])
    binTrim = freqToBin(freqTrim, sr, frameSize)
    stft = librosa.core.stft(x, n_fft=frameSize)
    trimmed = (stft[binTrim[0]:binTrim[1], :])
    pseudoCep = abs(np.fft.fft(trimmed, axis=1))
    if plot:
        avgCep = smooLin(pseudoCep)
        plt.plot(avgCep[0:200], label=label)
    return np.average(pseudoCep[0:200, :], axis=0)


def filteredCepstrumPeakEnergyFromSpec(S, sr = 48000, frameSize=2**14, plot=False, label='no label'):

    freqTrim = np.array([17000, 22000])
    binTrim = freqToBin(freqTrim, sr, frameSize)
    # stft = librosa.core.stft(x, n_fft=frameSize)
    trimmed = (S[binTrim[0]:binTrim[1], :])
    pseudoCep = abs(np.fft.fft(trimmed, axis=1))
    if plot:
        avgCep = smooLin(pseudoCep)
        plt.plot(avgCep[0:200], label=label)
    return np.average(pseudoCep[0:200, :], axis=0)
