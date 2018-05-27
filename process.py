import os
import sys
import numpy
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def readWave(inputFilename):    
    sample_rate, signal = wavfile.read(inputFilename)
    signal = signal[0:int(3.5 * sample_rate)] # 3.5 Search From Start
    return sample_rate, signal

def plotWave(inputSignal, inputSample_Rate,inputfilename):
    x_value = numpy.arange(0, len(inputSignal), 1) / float(inputSample_Rate)    
    plt.plot(x_value, inputSignal)    
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.title(inputfilename)
    plt.show()

if __name__ == "__main__":
    _filename = sys.argv[1]
    _sample_rate, _signal = readWave(_filename)
    plotWave(_signal, _sample_rate, _filename)