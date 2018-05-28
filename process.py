import os
import sys 
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def readWave(filename_):    
    sample_rate, signal = wavfile.read(filename_)
    signal = signal[0:int(3.5 * sample_rate)] # 3.5 (s) From Start
    return sample_rate, signal

def preEmphasis(signal_ ,alpha_):
    # y(t) = x(t) - Apha*X(t-1)
    signal = np.append(signal_[0], signal_[1:] - alpha_ * signal_[:-1])
    return signal

def plotWave(signal_, sample_Rate_,filename_):
    x_value = np.arange(0, len(signal_), 1) / float(sample_Rate_)    
    plt.plot(x_value, signal_)    
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.title(filename_)
    plt.box("off")
    plt.grid(True)
    plt.show()
    
def framing_signal(signal_, frame_size_, frame_stride_, sample_rate_):
    frame_length = frame_size_ * sample_rate_
    frame_step = frame_stride_ * sample_rate_
    signal_length = len(signal_)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal_, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames, frame_length

def windowing_frames(frames_, frame_length_):
    frames_ *= np.hamming(frame_length_)
    return frames_

def convert_frames_fft(frames_, NFTT_):
    mag_frames = np.absolute(np.fft.rfft(frames_, NFTT_))
    return mag_frames, NFTT_

def frame_power_spectrum(mag_frames_, NFTT_):
    return ((1.0 / NFTT_) * (mag_frames_ ** 2))

if __name__ == "__main__":
    _filename = sys.argv[1]
    _sample_rate, _signal = readWave(_filename)
    _signal = preEmphasis(_signal, 0.97)
    
    _frames, _frame_length = framing_signal(_signal, 0.025, 0.01, _sample_rate)
    _frames = windowing_frames(_frames, _frame_length)

    _mag_frames, _NFTT = convert_frames_fft(_frames, 512)
    _pow_spectrum = frame_power_spectrum(_mag_frames, _NFTT)