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

def fb(pow_spectrum_,sample_rate_, NFFT_, nfilt_):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate_ / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt_ + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    _bin = np.floor((NFFT_ + 1) * hz_points / sample_rate_)

    fbank = np.zeros((nfilt_, int(np.floor(NFFT_ / 2 + 1))))
    for m in range(1, nfilt_ + 1):
        f_m_minus = int(_bin[m - 1])   # left
        f_m = int(_bin[m])             # centersample_rate_
        f_m_plus = int(_bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - _bin[m - 1]) / (_bin[m] - _bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (_bin[m + 1] - k) / (_bin[m + 1] - _bin[m])
    filter_banks = np.dot(pow_spectrum_, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks

def mfcc(filter_bank_, num_ceps_, cep_lifter_):
    mfccframes = dct(filter_bank_, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps_ + 1)]
    (_,  ncoeff) = mfccframes.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter_ / 2) * np.sin(np.pi * n / cep_lifter_)
    mfccframes *= lift 
    return mfccframes

if __name__ == "__main__":
    _filename = sys.argv[1]
    _sample_rate, _signal = readWave(_filename)
    _signal = preEmphasis(_signal, 0.97)
    
    _frames, _frame_length = framing_signal(_signal, 0.025, 0.01, _sample_rate)
    _frames = windowing_frames(_frames, _frame_length)

    _mag_frames, _NFTT = convert_frames_fft(_frames, 512)

    _pow_spectrum = frame_power_spectrum(_mag_frames, _NFTT)
    _filter_banks = fb(_pow_spectrum, _sample_rate, _NFTT, 40)
    _mfcc = mfcc(_filter_banks, 12, 40)    

    print(_mfcc[0])