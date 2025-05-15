import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import os

def gcc_phat(sig1, sig2, fs):
    """Compute time delay between sig1 and sig2 using GCC-PHAT."""
    # Compute FFT of both signals
    N = len(sig1) + len(sig2)
    X1 = np.fft.rfft(sig1, n=N)
    X2 = np.fft.rfft(sig2, n=N)

    # Compute the cross power spectrum
    R = X1 * np.conj(X2)
    
    # Normalize (PHAT weighting)
    R /= np.abs(R) + 1e-10  # Avoid division by zero

    # Compute inverse FFT to get cross-correlation
    cc = np.fft.irfft(R, n=N)

    # Find the delay
    max_shift = len(sig1) // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift]))
    delay = np.argmax(cc) - max_shift

    return delay / fs  # Convert to seconds