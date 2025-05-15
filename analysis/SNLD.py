import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

#########################
# Parameters
#########################
N = 12000          # Number of samples per period (T = 0.25 s at 48 kHz)
P = 6              # Number of analysis periods per realization (first period is for settling)
P_plus_one = P + 1 # Total periods per realization

# Define the excited frequency range (in Hz)
f_min = 60.0
f_max = 3000.0

#########################
# File Loading Functions
#########################
def load_wav(filename):
    """Load a WAV file and return sampling rate and mono signal."""
    fs, data = wavfile.read(filename)
    if data.ndim > 1:
        data = data[:, 0]
    return fs, data

#########################
# Signal Segmentation Functions
#########################
def split_into_periods(signal, period_length):
    """Split signal into periods of length 'period_length' samples."""
    total_periods = len(signal) // period_length
    periods = signal[:total_periods * period_length].reshape(total_periods, period_length)
    return periods

def discard_settling_periods(periods, block_size, P):
    """
    Discard the first (settling) period in each block of 'block_size' periods.
    Return an array of analysis periods of shape (M, P, N).
    """
    num_realizations = periods.shape[0] // block_size
    analysis_periods_list = []
    for m in range(num_realizations):
        block = periods[m * block_size : (m + 1) * block_size]
        analysis_block = block[1:]  # discard settling period
        analysis_periods_list.append(analysis_block)
    analysis_periods = np.vstack(analysis_periods_list)
    return analysis_periods.reshape(num_realizations, P, periods.shape[1])

#########################
# FFT and Computation Functions
#########################
def compute_fft_periods(periods):
    """
    Compute FFT along each period and apply normalization by dividing by N.
    """
    return np.fft.fft(periods, axis=2) / N

def compute_G(Y_fft, U_fft):
    """
    Compute G[m,p] = Y[m,p] / U[m] for each period in each realization.
    """
    M, P, N = Y_fft.shape
    G = np.empty((M, P, N), dtype=complex)
    for m in range(M):
        for p in range(P):
            G[m, p, :] = Y_fft[m, p, :] / U_fft[m, :]
    return G

def average_over_periods(G):
    """
    Compute the average of G over the analysis periods for each realization.
    (Equation (8): G[m] = (1/P) sum_{p=1}^P G[m,p])
    """
    return np.mean(G, axis=1)

def compute_noise_variance(G, G_avg, k_min, k_max):
    """
    Compute noise variance for each realization using only excited frequency bins.
    (Equation (9) applied over frequency bins k_min:k_max)
    """
    M, P, _ = G.shape
    sigma2_nz = np.empty(M)
    excited_band = slice(k_min, k_max+1)
    for m in range(M):
        diff = G[m, :, excited_band] - G_avg[m, excited_band]  # shape (P, number_of_bins)
        num_bins = (k_max - k_min + 1)
        sigma2_nz[m] = np.sum(np.abs(diff) ** 2) / (P * (P - 1) * num_bins)
    return sigma2_nz

def compute_overall_FRF(G_avg):
    """
    Compute the overall Best Linear Approximation (BLA) FRF.
    (Equation (10): GBLA = (1/M) sum_{m=1}^M G[m])
    """
    return np.mean(G_avg, axis=0)

def compute_FRF_variance_per_freq(G_avg):
    """
    Compute frequency-dependent FRF variance across realizations.
    Returns:
      sigma2_BLA_freq: array of shape (N,) containing variance per frequency bin.
      GBLA_freq: the overall average FRF per frequency bin.
    (Using: sigma^2_BLA(f) = (1/(M(M-1))) sum_{m=1}^M |G[m,f] - GBLA(f)|^2)
    """
    M, N = G_avg.shape
    GBLA_freq = np.mean(G_avg, axis=0)  # shape (N,)
    sigma2_BLA_freq = np.zeros(N)
    for f in range(N):
        diffs = np.abs(G_avg[:, f] - GBLA_freq[f])**2
        sigma2_BLA_freq[f] = np.sum(diffs) / (M * (M - 1)) if M > 1 else 0
    return sigma2_BLA_freq, GBLA_freq

def compute_noise_variance_per_freq(G, G_avg):
    """
    Compute frequency-dependent noise variance for each realization and average over M.
    For each frequency bin f and realization m:
       sigma2_nz(m,f) = (1/(P*(P-1))) sum_{p=1}^P |G[m,p,f] - G_avg[m,f]|^2
    Then average across m.
    Returns:
      sigma2_BLA_nz_freq: an array of shape (N,)
    """
    M, P, N = G.shape
    sigma2_nz_freq = np.zeros((M, N))
    for m in range(M):
        for f in range(N):
            diffs = np.abs(G[m, :, f] - G_avg[m, f])**2
            sigma2_nz_freq[m, f] = np.sum(diffs) / (P * (P - 1)) if P > 1 else 0
    sigma2_BLA_nz_freq = np.mean(sigma2_nz_freq, axis=0)
    return sigma2_BLA_nz_freq

def compute_SNLD_spectrum(sigma2_BLA_freq, sigma2_BLA_nz_freq, U0_full, GBLA_freq, M):
    """
    Compute the frequency-dependent nonlinear distortion spectrum.
    For each frequency bin f, define:
      sigma_S^2(f) = max( sigma2_BLA(f) - sigma2_BLA_nz(f), 0 )
      sigma_S(f) = sqrt( sigma_S^2(f) )
      Y_S(f) = M * sigma_S(f) * U0(f)   [Equation (14)]
    Returns Y_S (nonlinear distortion spectrum per frequency) as well as:
      linear_spectrum = |GBLA(f) * U0(f)|,
      noise_spectrum = sqrt(sigma2_BLA_nz_freq) * U0(f)
    """
    sigma2_S_freq = np.maximum(sigma2_BLA_freq - sigma2_BLA_nz_freq, 0)
    sigma_S_freq = np.sqrt(sigma2_S_freq)
    Y_S_freq = M * sigma_S_freq * U0_full  # Nonlinear distortion spectrum
    linear_spectrum = np.abs(GBLA_freq * U0_full)
    noise_spectrum = np.sqrt(sigma2_BLA_nz_freq) * U0_full
    return Y_S_freq, linear_spectrum, noise_spectrum

#########################
# Main Analysis Function
#########################
def main():
    # Filenames (modify as needed)
    # original_file = "repetitive_multitone_shaped-peak-norm.wav"
    # recorded_file = "./Temp/Scaled/02-DUT-peak-norm-aligned_repetitive_multitone_shaped-peak-norm_predicted_3000-250427_1839.wav"
    # recorded_file = "./Temp/Scaled/02-DUT-peak-norm-aligned_repetitive_multitone_shaped-peak-norm-250427_1918.wav"
    # recorded_file = "./Temp/Scaled/02-DUT-peak-norm-aligned_repetitive_multitone_shaped-peak-norm_predicted_3000-250427_1839.wav"
    # original_file = "repetitive_multitone_shaped-peak-norm.wav"
    original_file = "repetitive_multitone_shaped-peak-norm_6kHz_48kHz.wav"
    # original_file = "./need_to_play/repetitive_multitone_shaped-peak-norm_predicted_1000LN_9db_best.wav"
    # recorded_file = "./Temp/Aligned/02-DUT-aligned_repetitive_multitone_shaped-peak-norm-250508_1737.wav"
    # recorded_file = "./Temp/Aligned/02-DUT-aligned_EQ_repetitive_multitone_shaped-peak-norm-250508_1754.wav"
    # recorded_file = "./Temp/Aligned/02-DUT-aligned-correct_repetitive_multitone_shaped-peak-norm_predicted_1000LN_9db_best-250508_1744.wav"
    # recorded_file = "./Temp/Aligned/02-DUT-aligned_EQ_repetitive_multitone_shaped-peak-norm_6kHz-250508_1755.wav"
    recorded_file = "./Temp/Aligned/02-DUT-aligned_repetitive_multitone_shaped-peak-norm_6kHz_predicted_180L120N_9db_best-250508_1742.wav"
    
    
    # Load WAV files.
    fs_input, original_signal = load_wav(original_file)
    fs_recorded, recorded_signal = load_wav(recorded_file)
    
    if fs_input != fs_recorded:
        print("Error: Sampling rates of input and recorded files do not match!")
        return
    
    fs = fs_input
    print(f"Sampling rate: {fs} Hz")
    
    # Split signals into periods.
    original_periods = split_into_periods(original_signal, N)
    recorded_periods = split_into_periods(recorded_signal, N)
    
    # Discard the first (settling) period in each realization.
    original_analysis = discard_settling_periods(original_periods, P_plus_one, P)
    recorded_analysis = discard_settling_periods(recorded_periods, P_plus_one, P)
    
    M_realizations = original_analysis.shape[0]
    print(f"Number of realizations: {M_realizations}, each with {P} analysis periods.")
    
    # Determine DFT resolution and excited frequency indices.
    T = N / fs         # Period length (0.25 s)
    f0 = 1 / T         # Frequency resolution (4 Hz)
    k_min = int(np.ceil(f_min / f0))
    k_max = int(np.floor(f_max / f0))
    print(f"Excited frequency band: {f_min} Hz to {f_max} Hz corresponds to bins {k_min} to {k_max}.")
    
    # Compute normalized FFTs.
    U_fft_analysis = compute_fft_periods(original_analysis)   # shape: (M, P, N)
    Y_fft_analysis = compute_fft_periods(recorded_analysis)     # shape: (M, P, N)
    
    # Assume within each realization the input FFT is constant; take first analysis period.
    U_fft = U_fft_analysis[:, 0, :]  # shape: (M, N)
    
    # Compute average input amplitude spectrum (full).
    U0_full = np.mean(np.abs(U_fft), axis=0)   # shape: (N,)
    
    # For plotting, create full frequency axis.
    freq_axis = np.linspace(0, fs, N, endpoint=False)
    
    # Optional: plot full stimulus spectrum vs. excited band.
    excited_band = slice(k_min, k_max+1)
        
    # Compute FRF for each period.
    G = compute_G(Y_fft_analysis, U_fft)  # shape: (M, P, N)
    G_avg = average_over_periods(G)         # shape: (M, N)
    
    # Compute frequency-dependent FRF variance across realizations.
    sigma2_BLA_freq, GBLA_freq = compute_FRF_variance_per_freq(G_avg)  # arrays of shape (N,)
    
    # Compute frequency-dependent noise variance (averaged across m).
    sigma2_BLA_nz_freq = compute_noise_variance_per_freq(G, G_avg)     # array of shape (N,)
    
    # Compute global variance measures (restricted to excited band) if needed.
    sigma2_BLA = np.sum(sigma2_BLA_freq[excited_band]) / (k_max - k_min + 1)
    sigma2_BLA_nz = np.sum(sigma2_BLA_nz_freq[excited_band]) / (k_max - k_min + 1)
    
    # Compute frequency-dependent nonlinear variance and spectra.
    Y_S_freq, linear_spectrum, noise_spectrum = compute_SNLD_spectrum(sigma2_BLA_freq, sigma2_BLA_nz_freq, U0_full, GBLA_freq, M_realizations)
    
    # Restrict spectra and frequency axis to the excited band.
    freq_excited = freq_axis[excited_band]
    Y_S_excited = Y_S_freq[excited_band]
    linear_excited = linear_spectrum[excited_band]
    noise_excited = noise_spectrum[excited_band]
    

    # Suppose we already have: freq_excited, linear_excited, Y_S_excited, noise_excited
    # which are linear magnitudes in "some units." We'll convert them to dB.

    # [Optional] If you have calibration factor alpha to get Pascals: 
    # linear_excited_pa = alpha * linear_excited
    # Y_S_excited_pa    = alpha * Y_S_excited
    # noise_excited_pa  = alpha * noise_excited
    # Then do 20*log10(... / p_ref) to get dB SPL.

    # Create a mask: if nonlinear distortion is below noise, set it to NaN.
    Y_S_excited_plot = np.where(Y_S_excited < noise_excited, np.nan, Y_S_excited)

    # Optionally, convert to dB.
    eps = 1e-20  # small constant to avoid log(0)
    linear_dB = 20 * np.log10(linear_excited + eps)
    Y_S_dB    = 20 * np.log10(Y_S_excited_plot + eps)
    noise_dB  = 20 * np.log10(noise_excited + eps)

    # Plot the curves.
    plt.figure(figsize=(10, 6))
    plt.plot(freq_excited, linear_dB, label="Linear FRF (BLA)", linewidth=2)
    plt.plot(freq_excited, Y_S_dB,    label="Nonlinear Distortion", linewidth=2)
    plt.plot(freq_excited, noise_dB,  label="Noise", linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Frequency-dependent Response Components")
    plt.legend()
    plt.grid(True)
    
    # Set fixed y-axis limits
    plt.ylim(-150, -40)  # Set y-axis range from -150 dB to -40 dB
    
    # Improve grid visibility
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    plt.show()
    
    # Print out some global results for comparison.
    SNLD_global = np.linalg.norm(Y_S_freq[excited_band]) / np.linalg.norm((GBLA_freq * U0_full)[excited_band])
    print("Analysis Results:")
    print("-----------------")
    print(f"Global sigma^2_BLA (excited band): {sigma2_BLA}")
    print(f"Global sigma^2_BLA,nz (excited band): {sigma2_BLA_nz}")
    print("Global SNLD (dimensionless):", SNLD_global)
    print("Global SNLD (%):", SNLD_global * 100)
    
if __name__ == "__main__":
    main()
