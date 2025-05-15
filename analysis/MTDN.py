import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import glob

def load_mono(fn):
    fs, d = wavfile.read(fn)
    if d.ndim>1: d = d[:,0]
    return fs, d.astype(np.float32)

def calculate_mtnd(stim, rec, fs, primary_freqs, f_min=20, f_max=5000, notch_width_hz=5, window_size_hz=10, reference_rms=None):
    """Calculate MTND for a single stimulus and recording pair"""
    # Get original length
    N = len(stim)
    
    # Use custom FFT size for better frequency resolution
    N_custom = 9600000
    f = np.fft.rfftfreq(N_custom, 1/fs)
    A_stim = np.abs(np.fft.rfft(stim, n=N_custom))
    A_rec = np.abs(np.fft.rfft(rec, n=N_custom))
    
    # Distortion = record – stim (allow negatives)
    D = A_rec - A_stim
    D_abs = np.abs(D)  # Absolute distortion
    
    # Applying notch filter ±notch_width_hz around primary frequencies
    D_clean = D_abs.copy()
    for tone in primary_freqs:
        notch = (f >= tone - notch_width_hz) & (f <= tone + notch_width_hz)
        D_clean[notch] = 0
    
    # Computing MTND with a sliding window
    df = f[1] - f[0]
    half_w = int(round((window_size_hz / 2) / df))
    centers = np.arange(f_min, f_max + window_size_hz / 2, window_size_hz)
    mtnd = []
    
    for fc in centers:
        idx = int(round(fc / df))
        lo = max(0, idx - half_w)
        hi = min(len(D_clean), idx + half_w + 1)
        window = D_clean[lo:hi]
        rms = np.sqrt(np.mean(window**2))
        mtnd.append(20 * np.log10(rms / reference_rms))
    
    return centers, mtnd

def smooth_mtnd(frequencies, mtnd_values, smoothing_factor=0.05):
    """
    Apply frequency-dependent smoothing to MTND values.
    Higher frequencies get more smoothing.
    
    Args:
        frequencies: Array of frequency centers
        mtnd_values: Array of MTND values
        smoothing_factor: Base smoothing factor (0-1)
        
    Returns:
        Smoothed MTND values
    """
    smoothed = np.array(mtnd_values).copy()
    n = len(frequencies)
    
    # Calculate frequency-dependent window sizes (larger at higher frequencies)
    for i in range(1, n-1):
        # Calculate window size based on frequency (more smoothing at higher frequencies)
        # This formula creates a gradually increasing window size
        freq_ratio = np.log10(frequencies[i] / frequencies[0]) + 1
        window_size = max(3, int(round(smoothing_factor * freq_ratio * 10)))
        
        # Ensure window doesn't exceed array bounds
        half_window = min(i, n-i-1, window_size//2)
        if half_window < 1:
            continue
            
        # Apply smoothing with centered window
        window_start = i - half_window
        window_end = i + half_window + 1
        smoothed[i] = np.mean(mtnd_values[window_start:window_end])
    
    return smoothed

# Setup parameters
fs = 48000
primary_freqs = np.array([67, 90.5, 173.5, 198, 303, 547, 684.5, 1224.5, 1974.5, 2435.5])
f_min, f_max = 20, 5000
notch_width_hz = 5
window_size_hz = 10
smoothing_factor = 0.3  # Increased from 0.1 for more smoothing

# Load stimulus file
stim_file = "Logorythmic_multitone_burst.wav"
fs, stim = load_mono(stim_file)

# Manual specification of DUT files and their display names
dut_files_and_names = [
    # Format: (file_path, display_name)
    ("./Temp/Scaled/02-DUT-rms-norm-aligned_Logorythmic_multitone_burst-250508_1733.wav", "Normal"),
    ("./Temp/Scaled/02-DUT-rms-norm-aligned_Logorythmic_multitone_burst_6kHz_predicted_180L120N_9db_best-250508_1736.wav", "6kHz"),
    ("./Temp/Scaled/02-DUT-rms-norm-aligned_Logorythmic_multitone_burst_predicted_1000LN_9db_best-250508_1736.wav", "48kHz"),
]

# Alternatively, uncomment and modify this to use glob pattern
# dut_pattern = "./Temp/Scaled/*Logorythmic_multitone_burst*.wav"
# dut_files = glob.glob(dut_pattern)
# dut_files_and_names = [(file, os.path.basename(file).split('-')[1]) for file in dut_files]

if not dut_files_and_names:
    print("No DUT files specified. Please add files to the dut_files_and_names list.")
    exit()

# Set up the plot
plt.figure(figsize=(12, 6))

# Store average values for reporting
average_mtnd_values = {}

# Process each DUT file
for i, (dut_file, display_name) in enumerate(dut_files_and_names):
    # Check if file exists
    if not os.path.exists(dut_file):
        print(f"Warning: File {dut_file} not found, skipping.")
        continue
        
    # Load DUT file
    fs2, rec = load_mono(dut_file)
    assert fs == fs2, f"Sample rate mismatch between stimulus ({fs}Hz) and DUT ({fs2}Hz)"

    # Calculate RMS of the recording
    rms = np.sqrt(np.mean(rec**2))
    
    # Calculate MTND
    centers, mtnd = calculate_mtnd(stim, rec, fs, primary_freqs, f_min, f_max, notch_width_hz, window_size_hz, rms)
    
    # Apply frequency-dependent smoothing
    smoothed_mtnd = smooth_mtnd(centers, mtnd, smoothing_factor)
    
    # Calculate average MTND (in dB) across the frequency range
    avg_mtnd = np.mean(mtnd)
    average_mtnd_values[display_name] = avg_mtnd
    
    # Plot with a different color for each device
    plt.plot(centers, smoothed_mtnd, label=f"{display_name} (avg: {avg_mtnd:.2f} dB)", color=f'C{i%10}')

# Print the average MTND values
print("\nAverage MTND Values (dB SPL):")
print("-" * 40)
for name, value in average_mtnd_values.items():
    print(f"{name}: {value:.2f} dB")
print("-" * 40)

# Add vertical lines at the primary frequencies
# for freq in primary_freqs:
#     plt.axvline(x=freq, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
#     plt.text(freq, plt.ylim()[1]-3, f'{freq}', rotation=90, va='top', ha='center', fontsize=8)

# Configure plot appearance
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Distortion amplitude relative to RMS (dB)')
plt.title('MTND (Notched Primary Tones ±5 Hz, Window = 10 Hz)')
plt.grid(True, which='both', ls='--')

# Improve frequency axis labels
freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000]
plt.xticks(freq_ticks, [str(x) for x in freq_ticks])
# plt.gca().xaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}' if x >= 100 else f'{x:.1f}'))
plt.gca().xaxis.set_minor_locator(plt.LogLocator(subs=np.arange(2, 10), numticks=20))
plt.grid(True, which='minor', axis='x', alpha=0.2)

plt.xlim(f_min, f_max)
plt.legend()
plt.tight_layout()
plt.show()
