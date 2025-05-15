import numpy as np
from scipy.io import wavfile
import os

def get_wav_info(wav_file_path):
    """
    Get detailed information about a WAV file.
    Returns a dictionary containing various audio file properties.
    """
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file_path)
    
    # Get basic information
    info = {
        "Filename": os.path.basename(wav_file_path),
        "Sample Rate": f"{sample_rate} Hz",
        "Duration": f"{len(data) / sample_rate:.3f} seconds",
        "Total Samples": len(data),
        "Data Type": str(data.dtype),
        "Channels": 1 if len(data.shape) == 1 else data.shape[1],
    }
    
    # Get amplitude information
    info["Max Amplitude"] = np.max(np.abs(data))
    info["Min Amplitude"] = np.min(data)
    info["Mean Amplitude"] = np.mean(data)
    info["RMS Amplitude"] = np.sqrt(np.mean(np.square(data.astype(float))))
    
    # Calculate bit depth based on data type
    if data.dtype == np.int16:
        info["Bit Depth"] = "16-bit"
    elif data.dtype == np.int32:
        info["Bit Depth"] = "32-bit"
    elif data.dtype == np.float32:
        info["Bit Depth"] = "32-bit float"
    elif data.dtype == np.float64:
        info["Bit Depth"] = "64-bit float"
    else:
        info["Bit Depth"] = f"Other ({data.dtype})"
    
    return info

def print_wav_info(info):
    """
    Print the WAV file information in a formatted way.
    """
    print("\nWAV File Information:")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key:.<30} {value}")
    print("=" * 50)

if __name__ == "__main__":
    # Get file paths from the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Example files to analyze
    files_to_analyze = [
        # os.path.join(parent_dir, 'Edited_recordings/downsampled_48kHz_02-float-peak-norm-DUT-250331_1915-01.Filter.Laplace.-6dBamp-aligned-l3000h60.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/original/02-DUT-aligned-250407_1802.repetitive_multitone_shaped.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/original/02-DUT-aligned-250407_1825.repetitive_multitone_shaped.FFT.Filtered.cropped1440000_predicted_predict_with_buffer.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/6kHz_02-DUT-aligned-250407_1802.repetitive_multitone_shaped.FFT.Filtered.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/6kHz_02-DUT-aligned-250407_1825.repetitive_multitone_shaped.FFT.Filtered.cropped1440000_predicted_predict_with_buffer.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/6kHz_repetitive_multitone_shaped.FFT.Filtered.cropped1440000.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/6kHz_02-DUT-scaled-aligned-250407_1802.repetitive_multitone_shaped.FFT.Filtered.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'LD14/Temp/Aligned/02-DUT-aligned_pink_noise_laplace.FFT.Filtered-250423_2223.wav'),
        # os.path.join(parent_dir, 'LD14/pink_noise_laplace.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'LD14/Temp/Scaled/02-DUT-scaled-aligned_pink_noise_laplace.FFT.Filtered-250423_2223.wav'),
        # os.path.join(parent_dir, '48_new_test', '02-DUT-scaled-aligned-250407_1808.pink_noise_laplace_48kHz.wav'),
        # os.path.join(parent_dir, '48_new_test', 'pink_noise_laplace.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'LD19', 'true_laplace_pinknoise.FFT.Filtered-peak-norm_6kHz.wav'),
        # os.path.join(parent_dir, 'LD19', 'Logorythmic_multitone_burst_6kHz.wav'),
        # os.path.join(parent_dir, 'LD19', 'repetitive_multitone_shaped-peak-norm_6kHz.wav'),
        os.path.join(parent_dir, 'LD19', 'Temp', 'Scaled', '02-DUT-peak-norm-aligned_Logorythmic_multitone_burst-250508_1733.wav'),
        os.path.join(parent_dir, 'LD19', 'Temp', 'Scaled', '02-DUT-peak-norm-aligned_Logorythmic_multitone_burst_6kHz_predicted_180L120N_9db_best-250508_1736.wav'),
        os.path.join(parent_dir, 'LD19', 'Temp', 'Scaled', '02-DUT-peak-norm-aligned_Logorythmic_multitone_burst_predicted_1000LN_9db_best-250508_1736.wav'),
        os.path.join(parent_dir, 'LD19',  'need_to_play', 'Logorythmic_multitone_burst.wav'),
        # os.path.join(parent_dir, 'LD16', 'Recordings', '02-DUT_repetitive_multitone_shaped-peak-norm-250427_1918.wav'),
        # os.path.join(parent_dir, 'LD14', 'Recordings', '02-DUT_repetitive_multitone_shaped-peak-norm-250423_2244.wav'),

        # os.path.join(parent_dir, 'Multitone_analysis/6kHz_02-DUT-peak-norm-aligned-250407_1807.multitone_test_signal_10bin_3khz.FFT.Filtered.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/48k_analysis/02-DUT-peak-norm-aligned-250409_1717_multitone_10bin_predicted.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/48k_analysis/multitone_test_signal_10bin_3khz.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/48k_analysis/pink_noise_laplace.FFT.Filtered.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/48k_analysis/02-DUT-aligned-250409_1717-01_pink_noise_predicted.wav'),
        # os.path.join(parent_dir, 'Multitone_analysis/48k_analysis/02-DUT-scaled-aligned-250409_1717-01_pink_noise_predicted.wav'),
    ]
    
    # Analyze each file
    for wav_file in files_to_analyze:
        if os.path.exists(wav_file):
            info = get_wav_info(wav_file)
            print_wav_info(info)
        else:
            print(f"\nFile not found: {os.path.basename(wav_file)}")