import numpy as np
from scipy.io import wavfile
import os
from GCC_with_PHAT import gcc_phat

def align_and_save_dut(original_file, dut_file, output_file):
    """
    Align DUT recording with original signal and save the aligned version.
    Trims the non-overlapping parts to ensure perfect alignment.
    """
    # Load the files
    fs_orig, original = wavfile.read(original_file)
    fs_dut, dut = wavfile.read(dut_file)
    
    # Verify sampling rates match
    if fs_orig != fs_dut:
        raise ValueError("Sampling rates must match!")
    
    # Store original lengths for reporting
    original_len = len(original)
    dut_len = len(dut)
    
    # Calculate the delay using GCC-PHAT
    delay_samples = int(gcc_phat(original, dut, fs_orig) * fs_orig)
    
    # Determine the overlapping region
    if delay_samples > 0:
        # DUT signal starts later than original
        aligned_dut = dut[:-delay_samples] if delay_samples < len(dut) else dut
        trimmed_original = original[delay_samples:delay_samples + len(aligned_dut)]
    else:
        # DUT signal starts earlier than original
        delay_samples = abs(delay_samples)
        aligned_dut = dut[delay_samples:] if delay_samples < len(dut) else dut
        trimmed_original = original[:len(aligned_dut)]
    
    # Ensure both signals have the same length
    min_length = min(len(aligned_dut), len(trimmed_original))
    aligned_dut = aligned_dut[:min_length]
    trimmed_original = trimmed_original[:min_length]
    
    # Save the aligned DUT signal
    wavfile.write(output_file, fs_orig, aligned_dut)
    
    # Return alignment information
    return {
        'duration': min_length / fs_orig,
        'delay_samples': delay_samples,
        'delay_seconds': delay_samples / fs_orig,
        'original_length': original_len,
        'dut_length': dut_len,
        'aligned_length': min_length,
        'length_diff_samples': abs(original_len - dut_len),
        'length_diff_seconds': abs(original_len - dut_len) / fs_orig,
        'sample_rate': fs_orig
    }

if __name__ == "__main__":
    # Get file paths from the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input files
    original_file = os.path.join(parent_dir, 'LD19', 'need_to_play','true_laplace_pinknoise.FFT.Filtered-peak-norm.wav')
    dut_file = os.path.join(parent_dir, 'LD19', 'Temp', '02-DUT_true_laplace_pinknoise.FFT.Filtered-peak-norm_predicted_1000LN_9db_best-250508_1751.wav')
    
    # Create output filename by inserting 'aligned-' after 'DUT-'
    dut_filename = os.path.basename(dut_file)
    aligned_filename = dut_filename.replace('DUT', 'DUT-aligned')
    output_file = os.path.join(parent_dir, 'LD19', 'Temp', 'Aligned', aligned_filename)
    
    # Align and save
    result = align_and_save_dut(original_file, dut_file, output_file)
    print(f"Aligned DUT file saved as: {aligned_filename}")
    print(f"\nAlignment Information:")
    print(f"Duration of aligned signal: {result['duration']:.3f} seconds ({result['aligned_length']} samples)")
    print(f"Time delay between signals: {result['delay_seconds']:.3f} seconds ({result['delay_samples']} samples)")
    print(f"Original file length: {result['original_length']} samples ({result['original_length']/result['sample_rate']:.3f} seconds)")
    print(f"DUT file length: {result['dut_length']} samples ({result['dut_length']/result['sample_rate']:.3f} seconds)")
    print(f"Length difference: {result['length_diff_samples']} samples ({result['length_diff_seconds']:.3f} seconds)") 