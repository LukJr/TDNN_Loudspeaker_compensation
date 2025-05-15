import numpy as np
from scipy.io import wavfile
import os

def get_peak_amplitude(data):
    """Calculate the peak amplitude of the signal."""
    return np.max(np.abs(data.astype(float)))

def scale_to_peak(input_file):
    """
    Scale the amplitude of input_file to have a peak amplitude of 1.0.
    Ensures no clipping by leaving a small headroom.
    """
    # Read the file
    fs, data = wavfile.read(input_file)
    
    # Calculate peak value
    peak = get_peak_amplitude(data)
    
    # Calculate scaling factor (leave 1% headroom)
    scale_factor = 0.99 / peak
    
    print(f"\nScaling Information:")
    print("=" * 50)
    print(f"Input Peak: {peak}")
    print(f"Scale factor: {scale_factor}")
    print(f"Target Peak: 0.99")
    
    # Scale the data
    scaled_data = data * scale_factor
    
    # Create new filename
    dir_path = os.path.dirname(input_file)
    filename = os.path.basename(input_file)
    
    # Insert 'peak-norm-' before 'DUT-' in the filename
    if '-DUT-' in filename:
        new_filename = filename.replace('DUT-', 'DUT-peak-norm-')
    else:
        # If -DUT- is not in filename, append -peak-norm before extension
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}-peak-norm{ext}"
    
    output_file = os.path.join(dir_path, new_filename)
    
    # Save scaled file
    wavfile.write(output_file, fs, scaled_data.astype(data.dtype))
    
    return output_file

def print_scaling_result(input_file, output_file):
    """
    Print information about the scaling process.
    """
    # Get information about both files
    _, input_data = wavfile.read(input_file)
    _, output_data = wavfile.read(output_file)
    
    print("\nAmplitude Scaling Results:")
    print("=" * 50)
    print(f"Input file: {os.path.basename(input_file)}")
    print(f"Input Peak: {get_peak_amplitude(input_data)}")
    print(f"\nOutput file: {os.path.basename(output_file)}")
    print(f"Output Peak: {get_peak_amplitude(output_data)}")
    print("=" * 50)

if __name__ == "__main__":
    # Get file paths from the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input file
    input_file = os.path.join(parent_dir, 'LD19', 'Temp', 'Aligned', '02-DUT-aligned_repetitive_multitone_shaped-peak-norm_6kHz-250508_1741.wav')
    
    # Scale file
    if os.path.exists(input_file):
        output_file = scale_to_peak(input_file)
        print_scaling_result(input_file, output_file)
        print("\nScaling completed successfully!")
    else:
        print(f"\nFile not found: {os.path.basename(input_file)}") 