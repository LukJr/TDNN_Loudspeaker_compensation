import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import wavfile
import os
import time

# Set the checkpoint to evaluate
CHECKPOINT_NAME = "LD16_180L140N_1000.pth"

# Try to import torch_directml
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False
    print("torch_directml not found. Using CPU instead.")

# Set fixed seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleAudioDataset(Dataset):
    def __init__(self, input_file, target_file, window_size=80):
        # Load and process audio data
        _, input_audio = wavfile.read(input_file)
        _, target_audio = wavfile.read(target_file)
        
        # Convert to mono if stereo
        if len(input_audio.shape) > 1:
            input_audio = np.mean(input_audio, axis=1)
        if len(target_audio.shape) > 1:
            target_audio = np.mean(target_audio, axis=1)
        
        # Normalize audio
        self.input_audio = input_audio.astype(np.float32) / np.max(np.abs(input_audio))
        self.target_audio = target_audio.astype(np.float32) / np.max(np.abs(target_audio))
        
        self.window_size = window_size
    
    def __len__(self):
        return len(self.input_audio) - self.window_size
    
    def __getitem__(self, idx):
        # Get window of samples
        input_window = self.input_audio[idx:idx + self.window_size]
        target_sample = self.target_audio[idx + self.window_size - 1]
        
        return torch.FloatTensor(input_window), torch.FloatTensor([target_sample])

class SimpleAudioNN(nn.Module):
    def __init__(self, window_size, hidden_size=32):
        super(SimpleAudioNN, self).__init__()
        self.layer1 = nn.Linear(window_size, hidden_size)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        x = self.layer2(x)
        return x

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
    
    def forward(self, x, y):
        return torch.sqrt(torch.mean((x - y) ** 2))

class RelativeError(nn.Module):
    def __init__(self):
        super(RelativeError, self).__init__()
    
    def forward(self, x, y):
        # Calculate L2 norm of difference (numerator)
        diff_norm = torch.norm(x - y, p=2)
        
        # Calculate L2 norm of average (denominator)
        avg_norm = torch.norm((x + y) / 2.0, p=2)
        
        # Avoid division by zero by adding small epsilon
        eps = 1e-8
        return diff_norm / (avg_norm + eps)

def main():
    start_time = time.time()
    
    # Set random seed
    set_seed(42)
    
    # Setup device
    if HAS_DIRECTML:
        try:
            device = torch_directml.device()
            print(f"Using device: DirectML (AMD GPU)")
        except Exception as e:
            print(f"DirectML error: {e}")
            device = torch.device("cpu")
            print(f"Falling back to CPU")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    global_dir = os.path.dirname(current_dir)  # TDNN directory
    
    # Checkpoints are still in the original location
    checkpoint_path = os.path.join(global_dir, "TDNN", "checkpoints", CHECKPOINT_NAME)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Parse checkpoint name to get model parameters
    # Format: ...<window_size>L<hidden_size>N...<epoch>.pth
    parts = CHECKPOINT_NAME.split('_')
    config_part = parts[1]  # e.g., "200L120N"
    window_size = int(config_part.split('L')[0])
    hidden_size = int(config_part.split('L')[1].replace('N', ''))
    print(f"Model config: window_size={window_size}, hidden_size={hidden_size}")
    
    # Get input and target files
    input_file = os.path.join(global_dir, 'LD16', '6kHz', '6kHz_02-DUT-peak-norm-aligned_true_laplace_pinknoise.FFT.Filtered-peak-norm-250424_2334.wav')
    target_file = os.path.join(global_dir, 'LD16', '6kHz', '6kHz_true_laplace_pinknoise.FFT.Filtered-peak-norm.wav')
    
    print(f"Loading dataset...")
    dataset = SimpleAudioDataset(input_file, target_file, window_size=window_size)
    print(f"Dataset size: {len(dataset)}")
    
    # Create data loader
    batch_size = 8192 if HAS_DIRECTML else 1024
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    # Load model
    print(f"Loading model from checkpoint: {CHECKPOINT_NAME}")
    model = SimpleAudioNN(window_size, hidden_size)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Calculate RMSE and Relative Error
    print(f"Calculating RMSE and Relative Error...")
    model.eval()
    rmse_criterion = RMSE().to(device)
    rel_error_criterion = RelativeError().to(device)
    total_rmse = 0.0
    total_rel_error = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            # Calculate both metrics
            rmse_loss = rmse_criterion(outputs, targets)
            rel_error = rel_error_criterion(outputs, targets)
            
            # Sum batch losses
            total_rmse += rmse_loss.item() * inputs.size(0)
            total_rel_error += rel_error.item() * inputs.size(0)
            num_samples += inputs.size(0)
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {num_samples} samples...")
    
    rmse = total_rmse / num_samples
    rel_error = total_rel_error / num_samples
    
    elapsed_time = time.time() - start_time
    print(f"\nResults for {CHECKPOINT_NAME}:")
    print(f"RMSE: {rmse:.6f}")
    print(f"Relative Error: {rel_error:.6f}")
    print(f"Evaluation completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 