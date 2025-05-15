import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import random

# Set fixed seeds for reproducibility
def set_seed(seed=42):
    """
    Set fixed seeds for reproducibility across all random number generators.
    
    Args:
        seed: Integer seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducible results")

try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False
    print("torch_directml not found. Install with: pip install torch-directml")

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
    def __init__(self, window_size, hidden_size):
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

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    if HAS_DIRECTML:
        try:
            device = torch_directml.device()
            print(f"Using device: DirectML (AMD GPU)")
            import gc
            gc.collect()
        except Exception as e:
            print(f"DirectML error: {e}")
            device = torch.device("cpu")
            print(f"Falling back to CPU")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU (DirectML not available)")
    
    model = model.to(device)
    model.train()
    
    # Get the current script directory (TDNN folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create checkpoints and models directories inside TDNN
    checkpoints_dir = os.path.join(current_dir, "checkpoints")
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    criterion = RMSE().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10,
        verbose=True       # Print message when LR is reduced
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training on device: {device}")
    print(f"Batch size: {next(iter(train_loader))[0].shape[0]}")
    
    # Pre-allocate tensors for validation to avoid memory fragmentation
    val_inputs = []
    val_targets = []
    print("Pre-loading validation data to GPU...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            val_inputs.append(inputs.to(device, non_blocking=True))
            val_targets.append(targets.to(device, non_blocking=True))
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Process multiple batches at once for better GPU utilization
        accumulated_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            accumulated_loss += loss.item()
            
            # Update every 4 batches or at the end of epoch
            if (batch_idx + 1) % 4 == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_loss += accumulated_loss
                accumulated_loss = 0
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation using pre-loaded data
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in zip(val_inputs, val_targets):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 1000 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            checkpoint_path = os.path.join(checkpoints_dir, f"LD16_180L140N_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss (RMSE): {avg_train_loss:.6f} | "
              f"Val Loss (RMSE): {avg_val_loss:.6f} | "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        gc.collect()
    
    return train_losses, val_losses

if __name__ == "__main__":
    # Set random seed
    set_seed(42)  # Change this value to get different but reproducible results
    
    # Parameters
    window_size = 180
    hidden_size = 120
    batch_size = 8192 if HAS_DIRECTML else 32  # Much larger batch size for GPU 8192
    num_epochs = 1000
    learning_rate = 0.001
    
    # Get file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(current_dir)
    input_file = os.path.join(code_dir, 'LD16', '6kHz', '6kHz_02-DUT-peak-norm-aligned_true_laplace_pinknoise.FFT.Filtered-peak-norm-250424_2334.wav')
    target_file = os.path.join(code_dir, 'LD16', '6kHz', '6kHz_true_laplace_pinknoise.FFT.Filtered-peak-norm.wav')
    # input_file = os.path.join(code_dir, 'Edited_recordings', 'downsampled_48kHz_02-float-peak-norm-DUT-250331_1915-01.Filter.Laplace.-6dBamp-aligned-l3000h60.wav')
    # target_file = os.path.join(code_dir, 'TestSignals', 'downsampled_48kHz_pink_noise_laplace.66Hz.HPF_5kHz.LPF-l3000h60-peak-norm.wav')
    

    
    print("Loading dataset...")
    full_dataset = SimpleAudioDataset(input_file, target_file, window_size=window_size)
    print(f"Dataset size: {len(full_dataset)}")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True  # Drop incomplete batches for better GPU utilization
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )
    
    # Create and train model
    model = SimpleAudioNN(window_size, hidden_size)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                         num_epochs=num_epochs, learning_rate=learning_rate)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('simple_nn_loss.png')
    plt.close() 