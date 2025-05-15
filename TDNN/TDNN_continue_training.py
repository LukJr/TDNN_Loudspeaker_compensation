import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import glob
from TDNN_traning import SimpleAudioNN, SimpleAudioDataset, set_seed, HAS_DIRECTML

# Try to import torch_directml
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False
    print("torch_directml not found. Install with: pip install torch-directml")

def get_latest_checkpoint(checkpoints_dir):
    """Get the path of the latest checkpoint file."""
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir,  'Good_48k', "1000LN_9db_*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found!")
    
    # Extract epoch numbers and find the latest
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_checkpoint

def continue_training():
    # Set the same parameters as original training
    window_size = 1000
    hidden_size = 1000
    batch_size = 32768 if HAS_DIRECTML else 32  # Try increasing this value  8192
    num_epochs = 2000  # Additional epochs to train
    learning_rate = 0.001  # This will be overridden by the checkpoint's learning rate
    
    # Set random seed
    set_seed(42)
    
    # Get file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    global_dir = os.path.dirname(current_dir)

    input_file = os.path.join(global_dir, 'LD15', 'Temp', 'Scaled', '02-DUT-peak-norm-aligned_true_laplace_pinknoise.FFT.Filtered-peak-norm-250424_2334.wav')
    target_file = os.path.join(global_dir, 'LD15', 'true_laplace_pinknoise.FFT.Filtered-peak-norm.wav')
    
    # Load the dataset
    print("Loading dataset...")
    full_dataset = SimpleAudioDataset(input_file, target_file, window_size=window_size)
    print(f"Dataset size: {len(full_dataset)}")
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )
    
    # Initialize model
    model = SimpleAudioNN(window_size, hidden_size)
    
    # Load the latest checkpoint
    checkpoints_dir = os.path.join(current_dir, "checkpoints")
    latest_checkpoint = get_latest_checkpoint(checkpoints_dir)
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
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
        print(f"Using device: CPU (DirectML not available)")
    
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10,
        verbose=True
    )
    
    # Load optimizer and scheduler states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    print(f"Continuing training from epoch {start_epoch}")
    
    # Pre-load validation data
    val_inputs = []
    val_targets = []
    print("Pre-loading validation data to GPU...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            val_inputs.append(inputs.to(device, non_blocking=True))
            val_targets.append(targets.to(device, non_blocking=True))
    
    # Continue training
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        accumulated_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % 4 == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_loss += accumulated_loss
                accumulated_loss = 0
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
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
        
        # Save checkpoint every 200 epochs
        if (epoch + 1) % 200 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            checkpoint_path = os.path.join(checkpoints_dir, f"1000LN_9db_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Save best model based on validation loss
        if not val_losses or avg_val_loss < min(val_losses[:-1], default=float('inf')):
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            best_model_path = os.path.join(checkpoints_dir, "1000LN_9db_best.pth")
            torch.save(best_checkpoint, best_model_path)
            print(f"New best model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.6f}")
        
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs} | "
              f"Train Loss (RMSE): {np.sqrt(avg_train_loss):.6f} | "
              f"Val Loss (RMSE): {np.sqrt(avg_val_loss):.6f} | "
              f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if HAS_DIRECTML:
            import gc
            gc.collect()
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Continued Training)')
    plt.legend()
    plt.savefig('continued_training_loss.png')
    plt.close()

if __name__ == "__main__":
    continue_training() 