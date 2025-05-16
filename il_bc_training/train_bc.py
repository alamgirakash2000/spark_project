import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os # For joining paths

# Remove: from data_processor import load_and_process_dataset
# We will load .pt files directly

# --- Configuration ---
# DATASET_DIR = "collected_data" # No longer needed for raw data here
PROCESSED_DATA_DIR = "processed_tensors" # Directory where .pt files are saved
FEATURES_X_PATH = os.path.join(PROCESSED_DATA_DIR, 'features_X.pt')
ACTIONS_Y_PATH = os.path.join(PROCESSED_DATA_DIR, 'actions_Y.pt')

INPUT_SIZE = 55  # 3(L_goal) + 3(R_goal) + 3*3(obstacles) + 20(pos) + 20(vel)
OUTPUT_SIZE = 20 # 20 joint velocity commands

# Model Hyperparameters
HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 128
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 1000
VAL_SPLIT_RATIO = 0.15

# --- Define the Neural Network Model (Behavioral Cloning Policy) ---
class BCNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()
        # Optional: Batch Normalization and Dropout
        # self.bn1 = nn.BatchNorm1d(hidden_size_1)
        # self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        # Optional: Batch Normalization and Dropout
        # self.bn2 = nn.BatchNorm1d(hidden_size_2)
        # self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.tanh = nn.Tanh() 

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        # if hasattr(self, 'bn1'): x = self.bn1(x) # Apply BN if defined
        # if hasattr(self, 'dropout1') and self.training: x = self.dropout1(x) # Apply Dropout if defined and in training mode

        x = self.relu2(self.fc2(x))
        # if hasattr(self, 'bn2'): x = self.bn2(x)
        # if hasattr(self, 'dropout2') and self.training: x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.tanh(x) 
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') # For saving the best model

    print(f"\nStarting training for {num_epochs} epochs on {device}...")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            except RuntimeError as e:
                if 'CUDA' in str(e):
                    print(f"CUDA error in batch {i}, skipping: {e}")
                    continue
                else:
                    raise e
                
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                try:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    running_val_loss += loss.item()
                except RuntimeError as e:
                    if 'CUDA' in str(e):
                        print(f"CUDA error in validation, skipping batch: {e}")
                        continue
                    else:
                        raise e
                    
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_bc_policy_model.pth')
            print(f"Best model saved with val_loss: {best_val_loss:.6f}")

    print("Training finished.")
    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()


if __name__ == '__main__':
    # Try to use GPU, but add graceful fallback to CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            # Test CUDA with a small tensor operation
            test_tensor = torch.zeros(1, device='cuda')
            # If successful, use CUDA
            device = torch.device('cuda')
            print(f"Using device: {device}")
            
            # Get GPU memory info
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(device)
                print(f"GPU: {props.name}, Memory: {props.total_memory / 1e9:.2f} GB")
                
            # Reset GPU cache to free memory
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"CUDA initialization failed: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
            use_cuda = False
    else:
        device = torch.device('cpu')
        print(f"CUDA not available, using device: {device}")

    # --- 1. Load Pre-processed Data ---
    print("Loading pre-processed dataset tensors...")
    try:
        # Use weights_only=True to address the FutureWarning and for safety
        features_X = torch.load(FEATURES_X_PATH, map_location='cpu', weights_only=False)
        actions_Y = torch.load(ACTIONS_Y_PATH, map_location='cpu', weights_only=False)
        
        # Convert to float32 to save memory
        features_X = features_X.float()
        actions_Y = actions_Y.float()
        
        print(f"Loaded features_X shape: {features_X.shape}")
        print(f"Loaded actions_Y shape: {actions_Y.shape}")
    except FileNotFoundError:
        print(f"Error: Processed data files not found at {FEATURES_X_PATH} or {ACTIONS_Y_PATH}")
        print("Please run data_processor.py first to generate these files.")
        exit()
    except Exception as e:
        print(f"Error loading processed data: {e}")
        exit()
        
    actual_input_size = features_X.shape[1]
    if actual_input_size != INPUT_SIZE:
        print(f"WARNING: Mismatch in INPUT_SIZE. Expected {INPUT_SIZE}, got {actual_input_size} from data.")
        INPUT_SIZE = actual_input_size
    
    actual_output_size = actions_Y.shape[1]
    if actual_output_size != OUTPUT_SIZE:
        print(f"WARNING: Mismatch in OUTPUT_SIZE. Expected {OUTPUT_SIZE}, got {actual_output_size} from data.")
        OUTPUT_SIZE = actual_output_size
        
    # --- 2. Create PyTorch Dataset and DataLoaders ---
    dataset = TensorDataset(features_X, actions_Y)
    val_size = int(VAL_SPLIT_RATIO * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Reduce num_workers to avoid potential CUDA issues
    num_workers = 0 if use_cuda else 2
    pin_memory = use_cuda
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    print(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 3. Initialize Model, Criterion, Optimizer ---
    model = BCNet(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #, weight_decay=1e-5) # Optional L2
    
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # --- 4. Train the Model ---
    try:
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)

        # --- 5. Save the Final Trained Model (or rely on best model saved during training) ---
        torch.save(model.state_dict(), 'final_bc_policy_model.pth')
        print("Final trained model saved to 'final_bc_policy_model.pth'")

        # --- 6. Plot Losses ---
        plot_losses(train_losses, val_losses)
    except Exception as e:
        print(f"Error during training: {e}")
        # If we have any losses recorded, try to plot them before exiting
        if 'train_losses' in locals() and 'val_losses' in locals() and len(train_losses) > 0:
            print("Plotting partial training results...")
            plot_losses(train_losses, val_losses)