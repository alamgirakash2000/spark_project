import os
import torch
import torch.nn as nn

# Define the same model architecture as in training
class BCNet(nn.Module):
    def __init__(self, input_size=55, hidden_size_1=256, hidden_size_2=128, output_size=20):
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)
        return x

def load_bc_model(model_path=None):
    """
    Load the trained behavioral cloning model.
    
    Args:
        model_path: Path to the model file. If None, will search in common locations.
        
    Returns:
        Loaded PyTorch model ready for inference
    """
    # Check for model in common locations if not specified
    if model_path is None:
        possible_paths = [
            "final_bc_policy_model.pth",
            "best_bc_policy_model.pth",
            os.path.join(os.path.dirname(__file__), "final_bc_policy_model.pth"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "il_bc_training", "final_bc_policy_model.pth"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "il_bc_training", "best_bc_policy_model.pth")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError("Could not find BC model file. Please specify model_path explicitly.")
    
    # Create model with the architecture from training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCNet()
    
    # Load weights and set to eval mode
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Successfully loaded BC model from {model_path}")
    return model

def get_bc_action(model, feature_vector, scaling_factor=2.0):
    """
    Run inference with the BC model to get control action.
    
    Args:
        model: Loaded BC model
        feature_vector: Processed feature vector (55-dim tensor)
        scaling_factor: Value to scale the output by (default: 2.0 to match normalization)
        
    Returns:
        numpy array with control commands
    """
    # Ensure input is a tensor
    if not isinstance(feature_vector, torch.Tensor):
        feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
    
    # Add batch dimension if needed
    if feature_vector.dim() == 1:
        feature_vector = feature_vector.unsqueeze(0)
    
    # Move to same device as model
    feature_vector = feature_vector.to(next(model.parameters()).device)
    
    # Run inference
    with torch.no_grad():
        output = model(feature_vector)
    
    # Convert to numpy and scale
    action = output.squeeze(0).cpu().numpy() * scaling_factor
    
    return action 