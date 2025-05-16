from spark_policy.base.base_policy import BasePolicy
from spark_robot import RobotKinematics, RobotConfig
import torch
import torch.nn as nn
import numpy as np
import os
from spark_policy.feedback.data_processor import DataProcessor

# Define the same neural network architecture as used in training
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

class BCInferencePolicy(BasePolicy):
    """
    Behavioral Cloning Inference Policy that uses the trained model to generate control commands
    """
    def __init__(self, robot_cfg: RobotConfig, robot_kinematics: RobotKinematics, 
                 model_path: str = None) -> None:
        super().__init__(robot_cfg, robot_kinematics)
        
        # Initialize data processor
        self.processor = DataProcessor()
        
        # Setup model parameters
        self.input_size = 55
        self.hidden_size_1 = 256
        self.hidden_size_2 = 128
        self.output_size = len(robot_cfg.DoFs)  # Typically 20
        
        # Default path in case none is provided
        if model_path is None:
            # First try to find the model in the current directory
            if os.path.exists("final_bc_policy_model.pth"):
                model_path = "final_bc_policy_model.pth"
            elif os.path.exists("best_bc_policy_model.pth"):
                model_path = "best_bc_policy_model.pth"
            else:
                possible_locations = [
                    os.path.join(os.path.dirname(__file__), "final_bc_policy_model.pth"),
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "il_bc_training", "final_bc_policy_model.pth"),
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "il_bc_training", "best_bc_policy_model.pth")
                ]
                
                for loc in possible_locations:
                    if os.path.exists(loc):
                        model_path = loc
                        break
                        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find a valid model file. Please specify model_path explicitly.")
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for BC inference")
        
        try:
            # Create the model with the same architecture as in training
            self.model = BCNet(self.input_size, self.hidden_size_1, self.hidden_size_2, self.output_size)
            
            # Load the pretrained weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Successfully loaded BC model from {model_path}")
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Since actions were normalized to [-1, 1] in training, we need to scale them back
            # Default to 2.0 (same as used in training normalization)
            self.action_scale = 2.0
            
        except Exception as e:
            print(f"Error initializing BC model: {e}")
            raise
    
    def act(self, agent_feedback: dict, task_info: dict):
        """Generate control commands using the BC model"""
        try:
            # Get the feature vector from the processor or use the one from G1TeleopPIDPolicy if available
            if "feature_vector" in agent_feedback:
                feature_vector = agent_feedback["feature_vector"]
            else:
                feature_vector = self.processor.process_data(agent_feedback, task_info)
            
            # Convert to tensor if it's not already
            if not isinstance(feature_vector, torch.Tensor):
                feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
            
            # Ensure we have a batch dimension
            if feature_vector.dim() == 1:
                feature_vector = feature_vector.unsqueeze(0)
                
            # Move to appropriate device
            feature_vector = feature_vector.to(self.device)
            
            # Run inference
            with torch.no_grad():
                model_output = self.model(feature_vector)
            
            # Convert output to numpy
            model_output_np = model_output.squeeze(0).cpu().numpy()
            
            # Denormalize the output to get actual joint velocities
            # The tanh activation in the model ensures outputs are in [-1, 1]
            dof_control = model_output_np * self.action_scale
            
            # Return the control command and info
            info = {
                "bc_raw_output": model_output_np
            }
            
            return dof_control, info
            
        except Exception as e:
            print(f"Error in BC inference: {e}")
            # Fallback to zero velocities
            zero_control = np.zeros(self.output_size)
            return zero_control, {"error": str(e)} 