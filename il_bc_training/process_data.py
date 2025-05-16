import json
import os
import numpy as np
import torch
from typing import List, Dict, Tuple

# --- Configuration ---
K_OBSTACLES = 3  # Number of closest obstacles to consider
PROCESSED_DATA_DIR = "processed_tensors" # Directory to save .pt files
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# --- Placeholder Normalization Parameters (IMPORTANT: Calculate these from your actual training dataset!) ---
# Example: Assuming joint positions are in radians, typical range might be -pi to pi
JOINT_POS_MIN = -np.pi
JOINT_POS_MAX = np.pi
# Example: Assuming joint velocities have a typical max
JOINT_VEL_MAX_ABS = 2.0  # Max absolute velocity in rad/s or m/s
# Example: Assuming workspace for relative positions has a certain extent
RELATIVE_POS_MAX_ABS = 2.0  # Max absolute distance for goals/obstacles in meters
OBSTACLE_PADDING_VALUE = 10.0 # A large distance to signify a non-existent or very far obstacle


def normalize_joint_positions(dof_pos: np.ndarray) -> np.ndarray:
    """Normalizes joint positions to roughly [-1, 1] based on pre-defined min/max."""
    return 2 * (dof_pos - JOINT_POS_MIN) / (JOINT_POS_MAX - JOINT_POS_MIN) - 1

def normalize_joint_velocities(dof_vel: np.ndarray) -> np.ndarray:
    """Normalizes joint velocities to roughly [-1, 1] based on a max absolute value."""
    return np.clip(dof_vel / JOINT_VEL_MAX_ABS, -1.0, 1.0)

def normalize_relative_positions(rel_pos: np.ndarray) -> np.ndarray:
    """Normalizes relative goal/obstacle positions to roughly [-1, 1]."""
    return np.clip(rel_pos / RELATIVE_POS_MAX_ABS, -1.0, 1.0)

def normalize_actions(actions: np.ndarray) -> np.ndarray:
    """Normalizes actions (assumed to be joint velocities) like joint velocities."""
    return normalize_joint_velocities(actions)


def get_closest_k_obstacles(robot_relative_obstacles: List[List[float]], k: int) -> np.ndarray:
    """
    Selects the K closest obstacles based on Euclidean distance.
    Pads with a large distance if fewer than K obstacles are present.
    """
    if not robot_relative_obstacles: # No obstacles
        return np.full((k, 3), OBSTACLE_PADDING_VALUE, dtype=np.float32)

    obstacles_np = np.array(robot_relative_obstacles, dtype=np.float32)
    distances = np.linalg.norm(obstacles_np, axis=1)
    sorted_indices = np.argsort(distances)

    closest_obstacles = []
    num_obstacles_found = 0
    for i in range(min(len(sorted_indices), k)):
        closest_obstacles.append(obstacles_np[sorted_indices[i]])
        num_obstacles_found += 1

    # Pad if fewer than k obstacles were found
    while num_obstacles_found < k:
        closest_obstacles.append(np.full(3, OBSTACLE_PADDING_VALUE, dtype=np.float32))
        num_obstacles_found += 1
    
    return np.array(closest_obstacles, dtype=np.float32).flatten()


def process_single_data_point(data_point: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes a single data point from the dataset into feature vector (X) and action (Y).
    """
    dof_pos = np.array(data_point['robot_state']['dof_pos'], dtype=np.float32)
    dof_vel = np.array(data_point['robot_state']['dof_vel'], dtype=np.float32)
    left_goal_rel = np.array(data_point['goals']['robot_relative']['left'], dtype=np.float32)
    right_goal_rel = np.array(data_point['goals']['robot_relative']['right'], dtype=np.float32)
    
    robot_obstacles_raw = data_point['obstacles']['robot_relative']
    # Ensure robot_obstacles_raw is a list of lists before passing to get_closest_k_obstacles
    if not isinstance(robot_obstacles_raw, list) or (robot_obstacles_raw and not isinstance(robot_obstacles_raw[0], list)):
        # If it's empty or not a list of lists (e.g. already flattened), try to reshape or pass empty
        if not robot_obstacles_raw: # Empty list
             processed_obstacles_rel = get_closest_k_obstacles([], K_OBSTACLES)
        else: # Attempt to make it a list of lists if it's a flat list of numbers assumed to be x,y,z,x,y,z...
            try:
                reshaped_obstacles = np.array(robot_obstacles_raw).reshape(-1,3).tolist()
                processed_obstacles_rel = get_closest_k_obstacles(reshaped_obstacles, K_OBSTACLES)
            except: # Fallback if reshape fails
                 print(f"Warning: Obstacle data format unexpected for point. Using empty. Data: {robot_obstacles_raw}")
                 processed_obstacles_rel = get_closest_k_obstacles([], K_OBSTACLES)

    else: # It's already a list of lists or an empty list
        processed_obstacles_rel = get_closest_k_obstacles(robot_obstacles_raw, K_OBSTACLES)

    norm_dof_pos = normalize_joint_positions(dof_pos)
    norm_dof_vel = normalize_joint_velocities(dof_vel)
    norm_left_goal_rel = normalize_relative_positions(left_goal_rel)
    norm_right_goal_rel = normalize_relative_positions(right_goal_rel)
    # Ensure processed_obstacles_rel is shaped correctly before normalizing
    norm_obstacles_rel = normalize_relative_positions(processed_obstacles_rel.reshape(K_OBSTACLES, 3)).flatten()

    feature_vector_x = np.concatenate([
        norm_left_goal_rel, norm_right_goal_rel, norm_obstacles_rel,
        norm_dof_pos, norm_dof_vel
    ], axis=0)

    action_y = np.array(data_point['action'], dtype=np.float32)
    norm_action_y = normalize_actions(action_y)

    return feature_vector_x, norm_action_y


def load_and_process_dataset(dataset_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    all_features_x = []
    all_actions_y = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dataset_dir, filename)
            print(f"Processing file: {filepath}")
            with open(filepath, 'r') as f:
                session_data = json.load(f)
                for data_point in session_data:
                    try:
                        features_x, action_y = process_single_data_point(data_point)
                        all_features_x.append(features_x)
                        all_actions_y.append(action_y)
                    except Exception as e:
                        print(f"Error processing data point: {e} in file {filepath}. Data: {data_point}. Skipping.")
    
    if not all_features_x:
        raise ValueError("No data processed. Check dataset directory and file contents.")

    tensor_x = torch.tensor(np.array(all_features_x), dtype=torch.float32)
    tensor_y = torch.tensor(np.array(all_actions_y), dtype=torch.float32)

    print(f"Dataset loaded and processed. X shape: {tensor_x.shape}, Y shape: {tensor_y.shape}")
    return tensor_x, tensor_y


if __name__ == '__main__':
    dataset_directory = "collected_data"
    
    print("Attempting to load and process the dataset...")
    try:
        features_X, actions_Y = load_and_process_dataset(dataset_directory)
        
        print("\n--- Example of first processed data point ---")
        if features_X.nelement() > 0 and actions_Y.nelement() > 0: # Check if tensors are not empty
            print("Features (X_sample):")
            print(features_X[0])
            print(f"Shape of X_sample: {features_X[0].shape}")
            
            print("\nAction (Y_sample):")
            print(actions_Y[0])
            print(f"Shape of Y_sample: {actions_Y[0].shape}")

            # --- Save the processed tensors ---
            features_save_path = os.path.join(PROCESSED_DATA_DIR, 'features_X.pt')
            actions_save_path = os.path.join(PROCESSED_DATA_DIR, 'actions_Y.pt')
            
            torch.save(features_X, features_save_path)
            torch.save(actions_Y, actions_save_path)
            print(f"\nProcessed features saved to: {features_save_path}")
            print(f"Processed actions saved to: {actions_save_path}")
        else:
            print("Processed tensors are empty. Nothing to save.")


    except ValueError as ve:
        print(f"ValueError: {ve}")
    except FileNotFoundError:
        print(f"Error: Dataset directory '{dataset_directory}' not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred during processing or saving: {e}")