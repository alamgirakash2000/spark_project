import numpy as np
import torch
from typing import Dict, List, Tuple

class DataProcessor:
    """
    Processes real-time robot data into the normalized feature vector needed for
    behavioral cloning inference.
    """
    def __init__(self):
        # Configuration params
        self.K_OBSTACLES = 3  # Number of closest obstacles to consider
        
        # Normalization parameters
        self.JOINT_POS_MIN = -np.pi
        self.JOINT_POS_MAX = np.pi
        self.JOINT_VEL_MAX_ABS = 2.0
        self.RELATIVE_POS_MAX_ABS = 2.0
        self.OBSTACLE_PADDING_VALUE = 10.0
        
    def normalize_joint_positions(self, dof_pos: np.ndarray) -> np.ndarray:
        """Normalizes joint positions to roughly [-1, 1] based on pre-defined min/max."""
        return 2 * (dof_pos - self.JOINT_POS_MIN) / (self.JOINT_POS_MAX - self.JOINT_POS_MIN) - 1

    def normalize_joint_velocities(self, dof_vel: np.ndarray) -> np.ndarray:
        """Normalizes joint velocities to roughly [-1, 1] based on a max absolute value."""
        return np.clip(dof_vel / self.JOINT_VEL_MAX_ABS, -1.0, 1.0)

    def normalize_relative_positions(self, rel_pos: np.ndarray) -> np.ndarray:
        """Normalizes relative goal/obstacle positions to roughly [-1, 1]."""
        return np.clip(rel_pos / self.RELATIVE_POS_MAX_ABS, -1.0, 1.0)
        
    def transform_to_robot_frame(self, world_position, robot_base_frame):
        """
        Transform a position from world coordinates to robot-relative coordinates
        """
        # Create homogeneous coordinates for the world position
        world_position_homog = np.append(world_position, 1.0)
        
        # Calculate the inverse of the robot's transformation matrix
        robot_base_frame_inv = np.linalg.inv(robot_base_frame)
        
        # Transform the position to robot-relative coordinates
        robot_relative_homog = robot_base_frame_inv @ world_position_homog
        
        # Return the 3D position (discard the homogeneous component)
        return robot_relative_homog[:3]
        
    def get_closest_k_obstacles(self, robot_relative_obstacles: List[List[float]]) -> np.ndarray:
        """
        Selects the K closest obstacles based on Euclidean distance.
        Pads with a large distance if fewer than K obstacles are present.
        """
        if not robot_relative_obstacles:  # No obstacles
            return np.full((self.K_OBSTACLES, 3), self.OBSTACLE_PADDING_VALUE, dtype=np.float32)

        obstacles_np = np.array(robot_relative_obstacles, dtype=np.float32)
        distances = np.linalg.norm(obstacles_np, axis=1)
        sorted_indices = np.argsort(distances)

        closest_obstacles = []
        num_obstacles_found = 0
        for i in range(min(len(sorted_indices), self.K_OBSTACLES)):
            closest_obstacles.append(obstacles_np[sorted_indices[i]])
            num_obstacles_found += 1

        # Pad if fewer than k obstacles were found
        while num_obstacles_found < self.K_OBSTACLES:
            closest_obstacles.append(np.full(3, self.OBSTACLE_PADDING_VALUE, dtype=np.float32))
            num_obstacles_found += 1
        
        return np.array(closest_obstacles, dtype=np.float32).flatten()
        
    def process_data(self, agent_feedback: Dict, task_info: Dict) -> torch.Tensor:
        """
        Process the agent_feedback and task_info into a single feature vector
        for behavioral cloning inference.
        
        Args:
            agent_feedback: Dictionary containing robot state information
            task_info: Dictionary containing task-related information like goals and obstacles
            
        Returns:
            torch.Tensor: Normalized feature vector of size [55]
        """
        # Extract robot state
        dof_pos = np.array(agent_feedback["dof_pos_cmd"], dtype=np.float32)
        dof_vel = np.array(agent_feedback["dof_vel_cmd"], dtype=np.float32)
        
        # Get robot base frame
        robot_base_frame = task_info['robot_base_frame']
        
        # Process goal data
        left_goal_world_pos = task_info['goal_teleop']['left'][:3, 3]
        right_goal_world_pos = task_info['goal_teleop']['right'][:3, 3]
        
        # Transform goals to robot-relative coordinates
        left_goal_rel = self.transform_to_robot_frame(left_goal_world_pos, robot_base_frame)
        right_goal_rel = self.transform_to_robot_frame(right_goal_world_pos, robot_base_frame)
        
        # Extract and transform obstacle positions
        obstacle_world_positions = task_info['obstacle']['frames_world'][:, :3, 3]
        
        # Transform obstacle positions to robot-relative coordinates
        obstacle_robot_relative = []
        for obs_pos in obstacle_world_positions:
            rel_pos = self.transform_to_robot_frame(obs_pos, robot_base_frame)
            obstacle_robot_relative.append(rel_pos.tolist())
        
        # Process obstacles to get K closest ones
        processed_obstacles_rel = self.get_closest_k_obstacles(obstacle_robot_relative)
        
        # Normalize everything
        norm_dof_pos = self.normalize_joint_positions(dof_pos[:20])  # Only first 20 DoFs
        norm_dof_vel = self.normalize_joint_velocities(dof_vel[:20])  # Only first 20 DoFs
        norm_left_goal_rel = self.normalize_relative_positions(left_goal_rel)
        norm_right_goal_rel = self.normalize_relative_positions(right_goal_rel)
        norm_obstacles_rel = self.normalize_relative_positions(processed_obstacles_rel.reshape(self.K_OBSTACLES, 3)).flatten()
        
        # Concatenate all features into one vector
        feature_vector_x = np.concatenate([
            norm_left_goal_rel,         # 3 elements
            norm_right_goal_rel,        # 3 elements
            norm_obstacles_rel,         # K_OBSTACLES * 3 elements (9 for K=3)
            norm_dof_pos,               # 20 elements (number of DoFs)
            norm_dof_vel                # 20 elements (number of DoFs)
        ], axis=0)
        
        # Convert to PyTorch tensor - shape should be [55]
        return torch.tensor(feature_vector_x, dtype=torch.float32) 