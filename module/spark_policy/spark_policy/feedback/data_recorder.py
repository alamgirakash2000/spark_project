# data_recorder.py
import numpy as np
import json
import os
from datetime import datetime

class DataRecorder:
    def __init__(self, save_dir="collected_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.session_counter = 0
        self.current_session_data = []
    
    def transform_to_robot_frame(self, world_position, robot_base_frame):
        """
        Transform a position from world coordinates to robot-relative coordinates
        
        Args:
            world_position: 3D position in world coordinates
            robot_base_frame: 4x4 transformation matrix of the robot
            
        Returns:
            3D position in robot-relative coordinates
        """
        # Create homogeneous coordinates for the world position
        world_position_homog = np.append(world_position, 1.0)
        
        # Calculate the inverse of the robot's transformation matrix
        robot_base_frame_inv = np.linalg.inv(robot_base_frame)
        
        # Transform the position to robot-relative coordinates
        robot_relative_homog = robot_base_frame_inv @ world_position_homog
        
        # Return the 3D position (discard the homogeneous component)
        return robot_relative_homog[:3]
    
    def record_data_point(self, agent_feedback: dict, task_info: dict, action: np.ndarray):
        """
        Record a single data point with all relevant information
        
        Args:
            agent_feedback: Dictionary containing robot feedback
            task_info: Dictionary containing task information including obstacles and goals
            action: The 20-dimensional action array
        """
        # Get robot base frame
        robot_base_frame = task_info['robot_base_frame']
        
        # Extract obstacle positions from frames_world
        obstacle_world_positions = task_info['obstacle']['frames_world'][:, :3, 3]  # Get 3D positions of all obstacles
        
        # Transform obstacle positions to robot-relative coordinates
        obstacle_robot_relative = []
        for obs_pos in obstacle_world_positions:
            rel_pos = self.transform_to_robot_frame(obs_pos, robot_base_frame)
            obstacle_robot_relative.append(rel_pos)
        
        # Extract goal positions from world frame
        left_goal_world_pos = task_info['goal_teleop']['left'][:3, 3]
        right_goal_world_pos = task_info['goal_teleop']['right'][:3, 3]
        
        # Transform goal positions to robot-relative coordinates
        left_goal_robot_relative = self.transform_to_robot_frame(left_goal_world_pos, robot_base_frame)
        right_goal_robot_relative = self.transform_to_robot_frame(right_goal_world_pos, robot_base_frame)
        
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'robot_state': {
                'dof_pos': agent_feedback["dof_pos_cmd"].tolist(),
                'dof_vel': agent_feedback["dof_vel_cmd"].tolist()
            },
            'goals': {
                'world_frame': {
                    'left': left_goal_world_pos.tolist(),
                    'right': right_goal_world_pos.tolist()
                },
                'robot_relative': {
                    'left': left_goal_robot_relative.tolist(),
                    'right': right_goal_robot_relative.tolist()
                }
            },
            'obstacles': {
                'world_frame': obstacle_world_positions.tolist(),
                'robot_relative': [pos.tolist() for pos in obstacle_robot_relative]
            },
            'robot_base_frame': robot_base_frame.tolist(),  # Save the robot's base frame too
            'action': action.tolist()
        }
        
        self.current_session_data.append(data_point)
        
        # Save to file after every 100 data points
        if len(self.current_session_data) >= 30000:
            self.save_current_session()
    
    def save_current_session(self):
        """Save the current session data to a file"""
        if len(self.current_session_data) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"session_{self.session_counter}_{timestamp}.json")
            
            # Save the data
            with open(filename, 'w') as f:
                json.dump(self.current_session_data, f, indent=2)
            
            print(f"Saved {len(self.current_session_data)} data points to {filename}")
            
            # Clear the current session data and increment counter
            self.current_session_data = []
            self.session_counter += 1
    
    def finalize(self):
        """Save any remaining data when the recording session ends"""
        self.save_current_session()