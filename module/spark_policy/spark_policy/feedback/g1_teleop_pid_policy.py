from spark_policy.base.base_policy import BasePolicy
from spark_agent import BaseAgent
from spark_robot import RobotKinematics, RobotConfig
import numpy as np
from spark_policy.feedback.data_recorder import DataRecorder
from spark_policy.feedback.data_processor import DataProcessor
from spark_policy.feedback.bc_model_loader import load_bc_model, get_bc_action

class G1TeleopPIDPolicy(BasePolicy):
    def __init__(self, robot_cfg: RobotConfig, robot_kinematics: RobotKinematics) -> None:
        super().__init__(robot_cfg, robot_kinematics)
        self.pos_K_p = np.ones(len(self.robot_cfg.DoFs))
        self.pos_K_d = np.zeros(len(self.robot_cfg.DoFs))
        self.recorder = DataRecorder()  # Initialize the recorder
        self.processor = DataProcessor()  # Initialize the data processor
        
        # Load BC model
        self.bc_model = load_bc_model()
    
    def tracking_pos_with_vel(self, 
                              desired_dof_pos,
                              dof_pos,
                              dof_vel):
        
        nominal_dof_vel = self.pos_K_p * (desired_dof_pos - dof_pos) - self.pos_K_d * dof_vel
        
        return nominal_dof_vel

    def act(self, agent_feedback: dict, task_info: dict):
        
        dof_pos_cmd = agent_feedback["dof_pos_cmd"]
        dof_vel_cmd = agent_feedback["dof_vel_cmd"]
        
        goal_teleop = task_info["goal_teleop"]
        
        # Get desired positions with inverse kinematics for PID control
        desired_dof_pos, _ = self.robot_kinematics.inverse_kinematics([goal_teleop["left"], goal_teleop["right"]])
        dof_control = self.tracking_pos_with_vel(desired_dof_pos, dof_pos_cmd, dof_vel_cmd)
        
        # Process the current state into a feature vector for BC
        feature_vector = self.processor.process_data(agent_feedback, task_info)
        
        # Get BC model output
        bc_control = get_bc_action(self.bc_model, feature_vector)
        info = {}
        
        #self.recorder.record_data_point(agent_feedback, task_info, dof_control)


        ###### EXAMPLE OPERATION DETAILS ######################
        '''
        To run with the PID policy, use: dof_control
        To run with the BC policy, use:  bc_control
        '''

        return bc_control, info