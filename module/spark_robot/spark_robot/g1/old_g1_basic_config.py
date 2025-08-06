from enum import IntEnum
from spark_robot.base.base_robot_config import RobotConfig
from spark_utils import Geometry, VizColor
import numpy as np

kNotUsedJoint = 29

class G1BasicConfig(RobotConfig):
    
    # ---------------------------------------------------------------------------- #
    #                                      Kinematics                              #
    # ---------------------------------------------------------------------------- #
    
    mjcf_path = 'g1/g1_29dof_rev_1_0.xml'

    
    joint_to_lock = [  
                        # Keep these locked (less critical):
                        "left_hip_yaw_joint",      # Keep locked
                        "left_ankle_roll_joint",   # Keep locked
                        "right_hip_yaw_joint",     # Keep locked
                        "right_ankle_roll_joint",  # Keep locked
                    ]
    
    # ---------------------------------------------------------------------------- #
    #                                   hardware                                   #
    # ---------------------------------------------------------------------------- #
    
    class RealMotors(IntEnum):

        # ADD leg motor indices
        LeftHipPitch    = 0
        LeftHipRoll     = 1
        LeftKnee        = 2
        LeftAnklePitch  = 3
        RightHipPitch   = 4
        RightHipRoll    = 5
        RightKnee       = 6
        RightAnklePitch = 7

        # Waist (UNCHANGED)
        WaistYaw   = 12
        WaistRoll  = 13
        WaistPitch = 14

        # Left arm (UNCHANGED)
        LeftShoulderPitch = 15
        LeftShoulderRoll  = 16
        LeftShoulderYaw   = 17
        LeftElbow         = 18
        LeftWristRoll     = 19
        LeftWristPitch    = 20
        LeftWristYaw      = 21
        
        # Right arm (UNCHANGED)
        RightShoulderPitch = 22
        RightShoulderRoll  = 23
        RightShoulderYaw   = 24
        RightElbow         = 25
        RightWristRoll     = 26
        RightWristPitch    = 27
        RightWristYaw      = 28
    
    # Based on https://support.unitree.com/home/en/G1_developer/about_G1
    RealMotorPosLimit = {
        # ADD leg joint limits
        RealMotors.LeftHipPitch: (-1.57, 1.57),      # ±90 degrees
        RealMotors.LeftHipRoll: (-0.52, 0.52),       # ±30 degrees  
        RealMotors.LeftKnee: (0.0, 2.35),            # 0-135 degrees
        RealMotors.LeftAnklePitch: (-0.87, 0.87),    # ±50 degrees
        RealMotors.RightHipPitch: (-1.57, 1.57),
        RealMotors.RightHipRoll: (-0.52, 0.52),
        RealMotors.RightKnee: (0.0, 2.35),
        RealMotors.RightAnklePitch: (-0.87, 0.87),
        
        # Upper body limits (UNCHANGED)
        RealMotors.WaistYaw: (-2.618, 2.618),
        RealMotors.WaistRoll: (-0.52, 0.52),
        RealMotors.WaistPitch: (-0.52, 0.52),
        RealMotors.LeftShoulderPitch: (-3.0892, 2.6704),
        RealMotors.LeftShoulderRoll: (-1.5882, 2.2515),
        RealMotors.LeftShoulderYaw: (-2.618, 2.618),
        RealMotors.LeftElbow: (-1.0472, 2.0944),
        RealMotors.LeftWristRoll: (-1.972222054, 1.972222054),
        RealMotors.LeftWristPitch: (-1.614429558, 1.614429558),
        RealMotors.LeftWristYaw: (-1.614429558, 1.614429558),
        RealMotors.RightShoulderPitch: (-3.0892, 2.6704),
        RealMotors.RightShoulderRoll: (-2.2515, 1.5882),
        RealMotors.RightShoulderYaw: (-2.618, 2.618),
        RealMotors.RightElbow: (-1.0472, 2.0944),
        RealMotors.RightWristRoll: (-1.972222054, 1.972222054),
        RealMotors.RightWristPitch: (-1.614429558, 1.614429558),
        RealMotors.RightWristYaw: (-1.614429558, 1.614429558)
    }
    
    # ---------------------------------------------------------------------------- #
    #                                      DoF                                     #
    # ---------------------------------------------------------------------------- #

    class DoFs (IntEnum):

        # KEEP ORIGINAL STRUCTURE EXACTLY - NO CHANGES TO PRESERVE BEHAVIOR

        # Waist (UNCHANGED)
        WaistYaw   = 0
        WaistRoll  = 1
        WaistPitch = 2

        # Left arm (UNCHANGED)
        LeftShoulderPitch = 3
        LeftShoulderRoll  = 4
        LeftShoulderYaw   = 5
        LeftElbow         = 6
        LeftWristRoll     = 7
        LeftWristPitch    = 8
        LeftWristYaw      = 9
        
        # Right arm (UNCHANGED)
        RightShoulderPitch = 10
        RightShoulderRoll  = 11
        RightShoulderYaw   = 12
        RightElbow         = 13
        RightWristRoll     = 14
        RightWristPitch    = 15
        RightWristYaw      = 16
        
        # Base movement (UNCHANGED - KEEP ORIGINAL INDICES)
        LinearX = 17
        LinearY = 18
        RotYaw  = 19
        
        # ADD leg joints at the end (NEW - separate indices)
        LeftHipPitch    = 20
        LeftHipRoll     = 21
        LeftKnee        = 22
        LeftAnklePitch  = 23
        RightHipPitch   = 24
        RightHipRoll    = 25
        RightKnee       = 26
        RightAnklePitch = 27

    # ---------------------------------------------------------------------------- #
    #                                   Dynamics                                   #
    # ---------------------------------------------------------------------------- #

    class Control (IntEnum):

        # KEEP ORIGINAL STRUCTURE EXACTLY - NO CHANGES TO PRESERVE BEHAVIOR

        # Waist (UNCHANGED)
        vWaistYaw   = 0
        vWaistRoll  = 1
        vWaistPitch = 2

        # Left arm (UNCHANGED)
        vLeftShoulderPitch = 3
        vLeftShoulderRoll  = 4
        vLeftShoulderYaw   = 5
        vLeftElbow         = 6
        vLeftWristRoll     = 7
        vLeftWristPitch    = 8
        vLeftWristYaw      = 9
        
        # Right arm (UNCHANGED)
        vRightShoulderPitch = 10
        vRightShoulderRoll  = 11
        vRightShoulderYaw   = 12
        vRightElbow         = 13
        vRightWristRoll     = 14
        vRightWristPitch    = 15
        vRightWristYaw      = 16
        
        # Base movement (UNCHANGED - KEEP ORIGINAL INDICES)
        vLinearX = 17
        vLinearY = 18
        vRotYaw  = 19
        
        # ADD leg controls at the end (NEW - separate indices)
        vLeftHipPitch    = 20
        vLeftHipRoll     = 21
        vLeftKnee        = 22
        vLeftAnklePitch  = 23
        vRightHipPitch   = 24
        vRightHipRoll    = 25
        vRightKnee       = 26
        vRightAnklePitch = 27
    
    ControlLimit = {
        # UNCHANGED - Keep exactly the same as original
        Control.vWaistYaw          : 0.0,
        Control.vWaistRoll         : 1.0,
        Control.vWaistPitch        : 0.0,
        Control.vLeftShoulderPitch : 5.0,
        Control.vLeftShoulderRoll  : 5.0,
        Control.vLeftShoulderYaw   : 5.0,
        Control.vLeftElbow         : 5.0,
        Control.vLeftWristRoll     : 5.0,
        Control.vLeftWristPitch    : 5.0,
        Control.vLeftWristYaw      : 5.0,
        Control.vRightShoulderPitch: 5.0,
        Control.vRightShoulderRoll : 5.0,
        Control.vRightShoulderYaw  : 5.0,
        Control.vRightElbow        : 5.0,
        Control.vRightWristRoll    : 5.0,
        Control.vRightWristPitch   : 5.0,
        Control.vRightWristYaw     : 5.0,
        Control.vLinearX           : 1.0,
        Control.vLinearY           : 1.0,
        Control.vRotYaw            : 1.0,
        
        # ADD leg control limits (NEW)
        Control.vLeftHipPitch: 3.0,
        Control.vLeftHipRoll: 2.0,
        Control.vLeftKnee: 3.0,
        Control.vLeftAnklePitch: 2.0,
        Control.vRightHipPitch: 3.0,
        Control.vRightHipRoll: 2.0,
        Control.vRightKnee: 3.0,
        Control.vRightAnklePitch: 2.0,
    }

    # UNCHANGED - Keep exactly the same as original
    NormalControl = [
        Control.vWaistYaw,
        Control.vWaistRoll,
        Control.vWaistPitch,
        Control.vLinearX,
        Control.vLinearY,
        Control.vRotYaw,
    ]
    
    WeakControl = [
        Control.vLeftShoulderPitch,
        Control.vLeftShoulderRoll,
        Control.vLeftShoulderYaw,
        Control.vLeftElbow,
        Control.vRightShoulderPitch,
        Control.vRightShoulderRoll,
        Control.vRightShoulderYaw,
        Control.vRightElbow,
    ]
    
    DelicateControl = [
        Control.vLeftWristRoll,
        Control.vLeftWristPitch,
        Control.vLeftWristYaw,
        Control.vRightWristRoll,
        Control.vRightWristPitch,
        Control.vRightWristYaw,
    ]

    '''
        x_dot = f(x) + g(x) * control

        For velocity control, f = 0, g = I
    '''

    @property
    def num_state(self):
        return len(self.DoFs)

    def compose_state_from_dof(self, dof_pos):
        '''
            dof_pos: [num_dof,]
        '''
        state = dof_pos.reshape(-1)
        return state

    def decompose_state_to_dof(self, state):
        '''
            state: [num_state,]
            return: [num_dof,]
        '''
        dof_pos = state.reshape(-1)
        return dof_pos

    def dynamics_f(self, state):
        '''
            state: [num_state, 1]
            return: [num_state, 1]
        '''
        
        return np.zeros((self.num_state, 1))

    def dynamics_g(self, state):
        '''
            state: [num_state, 1]
            return: [num_state, num_control]
        '''

        return np.eye(self.num_state)

    # ---------------------------------------------------------------------------- #
    #                                    MuJoCo                                    #
    # ---------------------------------------------------------------------------- #
    class MujocoDoFs(IntEnum):

        # From the MJCF structure - need to map correctly to new g1_29dof_rev_1_0.xml
        # Based on actual XML structure
        
        # Base movement (free joint equivalent - first 3 DoFs)
        LinearX = 0  # pelvis_x_joint
        LinearY = 1  # pelvis_y_joint  
        RotYaw  = 2  # pelvis_yaw_joint
        
        # Leg joints (unlocked ones in order they appear in XML)
        LeftHipPitch    = 3
        LeftHipRoll     = 4
        LeftKnee        = 5
        LeftAnklePitch  = 6
        RightHipPitch   = 7
        RightHipRoll    = 8
        RightKnee       = 9
        RightAnklePitch = 10
        
        # Upper body joints (continue from where legs end)
        WaistYaw   = 15
        WaistRoll  = 16
        WaistPitch = 17
        
        LeftShoulderPitch = 18
        LeftShoulderRoll  = 19
        LeftShoulderYaw   = 20
        LeftElbow         = 21
        LeftWristRoll     = 22
        LeftWristPitch    = 23
        LeftWristYaw      = 24
        
        RightShoulderPitch = 25
        RightShoulderRoll  = 26
        RightShoulderYaw   = 27
        RightElbow         = 28
        RightWristRoll     = 29
        RightWristPitch    = 30
        RightWristYaw      = 31

    class MujocoMotors(IntEnum):

        # Motors in order from MJCF actuator section
        LeftHipPitch    = 0
        LeftHipRoll     = 1
        LeftHipYaw      = 2  # Even though locked, it's in actuator list
        LeftKnee        = 3
        LeftAnklePitch  = 4
        LeftAnkleRoll   = 5  # Even though locked, it's in actuator list
        RightHipPitch   = 6
        RightHipRoll    = 7
        RightHipYaw     = 8  # Even though locked, it's in actuator list
        RightKnee       = 9
        RightAnklePitch = 10
        RightAnkleRoll  = 11 # Even though locked, it's in actuator list
        
        WaistYaw   = 12
        WaistRoll  = 13
        WaistPitch = 14
        
        LeftShoulderPitch = 15
        LeftShoulderRoll  = 16
        LeftShoulderYaw   = 17
        LeftElbow         = 18
        LeftWristRoll     = 19
        LeftWristPitch    = 20
        LeftWristYaw      = 21
        
        RightShoulderPitch = 22
        RightShoulderRoll  = 23
        RightShoulderYaw   = 24
        RightElbow         = 25
        RightWristRoll     = 26
        RightWristPitch    = 27
        RightWristYaw      = 28

    # ---------------------------------------------------------------------------- #
    #                                   Mappings                                   #
    # ---------------------------------------------------------------------------- #

    # Mapping from Mujoco DoFs to DoFs
    MujocoDoF_to_DoF = {
        # UNCHANGED upper body mappings
        MujocoDoFs.WaistYaw   : DoFs.WaistYaw,
        MujocoDoFs.WaistRoll  : DoFs.WaistRoll,
        MujocoDoFs.WaistPitch : DoFs.WaistPitch,

        MujocoDoFs.LeftShoulderPitch : DoFs.LeftShoulderPitch,
        MujocoDoFs.LeftShoulderRoll  : DoFs.LeftShoulderRoll,
        MujocoDoFs.LeftShoulderYaw   : DoFs.LeftShoulderYaw,
        MujocoDoFs.LeftElbow         : DoFs.LeftElbow,
        MujocoDoFs.LeftWristRoll     : DoFs.LeftWristRoll,
        MujocoDoFs.LeftWristPitch    : DoFs.LeftWristPitch,
        MujocoDoFs.LeftWristYaw      : DoFs.LeftWristYaw,

        MujocoDoFs.RightShoulderPitch : DoFs.RightShoulderPitch,
        MujocoDoFs.RightShoulderRoll  : DoFs.RightShoulderRoll,
        MujocoDoFs.RightShoulderYaw   : DoFs.RightShoulderYaw,
        MujocoDoFs.RightElbow         : DoFs.RightElbow,
        MujocoDoFs.RightWristRoll     : DoFs.RightWristRoll,
        MujocoDoFs.RightWristPitch    : DoFs.RightWristPitch,
        MujocoDoFs.RightWristYaw      : DoFs.RightWristYaw,

        MujocoDoFs.LinearX: DoFs.LinearX,
        MujocoDoFs.LinearY: DoFs.LinearY,
        MujocoDoFs.RotYaw : DoFs.RotYaw,
        
        # ADD leg mappings (NEW)
        MujocoDoFs.LeftHipPitch: DoFs.LeftHipPitch,
        MujocoDoFs.LeftHipRoll: DoFs.LeftHipRoll,
        MujocoDoFs.LeftKnee: DoFs.LeftKnee,
        MujocoDoFs.LeftAnklePitch: DoFs.LeftAnklePitch,
        MujocoDoFs.RightHipPitch: DoFs.RightHipPitch,
        MujocoDoFs.RightHipRoll: DoFs.RightHipRoll,
        MujocoDoFs.RightKnee: DoFs.RightKnee,
        MujocoDoFs.RightAnklePitch: DoFs.RightAnklePitch,
    }

    # Mapping from DoFs to Mujoco DoFs
    DoF_to_MujocoDoF = {
        # UNCHANGED upper body mappings
        DoFs.WaistYaw  : MujocoDoFs.WaistYaw,
        DoFs.WaistRoll : MujocoDoFs.WaistRoll,
        DoFs.WaistPitch: MujocoDoFs.WaistPitch,

        DoFs.LeftShoulderPitch: MujocoDoFs.LeftShoulderPitch,
        DoFs.LeftShoulderRoll : MujocoDoFs.LeftShoulderRoll,
        DoFs.LeftShoulderYaw  : MujocoDoFs.LeftShoulderYaw,
        DoFs.LeftElbow        : MujocoDoFs.LeftElbow,
        DoFs.LeftWristRoll    : MujocoDoFs.LeftWristRoll,
        DoFs.LeftWristPitch   : MujocoDoFs.LeftWristPitch,
        DoFs.LeftWristYaw     : MujocoDoFs.LeftWristYaw,

        DoFs.RightShoulderPitch: MujocoDoFs.RightShoulderPitch,
        DoFs.RightShoulderRoll : MujocoDoFs.RightShoulderRoll,
        DoFs.RightShoulderYaw  : MujocoDoFs.RightShoulderYaw,
        DoFs.RightElbow        : MujocoDoFs.RightElbow,
        DoFs.RightWristRoll    : MujocoDoFs.RightWristRoll,
        DoFs.RightWristPitch   : MujocoDoFs.RightWristPitch,
        DoFs.RightWristYaw     : MujocoDoFs.RightWristYaw,

        DoFs.LinearX: MujocoDoFs.LinearX,
        DoFs.LinearY: MujocoDoFs.LinearY,
        DoFs.RotYaw : MujocoDoFs.RotYaw,
        
        # ADD leg mappings (NEW)
        DoFs.LeftHipPitch: MujocoDoFs.LeftHipPitch,
        DoFs.LeftHipRoll: MujocoDoFs.LeftHipRoll,
        DoFs.LeftKnee: MujocoDoFs.LeftKnee,
        DoFs.LeftAnklePitch: MujocoDoFs.LeftAnklePitch,
        DoFs.RightHipPitch: MujocoDoFs.RightHipPitch,
        DoFs.RightHipRoll: MujocoDoFs.RightHipRoll,
        DoFs.RightKnee: MujocoDoFs.RightKnee,
        DoFs.RightAnklePitch: MujocoDoFs.RightAnklePitch,
    }

    # Mapping from Mujoco Motors to Control
    MujocoMotor_to_Control = {
        # UNCHANGED upper body mappings
        MujocoMotors.WaistYaw   : Control.vWaistYaw,
        MujocoMotors.WaistRoll  : Control.vWaistRoll,
        MujocoMotors.WaistPitch : Control.vWaistPitch,

        MujocoMotors.LeftShoulderPitch : Control.vLeftShoulderPitch,
        MujocoMotors.LeftShoulderRoll  : Control.vLeftShoulderRoll,
        MujocoMotors.LeftShoulderYaw   : Control.vLeftShoulderYaw,
        MujocoMotors.LeftElbow         : Control.vLeftElbow,
        MujocoMotors.LeftWristRoll     : Control.vLeftWristRoll,
        MujocoMotors.LeftWristPitch    : Control.vLeftWristPitch,
        MujocoMotors.LeftWristYaw      : Control.vLeftWristYaw,

        MujocoMotors.RightShoulderPitch : Control.vRightShoulderPitch,
        MujocoMotors.RightShoulderRoll  : Control.vRightShoulderRoll,
        MujocoMotors.RightShoulderYaw   : Control.vRightShoulderYaw,
        MujocoMotors.RightElbow         : Control.vRightElbow,
        MujocoMotors.RightWristRoll     : Control.vRightWristRoll,
        MujocoMotors.RightWristPitch    : Control.vRightWristPitch,
        MujocoMotors.RightWristYaw      : Control.vRightWristYaw,

        # ADD leg motor to control mappings (NEW - only for unlocked joints)
        MujocoMotors.LeftHipPitch: Control.vLeftHipPitch,
        MujocoMotors.LeftHipRoll: Control.vLeftHipRoll,
        MujocoMotors.LeftKnee: Control.vLeftKnee,
        MujocoMotors.LeftAnklePitch: Control.vLeftAnklePitch,
        MujocoMotors.RightHipPitch: Control.vRightHipPitch,
        MujocoMotors.RightHipRoll: Control.vRightHipRoll,
        MujocoMotors.RightKnee: Control.vRightKnee,
        MujocoMotors.RightAnklePitch: Control.vRightAnklePitch,
        
        # Note: Locked joints (hip_yaw, ankle_roll) don't get control mappings
    }

    # Mapping from real motors to Control
    RealMotor_to_Control = {
        # UNCHANGED upper body mappings
        RealMotors.WaistYaw  : Control.vWaistYaw,
        RealMotors.WaistRoll : Control.vWaistRoll,
        RealMotors.WaistPitch: Control.vWaistPitch,
        
        RealMotors.LeftShoulderPitch: Control.vLeftShoulderPitch,
        RealMotors.LeftShoulderRoll : Control.vLeftShoulderRoll,
        RealMotors.LeftShoulderYaw  : Control.vLeftShoulderYaw,
        RealMotors.LeftElbow        : Control.vLeftElbow,
        RealMotors.LeftWristRoll    : Control.vLeftWristRoll,
        RealMotors.LeftWristPitch   : Control.vLeftWristPitch,
        RealMotors.LeftWristYaw     : Control.vLeftWristYaw,
        
        RealMotors.RightShoulderPitch: Control.vRightShoulderPitch,
        RealMotors.RightShoulderRoll : Control.vRightShoulderRoll,
        RealMotors.RightShoulderYaw  : Control.vRightShoulderYaw,
        RealMotors.RightElbow        : Control.vRightElbow,
        RealMotors.RightWristRoll    : Control.vRightWristRoll,
        RealMotors.RightWristPitch   : Control.vRightWristPitch,
        RealMotors.RightWristYaw     : Control.vRightWristYaw,
        
        # ADD leg real motor mappings (NEW)
        RealMotors.LeftHipPitch: Control.vLeftHipPitch,
        RealMotors.LeftHipRoll: Control.vLeftHipRoll,
        RealMotors.LeftKnee: Control.vLeftKnee,
        RealMotors.LeftAnklePitch: Control.vLeftAnklePitch,
        RealMotors.RightHipPitch: Control.vRightHipPitch,
        RealMotors.RightHipRoll: Control.vRightHipRoll,
        RealMotors.RightKnee: Control.vRightKnee,
        RealMotors.RightAnklePitch: Control.vRightAnklePitch,
        
        # Control.vLinearX : None,
        # Control.vLinearY : None,
        # Control.vRotYaw  : None,
    }

    # ---------------------------------------------------------------------------- #
    #                                   Cartesian                                  #
    # ---------------------------------------------------------------------------- #
    
    class Frames(IntEnum):
        
        # KEEP EXACTLY THE SAME AS ORIGINAL - NO CHANGES TO PRESERVE COLLISION SPHERES
        
        waist_yaw_joint = 0
        waist_roll_joint = 1
        waist_pitch_joint = 2
        
        left_shoulder_pitch_joint = 3
        left_shoulder_roll_joint = 4
        left_shoulder_yaw_joint = 5
        left_elbow_joint = 6
        left_wrist_roll_joint = 7
        left_wrist_pitch_joint = 8
        left_wrist_yaw_joint = 9
        
        right_shoulder_pitch_joint = 10
        right_shoulder_roll_joint = 11
        right_shoulder_yaw_joint = 12
        right_elbow_joint = 13
        right_wrist_roll_joint = 14
        right_wrist_pitch_joint = 15
        right_wrist_yaw_joint = 16
        
        L_ee = 17
        R_ee = 18
        
        torso_link_1 = 19
        torso_link_2 = 20
        torso_link_3 = 21
        
        pelvis_link_1 = 22
        pelvis_link_2 = 23
        pelvis_link_3 = 24
    
    # ---------------------------------------------------------------------------- #
    #                                   Collision                                  #
    # ---------------------------------------------------------------------------- #
    
    CollisionVol = {
        # KEEP EXACTLY THE SAME AS ORIGINAL - NO CHANGES
        
        Frames.waist_yaw_joint  : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.waist_roll_joint : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.waist_pitch_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        
        Frames.left_shoulder_pitch_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_shoulder_roll_joint : Geometry(type='sphere', radius=0.06, color=VizColor.collision_volume),
        Frames.left_shoulder_yaw_joint  : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_elbow_joint         : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_wrist_roll_joint    : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_wrist_pitch_joint   : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.left_wrist_yaw_joint     : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        
        Frames.right_shoulder_pitch_joint: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_shoulder_roll_joint : Geometry(type='sphere', radius=0.06, color=VizColor.collision_volume),
        Frames.right_shoulder_yaw_joint  : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_elbow_joint         : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_wrist_roll_joint    : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_wrist_pitch_joint   : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.right_wrist_yaw_joint     : Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        
        Frames.L_ee: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.R_ee: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),

        Frames.torso_link_1: Geometry(type='sphere', radius=0.10, color=VizColor.collision_volume),
        Frames.torso_link_2: Geometry(type='sphere', radius=0.10, color=VizColor.collision_volume),
        Frames.torso_link_3: Geometry(type='sphere', radius=0.08, color=VizColor.collision_volume),
        
        Frames.pelvis_link_1: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.pelvis_link_2: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume),
        Frames.pelvis_link_3: Geometry(type='sphere', radius=0.05, color=VizColor.collision_volume)
    }

    # Pairs of adjacent joints to be ignored in collision checking
    AdjacentCollisionVolPairs = [
        # KEEP EXACTLY THE SAME AS ORIGINAL - NO CHANGES
        
        [Frames.waist_yaw_joint, Frames.waist_roll_joint],
        [Frames.waist_yaw_joint, Frames.waist_pitch_joint],
        [Frames.waist_yaw_joint, Frames.torso_link_1],
        [Frames.waist_roll_joint, Frames.waist_pitch_joint],
        [Frames.waist_roll_joint, Frames.torso_link_1],
        [Frames.waist_pitch_joint, Frames.torso_link_1],
        [Frames.torso_link_1, Frames.torso_link_2],
        [Frames.torso_link_1, Frames.torso_link_3],
        [Frames.torso_link_2, Frames.torso_link_3],

        [Frames.left_shoulder_pitch_joint, Frames.torso_link_1],
        [Frames.left_shoulder_pitch_joint, Frames.torso_link_2],
        [Frames.left_shoulder_roll_joint, Frames.torso_link_1],
        [Frames.left_shoulder_roll_joint, Frames.torso_link_2],
        [Frames.left_shoulder_pitch_joint, Frames.left_shoulder_roll_joint],
        [Frames.left_shoulder_pitch_joint, Frames.left_shoulder_yaw_joint],
        [Frames.left_shoulder_roll_joint, Frames.left_shoulder_yaw_joint],
        [Frames.left_shoulder_yaw_joint, Frames.left_elbow_joint],
        [Frames.left_elbow_joint, Frames.left_wrist_roll_joint],
        [Frames.left_wrist_roll_joint, Frames.left_wrist_pitch_joint],
        [Frames.left_wrist_roll_joint, Frames.left_wrist_yaw_joint],
        [Frames.left_wrist_roll_joint, Frames.L_ee],
        [Frames.left_wrist_pitch_joint, Frames.left_wrist_yaw_joint],
        [Frames.left_wrist_pitch_joint, Frames.L_ee],
        [Frames.left_wrist_yaw_joint, Frames.L_ee],

        [Frames.right_shoulder_pitch_joint, Frames.torso_link_1],
        [Frames.right_shoulder_pitch_joint, Frames.torso_link_2],
        [Frames.right_shoulder_roll_joint, Frames.torso_link_1],
        [Frames.right_shoulder_roll_joint, Frames.torso_link_2],
        [Frames.right_shoulder_pitch_joint, Frames.right_shoulder_roll_joint],
        [Frames.right_shoulder_pitch_joint, Frames.right_shoulder_yaw_joint],
        [Frames.right_shoulder_roll_joint, Frames.right_shoulder_yaw_joint],
        [Frames.right_shoulder_yaw_joint, Frames.right_elbow_joint],
        [Frames.right_elbow_joint, Frames.right_wrist_roll_joint],
        [Frames.right_wrist_roll_joint, Frames.right_wrist_pitch_joint],
        [Frames.right_wrist_roll_joint, Frames.right_wrist_yaw_joint],
        [Frames.right_wrist_roll_joint, Frames.R_ee],
        [Frames.right_wrist_pitch_joint, Frames.right_wrist_yaw_joint],
        [Frames.right_wrist_pitch_joint, Frames.R_ee],
        [Frames.right_wrist_yaw_joint, Frames.R_ee],
    ]

    SelfCollisionVolIgnored = [
        # KEEP EXACTLY THE SAME AS ORIGINAL - NO CHANGES
        
        Frames.waist_yaw_joint,
        Frames.waist_roll_joint,
        Frames.waist_pitch_joint,
        
        Frames.left_shoulder_pitch_joint,
        Frames.left_shoulder_yaw_joint,
        Frames.left_wrist_roll_joint,
        Frames.left_wrist_pitch_joint,
        Frames.left_wrist_yaw_joint,
        
        Frames.right_shoulder_pitch_joint,
        Frames.right_shoulder_yaw_joint,
        Frames.right_wrist_roll_joint,
        Frames.right_wrist_pitch_joint,
        Frames.right_wrist_yaw_joint,
        
        Frames.pelvis_link_1,
        Frames.pelvis_link_2,
        Frames.pelvis_link_3,
    ]