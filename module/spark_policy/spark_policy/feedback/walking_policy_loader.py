import numpy as np
import time

class SimpleWalkingPolicy:
    def __init__(self):
        self.phase = 0.0
        self.step_frequency = 0.8  # Hz (slow walking)
        self.step_height = 0.05    # meters
        self.hip_amplitude = 0.3   # radians
        self.knee_amplitude = 0.6  # radians
        
    def get_walking_control(self, dt=0.01):
        """
        Generate walking control for leg joints
        Returns: 8-element array [left_leg(4), right_leg(4)]
        Joint order: [hip_pitch, hip_roll, knee, ankle_pitch] per leg
        """
        # Update walking phase
        self.phase += 2 * np.pi * self.step_frequency * dt
        
        # Initialize control array for 8 leg joints
        leg_control = np.zeros(8)
        
        # Left leg indices: 0-3
        # Right leg indices: 4-7
        
        # Left leg pattern
        left_phase = self.phase
        leg_control[0] = self.hip_amplitude * np.sin(left_phase)      # Left hip pitch
        leg_control[1] = 0.1 * np.sin(left_phase * 2)                # Left hip roll (balance)
        leg_control[2] = max(0, self.knee_amplitude * np.sin(left_phase))  # Left knee (only bend forward)
        leg_control[3] = -0.5 * leg_control[0]                       # Left ankle pitch (compensate hip)
        
        # Right leg pattern (opposite phase)
        right_phase = self.phase + np.pi
        leg_control[4] = self.hip_amplitude * np.sin(right_phase)     # Right hip pitch  
        leg_control[5] = 0.1 * np.sin(right_phase * 2)               # Right hip roll (balance)
        leg_control[6] = max(0, self.knee_amplitude * np.sin(right_phase))  # Right knee
        leg_control[7] = -0.5 * leg_control[4]                       # Right ankle pitch
        
        return leg_control

class CPGWalkingPolicy:
    """Central Pattern Generator based walking - more sophisticated"""
    def __init__(self):
        self.oscillators = {
            'left_hip': {'phase': 0.0, 'freq': 1.0, 'amp': 0.4},
            'right_hip': {'phase': np.pi, 'freq': 1.0, 'amp': 0.4},
            'left_knee': {'phase': 0.5, 'freq': 2.0, 'amp': 0.8},
            'right_knee': {'phase': np.pi + 0.5, 'freq': 2.0, 'amp': 0.8},
        }
        
    def get_walking_control(self, dt=0.01):
        leg_control = np.zeros(8)
        
        # Update oscillators
        for osc in self.oscillators.values():
            osc['phase'] += 2 * np.pi * osc['freq'] * dt
            
        # Generate coordinated walking pattern
        # Left leg
        leg_control[0] = self.oscillators['left_hip']['amp'] * np.sin(self.oscillators['left_hip']['phase'])
        leg_control[1] = 0.05 * np.sin(self.oscillators['left_hip']['phase'] * 2)  # Hip roll for balance
        leg_control[2] = max(0, self.oscillators['left_knee']['amp'] * np.sin(self.oscillators['left_knee']['phase']))
        leg_control[3] = -0.3 * leg_control[0]  # Ankle compensates hip
        
        # Right leg  
        leg_control[4] = self.oscillators['right_hip']['amp'] * np.sin(self.oscillators['right_hip']['phase'])
        leg_control[5] = 0.05 * np.sin(self.oscillators['right_hip']['phase'] * 2)
        leg_control[6] = max(0, self.oscillators['right_knee']['amp'] * np.sin(self.oscillators['right_knee']['phase']))
        leg_control[7] = -0.3 * leg_control[4]
        
        return leg_control

def load_walking_policy(policy_type="simple"):
    """
    Load walking policy
    Args:
        policy_type: "simple", "cpg", or path to trained model
    """
    print("Hello from walking policy loader")
    if policy_type == "simple":
        return SimpleWalkingPolicy()
    elif policy_type == "cpg":
        return CPGWalkingPolicy()
    else:
        # For future: load trained walking model
        raise NotImplementedError(f"Walking policy {policy_type} not implemented")

def get_walking_action(walking_policy, dt=0.01):
    """
    Get walking control action
    Returns: 8-element array for leg joints
    """
    return walking_policy.get_walking_control(dt)