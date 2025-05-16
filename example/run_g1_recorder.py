from spark_pipeline import G1BenchmarkPipeline, G1BenchmarkPipelineConfig
from spark_utils import update_class_attributes
import gc
import numpy as np

# Monkey patch the G1BenchmarkPipeline to prevent memory issues during long runs
def patch_pipeline_run(original_run):
    def patched_run(self):
        # Reset environment
        agent_feedback, task_info = self.env.reset()
        
        # Initial action
        u_safe, action_info = self.algo.act(agent_feedback, task_info)
        
        # For memory efficiency, clear accumulated data periodically
        step_counter = 0
        
        print("Running with memory-efficient mode. Press Ctrl+C to exit.")
        try:
            while step_counter < self.max_num_steps:
                # Reset if necessary
                if task_info["done"]:
                    agent_feedback, task_info = self.env.reset()
                    u_safe, action_info = self.algo.act(agent_feedback, task_info)
                
                # Environment step
                agent_feedback, task_info = self.env.step(u_safe)
                
                # Next action
                u_safe, action_info = self.algo.act(agent_feedback, task_info)
                
                # Post physics step (rendering, logging, etc.)
                self.post_physics_step(agent_feedback, task_info, action_info)
                
                step_counter += 1
                
                # Periodically clear accumulated data to prevent memory issues
                if step_counter % 10000 == 0:
                    print(f"Step {step_counter}: Performing memory cleanup")
                    # Save current statistics
                    min_dist = min(self.min_dist_robot_to_env) if self.min_dist_robot_to_env else 0
                    avg_dist = sum(self.min_dist_robot_to_env)/len(self.min_dist_robot_to_env) if self.min_dist_robot_to_env else 0
                    mean_goal = sum(self.mean_dist_goal)/len(self.mean_dist_goal) if self.mean_dist_goal else 0
                    
                    # print(f"Average distance to obstacle: {avg_dist:.4f}")
                    # print(f"Minimum distance to obstacle: {min_dist:.4f}")
                    # print(f"Average distance to goal: {mean_goal:.4f}")
                    
                    # Reset accumulated data
                    self.min_dist_robot_to_env = []
                    self.mean_dist_goal = []
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Flush logger to clear buffered data
                    self.logger._summ_writer.flush()
            
            # End of benchmark
            print("Simulation ended")
            
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        
        # Convert to numpy arrays for final stats (if any data collected)
        if self.min_dist_robot_to_env:
            self.min_dist_robot_to_env = np.array(self.min_dist_robot_to_env)
            self.mean_dist_goal = np.array(self.mean_dist_goal)
            
            print("average distance to obstacle: ", np.mean(self.min_dist_robot_to_env))
            print("minimum distance to obstacle: ", np.min(self.min_dist_robot_to_env))
            print("average distance to goal: ", np.mean(self.mean_dist_goal))
            print("maximum distance to goal: ", np.max(self.mean_dist_goal))
    
    return patched_run

if __name__ == "__main__":
    
    cfg = G1BenchmarkPipelineConfig()
    cfg.max_num_steps = float('inf')  # Set to infinity for indefinite running
    
    # --------------------- configure safe control algorithm --------------------- #
    safe_control_algo = 'ssa' # 'ssa', 'sss', 'cbf', 'pfm', 'sma'
    params = {
        'ssa': dict(
                    class_name = "SafeSetAlgorithm",
                    eta = 1.0,
                    safety_buffer = 0.1, # positive to allow hold state
                    slack_weight = 1e3,
                    control_weight = [
                        1.0, 1.0, 1.0, # waist
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # left arm
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # right arm
                        10.0, 10.0, 10.0 # locomotion
                    ]
                ),
        'sss': dict(
                    class_name = "SublevelSafeSetAlgorithm",
                    lambda_SSS = 1.0,
                    slack_weight = 1e3,
                    control_weight = [
                        1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0
                    ]
                ),
        'cbf': dict(
                    class_name = "ControlBarrierFunction",
                    lambda_cbf = 1,
                    slack_weight = 1e3,
                    control_weight = [
                        1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0
                    ]
                ),
        'pfm': dict(
                    class_name = "PotentialFieldMethod",
                    lambda_pfm = 0.1
                ),
        'sma': dict(
                    class_name = "SlidingModeAlgorithm",
                    c_2 = 1.0
                )
    }
    
    update_class_attributes(cfg.algo.safe_controller.safe_algo, params[safe_control_algo])
    
    # ------------------------------- run pipeline ------------------------------- #
    try:
        # Apply monkey patch to run method for memory efficiency
        original_run = G1BenchmarkPipeline.run
        G1BenchmarkPipeline.run = patch_pipeline_run(original_run)
        
        # Create and run pipeline
        pipeline = G1BenchmarkPipeline(cfg)
        pipeline.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        if 'pipeline' in locals():
            del pipeline
        
        # Restore original method
        if 'original_run' in locals():
            G1BenchmarkPipeline.run = original_run
            
        gc.collect()
    
    
    