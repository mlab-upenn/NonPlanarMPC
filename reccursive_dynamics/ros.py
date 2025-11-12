#!/usr/bin/env python3
"""
PyTorch MPPI control node with GP learning
"""
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.spatial.transform import Rotation as R

from dynamics import ResidualGPDynamics 
from mppi import MPPIController


class TorchMPPINode(Node):
    """ROS2 node for PyTorch MPPI with GP"""
    
    def __init__(self, raceline, use_gp=False):
        super().__init__('torch_mppi_controller')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")
        
        self.use_gp = use_gp
        
        # Load raceline
        self.load_raceline(raceline)
        
        # Initialize dynamics
        params = self.get_f1tenth_params()
        if use_gp:
            self.dynamics = ResidualGPDynamics(params, self.device)
            self.get_logger().info("GP MODE - Learning enabled")
        else:
            from dynamics import DynamicBicycleModel
            self.dynamics = DynamicBicycleModel(params, self.device)
            self.get_logger().info("BASELINE MODE - No learning")
        
        # Initialize MPPI
        config = {
            'horizon': 20,
            'n_samples': 1000,
            'dt': 0.05,
            'lambda': 1.0,
            'u_min': [-2.0, -3.0],
            'u_max': [2.0, 3.0],
            'noise_sigma': [0.5, 0.5],
            'Q': np.diag([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
            'R': np.diag([0.01, 0.01])
        }
        self.controller = MPPIController(self.dynamics, config, self.device)
        
        # ROS setup
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', 
                                                  self.odom_callback, 10)
        
        # State tracking
        self.delta = 0.0
        self.speed = 0.0
        self.prev_state = None
        self.prev_control = None
        self.prev_time = None
        self.step_count = 0
        
    def load_raceline(self, raceline_file):
        """Load reference raceline"""
        # Simplified - load from CSV
        data = np.loadtxt(raceline_file, delimiter=';', skiprows=3)
        self.ref_x = data[:, 1]
        self.ref_y = data[:, 2]
        self.ref_vx = data[:, 5]
        self.num_waypoints = len(self.ref_x)
        
    def get_f1tenth_params(self):
        """Get F1Tenth vehicle parameters"""
        class Params:
            MU = 1.0
            M = 3.47
            I = 0.04712
            LR = 0.17145
            LF = 0.15875
            C_SF = 4.718
            C_SR = 5.4562
            H = 0.074
        return Params()
    
    def odom_callback(self, msg):
        """Main control loop"""
        self.step_count += 1
        current_time = self.get_clock().now().nanoseconds * 1e-9
        
        # Extract state
        state = self.extract_state(msg)
        
        # Update GP if using learning
        if self.use_gp and self.prev_state is not None:
            dt = current_time - self.prev_time
            if 0.001 < dt < 0.5:
                self.dynamics.update_gp(self.prev_state, self.prev_control, dt, state)
        
        # Get reference trajectory
        ref_traj = self.get_reference_trajectory(state)
        
        # Solve MPPI
        control = self.controller.solve(state.cpu().numpy(), ref_traj)
        
        # Integrate and publish
        self.delta += control[0] * 0.05
        self.delta = np.clip(self.delta, -0.52, 0.52)
        self.speed += control[1] * 0.05
        self.speed = np.clip(self.speed, 0.0, 10.0)
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = self.delta
        drive_msg.drive.speed = self.speed
        self.drive_pub.publish(drive_msg)
        
        # Store for GP update
        if self.use_gp:
            self.prev_state = state
            self.prev_control = torch.tensor(control, device=self.device)
            self.prev_time = current_time
        
        # Log progress
        if self.step_count % 100 == 0:
            gp_info = f"Updates: {self.dynamics.total_updates}" if self.use_gp else "No GP"
            self.get_logger().info(f"Step {self.step_count}, {gp_info}, Speed: {self.speed:.2f}")
    
    def extract_state(self, msg):
        """Extract state from odometry"""
        pose = msg.pose.pose
        twist = msg.twist.twist
        
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        yaw = R.from_quat(quat).as_euler('xyz')[2]
        
        beta = np.arctan2(twist.linear.y, twist.linear.x) if abs(twist.linear.x) > 1e-6 else 0.0
        
        state = torch.tensor([
            pose.position.x, pose.position.y, self.delta,
            twist.linear.x, yaw, twist.angular.z, beta
        ], device=self.device)
        
        return state
    
    def get_reference_trajectory(self, state):
        """Get local reference trajectory"""
        # Find nearest waypoint
        x, y = state[0].item(), state[1].item()
        dists = np.sqrt((self.ref_x - x)**2 + (self.ref_y - y)**2)
        idx = np.argmin(dists)
        
        # Extract horizon
        horizon = self.controller.horizon + 1
        indices = (np.arange(horizon) + idx) % self.num_waypoints
        
        ref_traj = np.zeros((horizon, 7))
        ref_traj[:, 0] = self.ref_x[indices]
        ref_traj[:, 1] = self.ref_y[indices]
        ref_traj[:, 3] = self.ref_vx[indices]
        
        return ref_traj


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raceline', required=True)
    parser.add_argument('--use-gp', action='store_true')
    args = parser.parse_args()
    
    rclpy.init()
    node = TorchMPPINode(args.raceline, args.use_gp)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down gracefully...")
    finally:
        try:
            node.destroy_node()
        except:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
