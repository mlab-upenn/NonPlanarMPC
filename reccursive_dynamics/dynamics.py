"""
PyTorch vehicle dynamics with GP residual learning
"""
import torch
import torch.nn as nn
import numpy as np


class DynamicBicycleModel(nn.Module):
    """Clean dynamic bicycle model in PyTorch"""
    
    def __init__(self, params, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # Store parameters as tensors
        self.mu = torch.tensor(params.MU, device=device)
        self.m = torch.tensor(params.M, device=device)
        self.I = torch.tensor(params.I, device=device)
        self.lr = torch.tensor(params.LR, device=device)
        self.lf = torch.tensor(params.LF, device=device)
        self.C_Sf = torch.tensor(params.C_SF, device=device)
        self.C_Sr = torch.tensor(params.C_SR, device=device)
        self.h = torch.tensor(params.H, device=device)
        self.g = torch.tensor(9.81, device=device)
        self.wheelbase = self.lf + self.lr
        
    def forward(self, state, control):
        """
        Compute state derivatives
        
        Args:
            state: [batch, 7] - [x, y, delta, v, yaw, yaw_rate, slip_angle]
            control: [batch, 2] - [steer_vel, accel]
            
        Returns:
            state_dot: [batch, 7]
        """
        # Extract states
        x = state[:, 0]
        y = state[:, 1]
        delta = state[:, 2]
        v = state[:, 3]
        yaw = state[:, 4]
        yaw_rate = state[:, 5]
        slip_angle = state[:, 6]
        
        # Extract controls
        delta_v = control[:, 0]
        a = control[:, 1]
        
        # Basic derivatives
        dx = v * torch.cos(yaw + slip_angle)
        dy = v * torch.sin(yaw + slip_angle)
        ddelta = delta_v
        dv = a
        
        # Low speed kinematic model
        dyaw_ks = v * torch.cos(slip_angle) * torch.tan(delta) / self.wheelbase
        
        dslip_angle_ks = (self.lr * delta_v) / (
            self.wheelbase * torch.cos(delta)**2 * 
            (1 + (torch.tan(delta)**2 * self.lr / self.wheelbase)**2)
        )
        
        ddyaw_ks = (1 / self.wheelbase) * (
            a * torch.cos(slip_angle) * torch.tan(delta)
            - v * torch.sin(slip_angle) * torch.tan(delta) * dslip_angle_ks
            + v * torch.cos(slip_angle) * delta_v / torch.cos(delta)**2
        )
        
        # High speed dynamic model
        dyaw_st = yaw_rate
        
        glr = self.g * self.lr - a * self.h
        glf = self.g * self.lf + a * self.h
        
        ddyaw_st = (
            -self.mu * self.m / (v * self.I * self.wheelbase) *
            (self.lf**2 * self.C_Sf * glr + self.lr**2 * self.C_Sr * glf) * yaw_rate +
            self.mu * self.m / (self.I * self.wheelbase) *
            (self.lr * self.C_Sr * glf - self.lf * self.C_Sf * glr) * slip_angle +
            self.mu * self.m / (self.I * self.wheelbase) * 
            self.lf * self.C_Sf * glr * delta
        )
        
        dslip_angle_st = (
            (self.mu / (v**2 * self.wheelbase) *
             (self.C_Sr * glf * self.lr - self.C_Sf * glr * self.lf) - 1) * yaw_rate
            - self.mu / (v * self.wheelbase) *
            (self.C_Sr * glf + self.C_Sf * glr) * slip_angle +
            self.mu / (v * self.wheelbase) * (self.C_Sf * glr) * delta
        )
        
        # Select based on velocity threshold
        use_kinematic = (v.abs() <= 1.5).float()
        dyaw = use_kinematic * dyaw_ks + (1 - use_kinematic) * dyaw_st
        ddyaw = use_kinematic * ddyaw_ks + (1 - use_kinematic) * ddyaw_st
        dslip_angle = use_kinematic * dslip_angle_ks + (1 - use_kinematic) * dslip_angle_st
        
        return torch.stack([dx, dy, ddelta, dv, dyaw, ddyaw, dslip_angle], dim=1)


class SimpleGPResidual(nn.Module):
    """Simple nearest-neighbor GP for residuals"""
    
    def __init__(self, max_samples=500, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.max_samples = max_samples
        
        # Storage for training data
        self.X_train = []
        self.Y_train = []
        
    def add_data(self, x, y):
        """Add training sample"""
        self.X_train.append(x.detach().cpu().numpy())
        self.Y_train.append(y.detach().cpu().numpy())
        
        if len(self.X_train) > self.max_samples:
            self.X_train.pop(0)
            self.Y_train.pop(0)
    
    def predict(self, x):
        """Nearest neighbor prediction"""
        if len(self.X_train) < 3:
            return torch.zeros(x.shape[0], device=self.device)
        
        X = torch.tensor(np.array(self.X_train), device=self.device)
        Y = torch.tensor(np.array(self.Y_train), device=self.device)
        
        # Compute distances
        dists = torch.cdist(x, X)
        nearest = torch.argmin(dists, dim=1)
        
        return Y[nearest]


class ResidualGPDynamics(nn.Module):
    """Dynamics with GP residual learning"""
    
    def __init__(self, params, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.base_dynamics = DynamicBicycleModel(params, device)
        
        # Three GPs for velocity, yaw_rate, slip_angle residuals
        self.gp_v = SimpleGPResidual(device=device)
        self.gp_yaw = SimpleGPResidual(device=device)
        self.gp_beta = SimpleGPResidual(device=device)
        
        self.total_updates = 0
        
    def forward(self, state, control):
        """
        Forward with GP corrections
        
        Args:
            state: [batch, 7]
            control: [batch, 2]
            
        Returns:
            state_dot: [batch, 7] with GP corrections applied
        """
        base_dot = self.base_dynamics(state, control)
        
        if len(self.gp_v.X_train) > 5:
            # Create GP input: [delta, v, yaw_rate, beta, steer_vel, accel]
            gp_input = torch.stack([
                state[:, 2], state[:, 3], state[:, 5], state[:, 6],
                control[:, 0], control[:, 1]
            ], dim=1)
            
            # Get predictions
            dv_residual = self.gp_v.predict(gp_input)
            dyaw_residual = self.gp_yaw.predict(gp_input)
            dbeta_residual = self.gp_beta.predict(gp_input)
            
            # Apply corrections
            base_dot[:, 3] += dv_residual
            base_dot[:, 5] += dyaw_residual
            base_dot[:, 6] += dbeta_residual
        
        return base_dot
    
    def update_gp(self, prev_state, control, dt, observed_state):
        """Update GP with observed transition"""
        with torch.no_grad():
            # Predict with base model
            base_dot = self.base_dynamics(prev_state.unsqueeze(0), control.unsqueeze(0))
            predicted = prev_state + base_dot.squeeze(0) * dt
            
            # Compute residuals
            dv_residual = (observed_state[3] - predicted[3]) / dt
            dyaw_residual = (observed_state[5] - predicted[5]) / dt
            dbeta_residual = (observed_state[6] - predicted[6]) / dt
            
            # Create GP input
            gp_input = torch.tensor([
                prev_state[2], prev_state[3], prev_state[5], prev_state[6],
                control[0], control[1]
            ], device=self.device)
            
            # Add to GPs
            self.gp_v.add_data(gp_input.unsqueeze(0), dv_residual.unsqueeze(0))
            self.gp_yaw.add_data(gp_input.unsqueeze(0), dyaw_residual.unsqueeze(0))
            self.gp_beta.add_data(gp_input.unsqueeze(0), dbeta_residual.unsqueeze(0))
            
            self.total_updates += 1
