"""
PyTorch MPPI implementation - clean and simple
"""
import torch
import torch.nn as nn


class MPPIController(nn.Module):
    """Model Predictive Path Integral controller in PyTorch"""
    
    def __init__(self, dynamics_model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.dynamics = dynamics_model.to(device)
        
        # MPPI parameters
        self.horizon = config['horizon']
        self.n_samples = config['n_samples']
        self.dt = config['dt']
        self.lambda_ = config['lambda']
        self.u_min = torch.tensor(config['u_min'], device=device)
        self.u_max = torch.tensor(config['u_max'], device=device)
        self.noise_sigma = torch.tensor(config['noise_sigma'], device=device)
        
        # Cost matrices
        self.Q = torch.tensor(config['Q'], device=device)
        self.R = torch.tensor(config['R'], device=device)
        
        # Control sequence
        self.u_nom = torch.zeros(self.horizon, 2, device=device)
        
    def rollout(self, x0, u_sequence, ref_traj):
        """
        Rollout dynamics for horizon
        
        Args:
            x0: [batch, 7] initial states
            u_sequence: [batch, horizon, 2] control sequences
            ref_traj: [horizon+1, 7] reference trajectory
            
        Returns:
            cost: [batch] total costs
        """
        batch_size = u_sequence.shape[0]
        x = x0.unsqueeze(0).repeat(batch_size, 1)
        total_cost = torch.zeros(batch_size, device=self.device)
        
        for t in range(self.horizon):
            u = u_sequence[:, t, :]
            
            # Integrate dynamics
            x_dot = self.dynamics(x, u)
            x = x + x_dot * self.dt
            
            # Compute stage cost
            x_err = x - ref_traj[t+1].unsqueeze(0)
            stage_cost = (
                torch.sum(x_err @ self.Q * x_err, dim=1) +
                torch.sum(u @ self.R * u, dim=1)
            )
            total_cost += stage_cost
        
        return total_cost
    
    def solve(self, x0, ref_traj):
        """
        Solve MPPI optimization
        
        Args:
            x0: [7] current state
            ref_traj: [horizon+1, 7] reference trajectory
            
        Returns:
            u_opt: [2] optimal control for next step
        """
        x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)
        ref_traj = torch.tensor(ref_traj, dtype=torch.float32, device=self.device)
        
        # Sample control perturbations
        noise = torch.randn(self.n_samples, self.horizon, 2, device=self.device) * self.noise_sigma
        u_samples = self.u_nom.unsqueeze(0) + noise
        u_samples = torch.clamp(u_samples, self.u_min, self.u_max)
        
        # Rollout all samples
        costs = self.rollout(x0, u_samples, ref_traj)
        
        # Compute weights
        beta = torch.min(costs)
        weights = torch.exp(-self.lambda_ * (costs - beta))
        weights = weights / torch.sum(weights)
        
        # Weighted average of control sequences
        self.u_nom = torch.sum(weights.view(-1, 1, 1) * u_samples, dim=0)
        
        # Shift for warm start
        self.u_nom = torch.roll(self.u_nom, -1, dims=0)
        self.u_nom[-1] = 0.0
        
        return self.u_nom[0].cpu().numpy()
