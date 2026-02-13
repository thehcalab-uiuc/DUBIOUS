from planning_algs.planner import Planner

import numpy as np
import numpy.linalg as la
import scipy as sp

import matplotlib.pyplot as plt

from tqdm import tqdm

# This file contains np/sp math that gives us a pretty big speedup over regular stomp.py
class STOMP(Planner):
    def __init__(self, env, costfn, **kwargs):
        super().__init__(env, costfn)
        self.n_waypoints = 45 if 'n_waypoints' not in kwargs else kwargs['n_waypoints']
        self.k_noisy_paths = 40 if 'k_noisy_paths' not in kwargs else kwargs['k_noisy_paths']
        self.r_inv_scaling = 1e3 if 'r_inv_scaling' not in kwargs else kwargs['r_inv_scaling']
        self.lambda_reg = 0.1 if 'lambda_reg' not in kwargs else kwargs['lambda_reg']
        self.max_iter = 1500 if 'max_iter' not in kwargs else kwargs['max_iter'] 
        self.convergence_threshold = 0 if 'convergence_threshold' not in kwargs else kwargs['convergence_threshold']

    def plan_path(self, start_position, goal_idx, **kwargs):
        # Checks for validity * other init steps
        super().plan_path(start_position, goal_idx, **kwargs)
        dim = len(start_position)
        goal_position = self.env.goals[goal_idx]

        # Initial path (straightline)
        path = np.linspace(start_position, 
                            goal_position, 
                            num=self.n_waypoints, 
                            endpoint=True)
        n = len(path)

        # finite difference matrix (Eqn 2)
        A = np.zeros((n+2, n))
        A[:n, :] += np.eye(n)
        A[1:n+1, :] += -2 * np.eye(n)
        A[2:n+2, :] += np.eye(n)
        # This SHOULD work for any dimension...

        # These don't change with dimension
        R_inv = np.linalg.inv(A.T @ A)
        M = R_inv / (n*np.max(R_inv, axis=0))
        cov = R_inv / self.r_inv_scaling

        Q = np.inf
        # converged = False
        history = {}
        # while not converged:
        for n_iter in tqdm(range(self.max_iter+1)):
            # Create K noisy trajectories
            # path has shape(n, dim)
            original_paths = np.stack([path]*self.k_noisy_paths) # shape (K, n, dim)
            e_k = np.random.multivariate_normal(np.zeros(n), cov, size=(dim*self.k_noisy_paths))
            e_k = e_k.reshape((self.k_noisy_paths, dim, n)).transpose((0, 2, 1))  # shape (K, n, dim)
            noisy_paths = original_paths + e_k  # shape (K, n, dim)

            S = self.costfn.batch_evaluate(noisy_paths) # shape (K, n) 
            P = sp.special.softmax( - (1/self.lambda_reg) * S , axis=0)  # shape (K, n)
            
            dtsquiggle = np.multiply(np.stack([P]*dim, axis=-1), e_k) # shape (K, n, dim)
            dtsquiggle = np.sum(dtsquiggle, axis=0) # shape (n, dim)

            dtsquiggle = M @ dtsquiggle  # shape (n, dim)
            dtsquiggle[0] = 0
            dtsquiggle[-1] = 0
            
            path += dtsquiggle
            
            path_cost = np.sum(self.costfn.evaluate_whole_traj(path)) # scalar
            smoother = np.sum(np.diag(0.5 * path.T @ (A.T@A) @ path)) # scalar... because we summed over diagonal
            new_Q = path_cost + smoother
            # converged = np.abs(new_Q - Q) < self.convergence_threshold
            
            Q = new_Q

            # Add to history every so frequently
            if n_iter in [0, 10] or n_iter % 20 == 0:
                history[n_iter] = {
                    "path": path.copy(),
                    "Q": Q,
                    "path_cost": path_cost,
                    "smoother": smoother
                }

            n_iter += 1

        return path, history
