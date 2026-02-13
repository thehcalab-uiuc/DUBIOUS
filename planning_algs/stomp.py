from planning_algs.planner import Planner

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

# naive implementation of STOMP
class STOMP(Planner):
    def __init__(self, env, costfn, **kwargs):
        super().__init__(env, costfn)
        self.n_waypoints = 45 if 'n_waypoints' not in kwargs else kwargs['n_waypoints']
        self.k_noisy_paths = 40 if 'k_noisy_paths' not in kwargs else kwargs['k_noisy_paths']
        self.r_inv_scaling = 1e4 if 'r_inv_scaling' not in kwargs else kwargs['r_inv_scaling']
        self.lambda_reg = 3 if 'lambda_reg' not in kwargs else kwargs['lambda_reg']

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
        M = R_inv / np.max(R_inv, axis=0)
        M *= (1/n)

        Q = np.inf
        converged = False
        n_iter = 0
        while not converged:
            # Create K noisy trajectories
            noisy_paths = []
            all_ek = []
            cov = R_inv / self.r_inv_scaling
            sampled_ek = np.random.multivariate_normal(np.zeros(n), cov, size=(dim*self.k_noisy_paths))
            for l in range(self.k_noisy_paths):
                e_k = np.array([sampled_ek[2*l], sampled_ek[2*l+1]]).T # shape (n, dim)
                noisy_paths.append(path + e_k)
                all_ek.append(e_k)
            noisy_paths = np.array(noisy_paths)  # shape (K, n)
            all_ek = np.array(all_ek)          # shape (K, n)

            S = np.zeros((self.k_noisy_paths, n))
            P = np.zeros((self.k_noisy_paths, n))
            for k in range(self.k_noisy_paths):
                for i in range(n):
                    S[k, i] = self.costfn(noisy_paths[k][:i])

            for i in range(n):
                all_sum = np.sum( np.exp(- (1/self.lambda_reg) * S[:, i]) , axis=0)
                if all_sum == 0:
                    all_sum = 1e-10
                for k in range(self.k_noisy_paths):
                    P[k, i] = np.exp(- (1/self.lambda_reg) * S[k, i]) / all_sum

            dtsquiggle = np.zeros((n, dim))
            for i in range(1, n):
                dtsquiggle[i] = np.sum( [ P[k, i] * all_ek[k][i] for k in range(self.k_noisy_paths) ], axis=0)

            path += (M @ dtsquiggle)
           
            path_cost_fn = sum( [ self.costfn(path[:i]) for i in range(n) ] )
            new_Q = np.sum(np.diag(0.5 * path.T @ (A.T@A) @ path))

            new_Q += path_cost_fn
            converged = np.abs(new_Q - Q) < .01
            # print(f"Iteration {n_iter}: Q={Q}, diff={np.abs(Q - new_Q)}, path_cost={path_cost_fn}")
            Q = new_Q
            n_iter += 1

        return path, n_iter, Q