from planning_algs.planner import Planner

import numpy as np
import numpy.linalg as la
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("QtAgg")

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
        self.convergence_threshold = 1e-5 if 'convergence_threshold' not in kwargs else kwargs['convergence_threshold']

    def _save_iteration_state(self, n_iter, path, pathinfo, save_dir):
        self.env.save_to_png(filename=f"{save_dir}/iter_{n_iter}.png", path=path)
        from experiment import Experiment # this line here and not at the top to prevent circular import
        Experiment.save_env_and_path(
            env=self.env,
            path=path,
            pathinfo=pathinfo,
            costfn=self.costfn,
            filename=f"{save_dir}/iter_{n_iter}_env_path.json"
        )

    def _merge_close_points(self, path, n_iter, removal_threshold=0.1):
        """Merge consecutive waypoints in path that are closer than removal_threshold.
        Returns the possibly shortened path.
        """
        i = 0
        while i < len(path) - 1:
            if la.norm(path[i+1] - path[i]) < removal_threshold:
                print(f"Iteration {n_iter}: Merged points {i} - distance was {la.norm(path[i+1] - path[i])}")
                path[i] = (path[i] + path[i+1]) / 2
                path = np.delete(path, i+1, axis=0)
            else:
                i += 1
        return path

    def plan_path(self, start_position, goal_idx, **kwargs):
        intermittent_save = kwargs.get("intermittent_save", False)
        save_dir = kwargs.get("save_dir", None)

        # Checks for validity * other init steps
        super().plan_path(start_position, goal_idx, **kwargs)
        dim = len(start_position)
        goal_position = self.env.goals[goal_idx]

        # Initial path (straightline)
        path = np.linspace(start_position, 
                            goal_position, 
                            num=self.n_waypoints, 
                            endpoint=True)

        Q = np.inf
        converged = False
        history = {}
        # while not converged:
        for n_iter in tqdm(range(self.max_iter+1)):
            if converged:
                print(f"Converged at iteration {n_iter}. Q={Q}")
                break
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
            smoother = 0 # np.sum(np.diag(0.5 * path.T @ (A.T@A) @ path)) # scalar... because we summed over diagonal
            new_Q = path_cost + smoother
            converged = np.abs(new_Q - Q) < self.convergence_threshold
            
            Q = new_Q

            # Merge points that are too close to each other
            path = self._merge_close_points(path, n_iter)

            # intermittent save. 
            if n_iter in [0, 10] or \
                (n_iter % 20 == 0 and self.max_iter <= 1000) or \
                (n_iter % 50 == 0 and self.max_iter > 1000):
                # intermittent save. 
                pathinfo = {
                    "Q": Q,
                    "path_cost": path_cost,
                    "smoother": smoother
                }
                if intermittent_save:
                    self._save_iteration_state(n_iter, path, pathinfo, save_dir)
                else:
                    history[n_iter] = {
                        "path": path.copy(),
                        **pathinfo
                    }


            # print(f"Iteration {n_iter}: Q={Q}, path_cost={path_cost}, smoother={smoother}")

            n_iter += 1
        
        self._save_iteration_state(n_iter, path, pathinfo, save_dir)
        return path, history 

