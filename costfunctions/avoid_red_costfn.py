from costfunctions.costfn import CostFunction
import numpy as np

# equatoin 3.7 from the thesis
class AvoidRedCostFn(CostFunction):
    def __init__(self, env):
        super().__init__(env)

    # trajectory = np array of size (T, 2)
    # returns an INT. 
    def __call__(self, trajectory):
        diff = trajectory[1:] - trajectory[:-1]
        norms = np.linalg.norm(diff, axis=1)
        normsquared = norms * norms
        s = np.sum(1 * normsquared)

        if len(trajectory) > 0:
            # Multiply by % of trajectory in the green zone
            n_bad_scores = 0
            for t in trajectory:
                if self.env.get_score(t) <= 0:
                    n_bad_scores += 1
            s *= 100* ((n_bad_scores / len(trajectory)))

        return s / 2.0
