from costfunctions.costfn import CostFunction
import numpy as np

# equatoin 3.7 from the thesis
class DraganCostFunction(CostFunction):
    def __init__(self, world):
        super().__init__(world)

    # trajectory = np array of size (T, 2)
    # returns an INT. 
    def __call__(self, trajectory):
        # s = 0
        # for i in range(1, len(trajectory)):
        #     diff = trajectory[i] - trajectory[i-1]
        #     norm = np.linalg.norm(diff)
        #     normsquared = norm * norm

        #     s += (self.world.dt * normsquared)

        diff = trajectory[1:] - trajectory[:-1] # shape (T-1, 2)
        norms = np.linalg.norm(diff, axis=1) # shape (T-1,)
        normsquared = norms * norms
        s = np.sum(1 * normsquared)


        return s / 2

    # def evaluate_whole_traj(self, trajectory):
    #     diff = trajectory[1:] - trajectory[:-1]
    #     norms = np.linalg.norm(diff, axis=1)
    #     normsquared = norms * norms
    #     normsquared *= 1 # dt = 1
    #     normsquared /= 2
        # return np.cumsum(normsquared)