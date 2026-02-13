import numpy as np

class CostFunction:
    def __init__(self, env):
        self.env = env

    # returns a SINGLE cost of the trajectory. 
    def __call__(self, trajectory):
        raise NotImplementedError("This method should be overridden by subclasses")

    # This is your chance to be optimal about evaluating the whole trajectory..
    # especially if cost of the ith element is dependent on the previous steps. 
    # input = trajectory of length T
    # output = cost of [:i] FOR EACH element.. meaning array length=T. 
    def evaluate_whole_traj(self, trajectory):
        # dumb implementation; override for smarter functions. 
        costs = []
        for i in range(len(trajectory)):
            costs.append( self.__call__( trajectory[:i] ) )
        return np.array(costs)

    # in case we can optimize to batch evaluate many whole trajectories.
    # trajectories = list of trajectories size (K, T, dim) (k is # of trajs)
    # output = array of size (K, T) costs for each trajectory
    def batch_evaluate(self, trajectories):
        # dumb way; override this function if cost function can be done this way
        costs = []
        for traj in trajectories:
            costs.append( self.evaluate_whole_traj( traj ) )
        return np.array(costs)
