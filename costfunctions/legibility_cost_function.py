from costfunctions.costfn import CostFunction
import numpy as np

class LegibilityCostFunction(CostFunction):
    def __init__(self, env, costfn):
        super().__init__(env)
        self.costfn_for_CT = costfn  # cost function to compute CT

    # returns a SINGLE cost of the trajectory. 
    def __call__(self, trajectory):
        # Just in case some parent class function exists. 
        raise NotImplementedError("This method is intentionally not implemented. ")

    # This is your chance to be optimal about evaluating the whole trajectory..
    # especially if cost of the ith element is dependent on the previous steps. 
    # input = trajectory of length T
    # output = cost of [:i] FOR EACH element.. meaning array length=T. 
    # if scale=True, scale legibility costs to [-1, 1]
    def evaluate_whole_traj(self, trajectory, target_goal_idx=0, scale=False, T = None):
        T = len(trajectory) if T is None else T
        Pgr = 1/len(self.env.goals)
        real_goal_idx = target_goal_idx

        def f(t):
            return len(trajectory) - t

        hashed_VG = {}
        def VG(t, goal_idx, num_linspace): # THIS IS A PRETTY STRONG ASSUMPTION.
            # if t == 0:
            #     print(trajectory[t], self.env.goals[goal_idx])
            straightline = np.linspace(trajectory[t], self.env.goals[goal_idx], num=num_linspace, endpoint=True)
            return self.costfn_for_CT( straightline )

        numerators = []
        denominators = []
        for t in range(len(trajectory)):
            # numerator
            cost = self.costfn_for_CT( trajectory[:t] )
            # g = np.exp(VG(0, real_goal_idx) - VG(t, real_goal_idx)) * Pgr
            # h = sum( np.exp(VG(0, j) - VG(t, j)) * Pgr for j in range(len(self.env.goals)) )

            # original cost
            # g = np.exp(-1 * cost - VG(t, real_goal_idx)) * Pgr
            # h = np.exp(-1 * VG(0, real_goal_idx))


            # with denominator
            g = np.exp(-1 * (cost + VG(t, real_goal_idx, T-t+1))) * Pgr / np.exp(-1 * VG(0, real_goal_idx, T+1))
            h = np.sum([np.exp(-1 * (cost + VG(t, j, T-t+1))) * Pgr / np.exp(-1 * VG(0, j, T+1)) for j in range(len(self.env.goals))])
            # print("===")
            # print(g, h, g/h)

            # no denominator
            # g = np.exp(-1 * (cost + VG(t, real_goal_idx, T-t+1))) * Pgr
            # h = np.sum([np.exp(-1 * (cost + VG(t, j, T-t+1))) * Pgr for j in range(len(self.env.goals))])
            # print(g, h, g/h)

            PGR_given_traj = g/h

            update = PGR_given_traj * f(t)
            new_numerator = numerators[-1] + update if len(numerators) > 0 else update
            numerators.append(new_numerator)


            # denominator
            new_denom = denominators[-1] + f(t) if len(denominators) > 0 else f(t)
            denominators.append(new_denom)

        legibility_costs = np.array(numerators) / np.array(denominators)

        scale_factor = 2 
        if scale:
            return (-1 * legibility_costs * scale_factor) + 1 
        return -1 * legibility_costs