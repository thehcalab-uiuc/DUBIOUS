import numpy as np
import matplotlib.pyplot as plt

from environment import env

# NOTE: I FEEL LIKE THERE'S A LOT OF DUPLICATE WORK HERE 
# ESPECIALLY IN TERMS OF LEGIBILITY, ETC
# BUT OH WELL; THE CODE IS CLEAN. 
class PathEvaluator:
    def __init__(self, env, path, pathinfo, evaluator_costfn):
        self.env = env
        self.path = np.array(path)
        self.pathinfo = pathinfo
        self.costfn = evaluator_costfn

        fig, ax = plt.subplots()
        env.display(ax)
        ax.plot(self.path[:, 0], self.path[:, 1], 'b.-', label='Path')
        # plt.show()

        self.goal_idx = -1
        for i, goal in enumerate(self.env.goals):
            if np.array_equal(goal, path[-1]):
                self.goal_idx = i
                break

        # set self.decoy_goal_idx to the farthest goal from the actual
        dists = [np.linalg.norm(np.array(goal) - np.array(path[-1])) for goal in self.env.goals]
        self.decoy_goal_idx = np.argmax(dists)

        self.preprocess()

    # returns V_G(Q) = cost of shortest path from Q to G. 
    def _V(self, Q, G, num):
        shortest_path = np.linspace(Q, G, num + 1, endpoint=True)  # simple straight-line path
        cost = self.costfn(shortest_path)
        return cost

    # Prepares self.probabilities, which is a (|O|+1 x |G| x n) array of probabilities
    # ans[o][g][n] = P(g | path up to n timesteps) as per observer o
    # if o = -1, this is the "whole env" observer
    def preprocess(self):
        probabilities = np.zeros((self.env.n_observers+1,
                                  len(self.env.goals),
                                  len(self.path)))
        # o+1 x n 
        visibility_masks = [self.env.get_observer_view(self.path, o) for o in range(self.env.n_observers)]
        visibility_masks.append(np.ones(len(self.path), dtype=bool))  # whole env observer
        visibility_masks = np.array(visibility_masks, dtype=bool)
        assert visibility_masks.shape == (self.env.n_observers+1, len(self.path))
        PG = 1/len(self.env.goals) # uniform goal distribution
        
        # for o in range(self.env.n_observers+1):
        #     for n in range(len(self.path)):
        #         # turn all visibility after point n to False
        #         this_vis_mask = visibility_masks[o].copy()
        #         this_vis_mask[n+1:] = False
        #         visible_path = self.path[this_vis_mask]
        #         if len(visible_path) == 0:
        #             continue
        #         cost = self.costfn(visible_path) # single number
        #         VGs = [self._V(visible_path[-1], self.env.goals[i], n) \
        #                 for i in range(len(self.env.goals))] # VGs[i] = V_{G_i}(visible_path[-1])
                
        #         exp_neg_costs = np.array([np.exp(-cost - VG) for VG in VGs])
        #         assert exp_neg_costs.shape == (len(self.env.goals),)

        #         PGs = (exp_neg_costs*PG) / np.sum(exp_neg_costs*PG)  # P(G_i | visible_path)
        #         probabilities[o, :, n] = PGs

        # self.probabilities = probabilities
        # return probabilities

        pathlen = len(self.path)

        for o in range(self.env.n_observers + 1):
            for g in range(len(self.env.goals)):
                for n in range(1, len(self.path)):
                    this_vis_mask = visibility_masks[o].copy()
                    this_vis_mask[n+1:] = False
                    visible_path = self.path[this_vis_mask]
                    if len(visible_path) == 0:
                        continue

                    visible_to_end = len(self.path) - np.argmax(this_vis_mask)
                    remaining_length = np.argmax(this_vis_mask[::-1])
                    if remaining_length < 2:
                        remaining_length = 2
                
                    cost = self.costfn(visible_path)
                    # numerator = np.exp(self._V(visible_path[0], self.env.goals[g], visible_to_end) - self._V(visible_path[-1], self.env.goals[g], remaining_length)) * PG
                    # denominator = np.sum([np.exp(self._V(visible_path[0], self.env.goals[i], visible_to_end) - self._V(visible_path[-1], self.env.goals[i], remaining_length)) * PG for i in range(len(self.env.goals))])
                    # print(n, numerator, denominator, visible_path[0], visible_path[-1])

                    # print(self._V(visible_path[-1], self.env.goals[g], n))
                    # if n >= pathlen - 1 and o == 1:
                    #     for g1 in range(len(self.env.goals)):
                    #         print()
                    #         print("visible_to_end: ", visible_to_end)
                    #         print("remaining_length: ", remaining_length)
                    #         print("S: ", visible_path[0])
                    #         print("G: ", self.env.goals[g1])
                    #         VS = self._V(visible_path[0], self.env.goals[g1], visible_to_end)
                    #         VG = self._V(visible_path[-1], self.env.goals[g1], remaining_length)
                    #         print("cost: ", cost)
                    #         print("VG: ", VG)
                    #         print("VS: ", VS)
                    #         print("cost + VG - VS: ", cost + VG - VS)
                    #     print("===")

                    numerator = np.exp(-1 * (cost + self._V(visible_path[-1], self.env.goals[g], remaining_length))) * PG / np.exp(-1 * self._V(visible_path[0], self.env.goals[g], visible_to_end))
                    denominator = np.sum([np.exp(-1 * (cost + self._V(visible_path[-1], self.env.goals[i], remaining_length))) * PG / np.exp(-1 * self._V(visible_path[0], self.env.goals[i], visible_to_end)) for i in range(len(self.env.goals))])
                    # denominator = np.sum([np.exp(-1 * (cost + self._V(visible_path[-1], self.env.goals[i], remaining_length))) * PG for i in range(len(self.env.goals))])            
                    probabilities[o, g, n] = numerator / denominator

        for o in range(self.env.n_observers + 1):
            for n in range(len(self.path)):
                sum_probs = np.sum(probabilities[o, :, n])
                if sum_probs > 0:
                    probabilities[o, :, n] /= sum_probs
        self.probabilities = probabilities
        return probabilities

    def calculate_all_legibility_scores(self):
        leg_scores = np.zeros((self.env.n_observers + 1, len(self.env.goals)))
        for g in range(len(self.env.goals)):
            leg_scores[:, g] = self.calculate_legibility_score(goal_idx=g)
        return leg_scores

    # Returns total legibility score, and then legibility scores for each observer. [total, obs1, obs2, ...]
    # Legibility is calculated as Eq 5. in "Generating Legible Motion" (Dragan & Srinivasa) 
    def calculate_legibility_score(self, goal_idx=None):
        if goal_idx is None:
            goal_idx = self.goal_idx
        leg_scores_numerators = np.zeros(self.env.n_observers + 1)
        leg_scores_denominators = np.zeros(self.env.n_observers + 1)
        T = len(self.path)
        for o in range(self.env.n_observers + 1):
            for n in range(len(self.path)):
                P = self.probabilities[o, goal_idx, n]
                f = T-n
                leg_scores_numerators[o] += P * f
                leg_scores_denominators[o] += f
        return leg_scores_numerators / leg_scores_denominators

    # returns (|O|+1 x |G| x n) array of probabilities
    # ans[o][g][n] = P(g | path up to n timesteps) as per observer o
    def calculate_prob_values(self):
        return self.probabilities

    # Requires to run calculate_prob_values first. [total, obs1, obs2, ...]
    # For each observer, finds the earliest timestep where 
    # predicted goal is same as correct goal with threshold margin.
    # ex. probabilites = [0.33, 0.33, 0.34] t=0.05 -> not correct
    # ex. probabilities = [0.10, 0.10, 0.80] t=0.05 -> correct
    def calculate_earliest_correct_guess(self, threshold=0.005):
        # print("THRESHOLD: ", threshold)
        pred_times = np.ones(self.env.n_observers + 1) * -1
        for obs in range(self.env.n_observers + 1):
            for t in range(len(self.path)):
                top_pred = np.argmax(self.probabilities[obs, :, t])
                if top_pred != self.goal_idx:
                    continue
                
                top_conf = self.probabilities[obs, top_pred, t]
                top_remove = np.delete(self.probabilities[obs, :, t], top_pred)
                next_conf = np.max(top_remove)
                if top_conf - next_conf > threshold:
                    pred_times[obs] = t
                    break

        return pred_times

    def calculate_earliest_correct_guess_percent(self, threshold=0.05):
        # print("THRESHOLD: ", threshold)
        pred_times = self.calculate_earliest_correct_guess(threshold)
        percent_times = pred_times / len(self.path)
        return percent_times
    
    # Requires to run calculate_prob_values first. 
    # Returns [total, obs1, obs2, ...]
    # returns % correct guesses AFTER first correct guess. 
    def calculate_percent_correct_guesses(self, threshold=0.05):
        # print("THRESHOLD: ", threshold)
        num_correct = np.zeros(self.env.n_observers + 1)
        first_correct = self.calculate_earliest_correct_guess(threshold)
        N = len(self.path)

        # for each observer
        #   number of guesses correct above threshold / number of timesteps
        for obs in range(self.env.n_observers + 1):
            for t in range(len(self.path)):
                top_pred = np.argmax(self.probabilities[obs, :, t])
                if top_pred != self.goal_idx:
                    continue
                
                top_conf = self.probabilities[obs, top_pred, t]
                top_remove = np.delete(self.probabilities[obs, :, t], top_pred)
                next_conf = np.max(top_remove)
                if top_conf - next_conf > threshold:
                    num_correct[obs] += 1

        return num_correct / (N - first_correct + 1e-5)
 
    def calculate_illegibility_decoy(self):
        leg_scores_numerators = np.zeros(self.env.n_observers + 1)
        leg_scores_denominators = np.zeros(self.env.n_observers + 1)
        T = len(self.path)
        for o in range(self.env.n_observers + 1):
            for n in range(len(self.path)):
                P = self.probabilities[o, self.decoy_goal_idx, n]
                f = T-n
                leg_scores_numerators[o] += P * f
                leg_scores_denominators[o] += f
        return leg_scores_numerators / leg_scores_denominators
    
    def calculate_illegibility_delay(self):    
        leg_scores_numerators = np.zeros(self.env.n_observers + 1)
        leg_scores_denominators = np.zeros(self.env.n_observers + 1)
        T = len(self.path)
        for o in range(self.env.n_observers + 1):
            for n in range(len(self.path)):
                P = self.probabilities[o, self.goal_idx, n]
                P_others = np.delete(self.probabilities[o, :, n], self.goal_idx)
                abs_sum =  np.sum(np.abs(P-P_others)) / len(self.env.goals)

                f = T-n
                leg_scores_numerators[o] += f* (1-abs_sum)
                leg_scores_denominators[o] += f
        scores = leg_scores_numerators / leg_scores_denominators
        return scores/3 # divide by 3 is arbitrary, just gets the delay scores in the same range as the decoy ones. 


    def calculate_illegibility_score(self):
        print("Illegibility decoy: ", self.calculate_illegibility_decoy())
        print("Illegibility delay: ", self.calculate_illegibility_delay())
        return np.max([self.calculate_illegibility_decoy(), self.calculate_illegibility_delay()], axis=0)