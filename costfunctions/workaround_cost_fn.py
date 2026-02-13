from costfunctions.costfn import CostFunction
import numpy as np

# for best results use  p.plan_path(start_position = [-0.9, 0.9], goal_idx = 1)
class WorkaroundCostFunction(CostFunction):
    def __init__(self, world):
        super().__init__(world)


    # trajectory = np array of size (T, 2)
    def __call__(self, trajectory):
        if len(trajectory) == 0:
            return 0
        return self.evaluate_point(trajectory[-1])
    
    def evaluate_point(self, point):
        # breakpoint()
        x, y = 0, point[0]
        # if y > 1 or y < -1 or x > 1 or x < -1:
        #     return 1
        return ((y-1)**2 if y <4 else 1)
