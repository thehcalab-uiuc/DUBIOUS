from planning_algs import Planner
from planning_algs.graph import Graph
import numpy as np

class RRT(Planner):
    def __init__(self, env, costfn=None):
        super().__init__(env, costfn)
        self.max_rrt_iterations = 1000
        self.goal_radius= 0.1
        self.rrt_expand_radius = 0.1

    def plan_path(self, start_position, goal_idx, **kwargs):
        super().plan_path(start_position, goal_idx, **kwargs)
        goal_position = self.env.goals[goal_idx]

        # reset self.g
        g = Graph(directed=False) # default weight fn is eucliedan distance anyway
        self.g = g
        g.add_vertex(tuple(start_position))
        for _ in range(self.max_rrt_iterations):
            # sample a random point in bounds 
            p = (np.random.uniform(self.env.x_range[0], self.env.x_range[1]), \
                    np.random.uniform(self.env.y_range[0], self.env.y_range[1]))
            
            # find nearest node in tree
            nearest_neighbor = g.find_nearest(p, 1)[0]

            # extend towards sampled point
            direction = np.array(p) - np.array(nearest_neighbor)
            direction /= np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else 1
            new_node_pos = np.array(nearest_neighbor) + direction * self.rrt_expand_radius
            new_tuple = tuple(new_node_pos)

            # check for collision
            if self.env.point_in_collision(new_node_pos):
                continue
            if self.edge_in_collision(np.array(nearest_neighbor), new_node_pos):
                continue

            g.add_vertex(new_tuple)
            g.add_edge(nearest_neighbor, new_tuple)

            # Check if we are now close enough to the goal
            if np.linalg.norm(new_node_pos - goal_position) < self.goal_radius:
                goal_tuple = tuple(goal_position)
                g.add_vertex(goal_tuple)
                g.add_edge(new_tuple, goal_tuple)
                # set self.g
                return g.get_shortest_path(tuple(start_position), goal_tuple), {}
            
        # failed to find a path in desired # of iterations
        return [], {}

    def display_plan(self, ax):
        self.g.plot_graph(ax, color="black")
