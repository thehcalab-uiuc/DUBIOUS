import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import json
import numpy as np
import shapely
import scipy as sp

class env:
    '''
    kwargs includes:
    bounds = [xmin, xmax, ymin, ymax]
    observers = [[posx, posy, motive, agent_visibility_polygon_vertices...], ]
    '''
    def __init__(self, **kwargs):
        if "bounds" in kwargs:
            self.bounds = kwargs["bounds"]
        else:
            self.bounds = [-1, 1, -1, 1]  # default bounds: [xmin, xmax, ymin, ymax
        self.x_range = self.bounds[:2]
        self.y_range = self.bounds[2:]

        if "observers" in kwargs:
            observers = kwargs["observers"]
            self.n_observers = len(observers)
            self.observer_positions = [observer[0:2] for observer in observers]
            self.observer_motives = [observer[2] for observer in observers]
            self.observer_polygons = [Polygon(observer[3:]) for observer in observers]
        else:
            self.n_observers = 1
            self.adversary_positions = [[0.5, -0.5], [-0.5, 0.5]]
            self.adversary_polygons = [Polygon([[0.4, -0.4], [0.6, -0.4], [0.6, -0.6], [0.4, -0.6]]), Polygon([[-0.6, 0.4], [-0.4, 0.4], [-0.4, 0.6], [-0.6, 0.6]])]

        if "goals" in kwargs:
            self.goals = kwargs["goals"]
        else:
            self.goals = [[0.8, 0], [0.8, -0.8], [0.8, 0.8]]
        self.goals = np.array(self.goals)

        if "obstacles" in kwargs:
            self.obstacles = kwargs["obstacles"]
        else:
            self.obstacles = [[]]
        self.obstacles = [Polygon(obst) for obst in self.obstacles if len(obst) >=3]

    # given a trajectory, returns a MASK of the trajectory points visible to the observer
    def get_observer_view(self, trajectory, observer_idx):
        traj_points = shapely.points(trajectory)
        mask = np.where(self.observer_polygons[observer_idx].contains(traj_points), True, False)
        return mask

    # Returns the sum of the motives of the observers who can see at each point along the trajectory
    # Returns: (T,1) vector, each number is the sum of the motives at that point
    def sum_obs_view(self, trajectory):
        values = np.zeros(trajectory.shape[0])
        for obs in range(self.n_observers):
            mask = self.get_observer_view(trajectory, obs)
            values[mask] += np.abs(self.observer_motives[obs])
        values = np.where(values == 0, 1, values)
        return values

    # obsolete
    def visibility_cost(self, trajectory, observer_idx, certainty):
        values = np.zeros(trajectory.shape[0])
        traj_points = shapely.points(trajectory)
        mask = np.where(self.observer_polygons[observer_idx].contains(traj_points), 1, -1)
        dist = self.observer_polygons[observer_idx].exterior.distance(traj_points)
        sig_dist = sp.special.expit(dist * mask * certainty)
        values += self.observer_motives[observer_idx] * sig_dist
        return values

    def point_in_collision(self, pt):
        point = Point([pt])
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                return True
        return False
    
    def get_score(self, pt):
        if self.point_in_collision(pt):
            return -float('inf')
        score = 0
        for adv_poly in self.adversary_polygons:
            if adv_poly.contains(Point(pt)):
                score -= 1
        for friend_poly in self.friendly_polygons:
            if friend_poly.contains(Point(pt)):
                score += 1
        return score

    def display(self, ax):
        # Display the environment bounds
        xmin, xmax, ymin, ymax = self.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Turn off axis ticks and stuff, but leave the box
        # ax.set_xticks([])
        # ax.set_yticks([])

        def plot_polygon(poly, color, alpha):
            coords = list(zip(*poly.exterior.xy))
            poly_patch = plt.Polygon(coords, color=color, alpha=alpha)
            ax.add_patch(poly_patch)

        def plot_point(pt, color, marker, size=15):
            ax.plot(pt[0], pt[1], marker=marker, color=color, markersize=size)

        # display the obstacles
        for obstacle in self.obstacles:
            if len(obstacle) >= 3:  # need at least 3 points to form a polygon
                plot_polygon(Polygon(obstacle), color='gray', alpha=1)

        # Display the observers and what they see
        for i in range(self.n_observers):
            pos = self.observer_positions[i]
            polygon = self.observer_polygons[i]

            color = 'darkred' if self.observer_motives[i] < 0 else 'green'
            # plot_point(pos, color=color, marker='x')
            plot_polygon(polygon, color=color, alpha= 0.1 + (0.2 * abs(self.observer_motives[i])))

        # Display the goals
        for goal in self.goals:
            plot_point(goal, color='purple', marker='*', size=15)


    def to_json_dict(self):
        data = {
            "bounds": self.bounds,
            "observers": [self.observer_positions[i] + \
                            [self.observer_motives[i]] + \
                            list(self.observer_polygons[i].exterior.coords) \
                        for i in range(self.n_observers)],
            "goals": self.goals,
            "obstacles": [list(obst.exterior.coords) for obst in self.obstacles]
        }
        return data

    def to_json(self, filename):
        data = self.to_json_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    # DOES NOT AUTOMATICALLY FILL IN DEFAULT PARAMETER VALUES. 
    # but if you use to_json it should be fine ? 
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_json_dict(data)

    @classmethod
    def from_json_dict(cls, data):
        return cls(
            bounds=data["bounds"],
            observers=data["observers"],
            goals=data["goals"],
            obstacles=data["obstacles"]
        )