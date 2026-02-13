import rustworkx as rx
import rustworkx.visualization
import numpy as np

class Graph:
    def __init__(self, directed=False, weightfn=None):
        if not directed:
            self.graph = rx.PyGraph()
        else:
            self.graph = rx.PyDiGraph()
        self.directed = directed
        self.vertices = {} # 3d point to vertex id
        
        if weightfn is None:
            # euclidean distance. 
            self.weightfn = lambda x, y: np.linalg.norm(np.array(x) - np.array(y))
        else:
            self.weightfn = weightfn

    # PROPERTIES
    @property
    def connected_components(self):
        return self.num_connected_components()
    def num_connected_components(self):
        return rx.connected_components(self.graph)
    
    '''All of the below return a COPY of the list of nodes in the graph.'''

    def get_nodes(self):
        return self.graph.nodes() # gets the list of DATA of nodes
    def get_vertices(self):
        return self.get_nodes()
    
    def get_edges(self):
        return [(self.graph.get_node_data(s), self.graph.get_node_data(t)) \
                            for s, t in self.graph.edge_list()]

    # BASIC OPERATIONS
    def reset(self):
        self.graph = rx.PyGraph()
        self.vertices = {}

    def copy(self):
        new_graph = Graph(directed=self.directed, weightfn=self.weightfn)
        new_graph.graph = self.graph.copy()
        new_graph.vertices = self.vertices.copy()
        return new_graph
    
    # ADDERS AND REMOVERS

    '''
    Input: v: object to add to the graph (not necessarily coordinates)
    Output: The vertex ID of the vertex. 

    This function silently rejects duplicate vertices. 
    '''
    def add_vertex(self, v):
        if v in self.vertices:
            return self.vertices[v]
        vid = self.graph.add_node(v)
        self.vertices[v] = vid
        return vid
    
    '''
    Input: vertices: list of objects to add to the graph (not necessarily coordinates)
    Output: None

    This function silently rejects duplicate vertices.
    '''
    def add_many_vertices(self, vertices):
        for v in vertices:
            self.add_vertex(v)

    '''
    Input: v: object to remove from the graph
    Output: None

    This function removes the vertex and all edges connected to it.
    Will throw an error if v is not in the graph. 
    '''
    def remove_vertex(self, v):
        vid = self.vertices[v]
        neighbors = self.graph.neighbors(vid)
        for n in neighbors:
            self.graph.remove_edge(vid, n)
        self.graph.remove_node(vid)
        del self.vertices[v]
    
    '''
    Input: c1, c2: objects that are already in the graph (NOT VIDs)
           edge_weight: weight of the edge. If None, will use the weight function.
    Output: None
    '''
    def add_edge(self, c1, c2, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.weightfn(c1, c2)
        self.graph.add_edge(self.vertices[c1], self.vertices[c2], edge_weight)

    '''
    Input: c1, c2: objects that are already in the graph (NOT VIDs)
    Output: None

    Remove edge (c1, c2) from the graph. c1 and c2 must be in the graph. 
    This function does not remove vertices c1 and c2. 
    '''
    def remove_edge(self, c1, c2):
        self.graph.remove_edge(self.vertices[c1], self.vertices[c2])


    # ALGORITHMS

    '''
    Input: start, goal: objects that are already in the graph, start and end for shortest path. 
    Output: List of objects that are the shortest path from start to goal. (NOT VIDs)
    '''
    def get_shortest_path(self, start, goal):
        source = self.vertices[start]
        target = self.vertices[goal]
        if self.directed:
            path = rustworkx.digraph_dijkstra_shortest_paths(graph=self.graph, 
                                                    source=source, 
                                                    target=target, 
                                                    weight_fn=lambda x:x)[target]
        else:
            path = rustworkx.graph_dijkstra_shortest_paths(graph=self.graph, 
                                                    source=source, 
                                                    target=target, 
                                                    weight_fn=lambda x:x)[target]
        return [self.graph.get_node_data(node) for node in path]

    '''
    Input: point: object in graph. 
           num: number of nearest points to find (NOT NECESSARILY NEIGHBORS). 
           radius: maximum distance to consider. If -1, will consider all points.
    Output: List of objects that are the num nearest points to point. (NOT VIDs)

    Find the nearest num points to this one, that is at MOST radius dist away.
    This function will not include `point` if it is in the graph. 
    '''
    def find_nearest_at_most(self, point, num, max_dist=-1):
        # Get list of nodes
        nodes = self.graph.nodes()

        # We do not want cfg to show up in our list of neighbors. 
        try:
            nodes.remove(point)
        except:
            pass

        # Create list of cfgs and distances
        pt_dist = [(v, self.weightfn(point, v)) for v in nodes]

        if max_dist != -1:
            # Filter by distance
            pt_dist = list(filter(lambda w : w[1] <= max_dist, pt_dist))

        # Sort list by distances
        pt_dist.sort(key=lambda w: w[1])

        # Return the num nearest neighbors
        return [v[0] for v in pt_dist[0:num]]
    
    '''
    Input: point: object in graph.
              num_neighbors: number of nearest points to find (NOT NECESSARILY NEIGHBORS).
              min_dist: minimum distance to consider. If 0, will consider all points.
    Output: List of objects that are the num nearest points to point. (NOT VIDs)

    Find the nearest num points to this one, that is at LEAST min_dist away.
    This function will not include `point` if it is in the graph.
    '''
    def find_nearest_at_least(self, point, num_neighbors, min_dist=0):
        # Get list of nodes
        nodes = self.graph.nodes()

        # We do not want cfg to show up in our list of neighbors. 
        try:
            nodes.remove(point)
        except:
            pass

        # Create list of cfgs and distances
        neighbors = [(v, self.weightfn(point, v)) for v in nodes]

        # Filter by distance
        neighbors = list(filter(lambda w : w[1] >= min_dist, neighbors))

        # Sort list by distances
        neighbors.sort(key=lambda w: w[1])

        # Return the num_neighbors nearest neighbors
        return [v[0] for v in neighbors[0:num_neighbors]]
    
    # UNTESTED, but if works can remove the other 2 confusing functions
    '''
    Input: point: object in graph.
           num: number of nearest points to find (NOT NECESSARILY NEIGHBORS).
           min_dist: minimum distance to consider. Default is 0.
           max_dist: maximum distance to consider. Default is -1, meaning no maximum.
    Output: List of objects. (NOT VIDs)

    Find the nearest num points to this one, that is at LEAST min_dist away and at MOST max_dist away.
    This function will not include `point` if it is in the graph.
    '''
    def find_nearest(self, point, num, min_dist=0, max_dist=-1):
        nodes = self.get_nodes()
        assert len(nodes) > 0, "Graph is empty, cannot find nearest neighbors."

        # We do not want `point` in our list of nearest
        try:
            nodes.remove(point)
        except:
            pass

        # Create list of nodes and distances
        pt_dist = [(v, self.weightfn(point, v)) for v in nodes]

        # Filter by distance 
        pt_dist = list(filter(lambda w : w[1] >= min_dist, pt_dist))
        if max_dist != -1:
            pt_dist = list(filter(lambda w : w[1] <= max_dist, pt_dist))

        # Sort list by distances (in increasing order).
        pt_dist.sort(key=lambda w: w[1])

        # Return the num nearest neighbors
        return [v[0] for v in pt_dist[0:num]]

    '''
    Input: point: object in graph.
    Output: Distance to the nearest point in the graph.
    '''
    def dist_to_nearest_vertex(self, point):
        # return min([self.weightfn(point, v) for v in self.graph.nodes()])
        nearest = self.find_nearest(point, 1)
        if len(nearest) == 0:
            return np.inf
        return self.weightfn(point, nearest[0])
    
    

    # ============================== PLOTTING ============================== #
    def plot_graph(self, ax, color="blue"):
        pos = {self.vertices[node]: (node[0], node[1]) for node in self.graph.nodes()}
        rx.visualization.mpl_draw(self.graph, 
                                pos=pos, 
                                ax=ax, 
                                node_size=5, 
                                node_color='black', 
                                width=1, 
                                edge_color=color)

    def plot_subgraph(self, graph, ax, color="green"):
        pos = {self.vertices[node]: (node[0], node[1]) for node in graph.nodes()}
        rx.visualization.mpl_draw(graph, 
                                pos=pos, 
                                ax=ax, 
                                node_size=5, 
                                node_color=color, 
                                width=1, 
                                edge_color=color)

