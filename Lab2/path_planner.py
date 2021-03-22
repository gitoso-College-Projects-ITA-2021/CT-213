from grid import Node, NodeGrid
from math import inf
import time
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
		# [Done?] Todo: implement the Dijkstra algorithm
		# The first return is the path as sequence of tuples (as returned by the method construct_path())
		# The second return is the cost of the path
        self.node_grid.reset()

        # Search Algorithm
        pq = []
        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        start_node.f = 0
        heapq.heappush(pq, (start_node.f, start_position))
        while len(pq) > 0:
            f, node_pos = heapq.heappop(pq)
            if node_pos == goal_position:
                break
            node = self.node_grid.get_node(node_pos[0], node_pos[1])
            for successor in self.node_grid.get_successors(node_pos[0], node_pos[1]):
                successor_node = self.node_grid.get_node(successor[0], successor[1])
                if successor_node.f > node.f + self.cost_map.get_edge_cost(node_pos, successor):
                    successor_node.f = node.f + self.cost_map.get_edge_cost(node_pos, successor)
                    successor_node.parent = node
                    heapq.heappush(pq, (successor_node.f, successor))
        
        # Build path and calculate total cost
        path = []
        total_cost = 0
        prev_node = self.node_grid.get_node(goal_position[0], goal_position[1])
        node = prev_node.parent
        path.append(prev_node.get_position())
        while node != start_node:
            path.append(node.get_position())
            total_cost += self.cost_map.get_edge_cost(prev_node.get_position(), node.get_position())
            aux = node
            node = node.parent
            prev_node = aux
        path.append(node.get_position())
        total_cost += self.cost_map.get_edge_cost(prev_node.get_position(), node.get_position())
        return path, total_cost

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
		# [Done?] Todo: implement the Greedy Search algorithm
		# The first return is the path as sequence of tuples (as returned by the method construct_path())
		# The second return is the cost of the path
        self.node_grid.reset()

        # Search Algorithm
        pq = []
        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        start_node.f = start_node.distance_to(goal_position[0], goal_position[1])
        heapq.heappush(pq, (start_node.f, start_position))
        while len(pq) > 0:
            f, node_pos = heapq.heappop(pq)
            node = self.node_grid.get_node(node_pos[0], node_pos[1])
            node.closed = True
            for successor_pos in self.node_grid.get_successors(node_pos[0], node_pos[1]):
                successor_node = self.node_grid.get_node(successor_pos[0], successor_pos[1])
                if not successor_node.closed:
                    successor_node.closed = True
                    successor_node.parent = node
                    if successor_pos == goal_position:
                        # Build path and calculate total cost
                        path = []
                        total_cost = 0
                        prev_node = self.node_grid.get_node(goal_position[0], goal_position[1])
                        node = prev_node.parent
                        path.append(prev_node.get_position())
                        while node != start_node:
                            path.append(node.get_position())
                            total_cost += self.cost_map.get_edge_cost(prev_node.get_position(), node.get_position())
                            aux = node
                            node = node.parent
                            prev_node = aux
                        path.append(node.get_position())
                        total_cost += self.cost_map.get_edge_cost(prev_node.get_position(), node.get_position())
                        return path, total_cost

                    successor_node.f = successor_node.distance_to(goal_position[0], goal_position[1])
                    heapq.heappush(pq, (successor_node.f, successor_pos))

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
		# Todo: implement the A* algorithm
		# The first return is the path as sequence of tuples (as returned by the method construct_path())
		# The second return is the cost of the path
        self.node_grid.reset()

        # Search algorithm
        pq = []
        start_node = self.node_grid.get_node(start_position[0], start_position[1])
        start_node.g = 0
        start_node.f = start_node.distance_to(goal_position[0], goal_position[1])
        heapq.heappush(pq, (start_node.f, start_position))
        while len(pq) > 0:
            f, node_pos = heapq.heappop(pq)
            node = self.node_grid.get_node(node_pos[0], node_pos[1])
            node.closed = True
            if node_pos == goal_position:
                # Build path and calculate total cost
                path = []
                total_cost = 0
                prev_node = self.node_grid.get_node(goal_position[0], goal_position[1])
                node = prev_node.parent
                path.append(prev_node.get_position())
                while node != start_node:
                    path.append(node.get_position())
                    total_cost += self.cost_map.get_edge_cost(prev_node.get_position(), node.get_position())
                    aux = node
                    node = node.parent
                    prev_node = aux
                path.append(node.get_position())
                total_cost += self.cost_map.get_edge_cost(prev_node.get_position(), node.get_position())
                return path, total_cost
            for successor_pos in self.node_grid.get_successors(node_pos[0], node_pos[1]):
                successor_node = self.node_grid.get_node(successor_pos[0], successor_pos[1])
                if not successor_node.closed:
                    if successor_node.f > node.g + self.cost_map.get_edge_cost(node_pos, successor_pos) + successor_node.distance_to(goal_position[0], goal_position[1]):
                        successor_node.g = node.g + self.cost_map.get_edge_cost(node_pos, successor_pos)
                        successor_node.f = successor_node.g + successor_node.distance_to(goal_position[0], goal_position[1])
                        successor_node.parent = node
                        heapq.heappush(pq, (successor_node.f, successor_pos))
