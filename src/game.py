import numpy as np
import matplotlib.pyplot as plt


class Game() :
    '''
    A class describing a board in grid world MDP
    '''
    def __init__(self, dim, start, goal, action_cost):
        self.dim = dim
        self.start = start
        self.goal = goal
        self.action_cost = action_cost
        self.goal_reward = self.get_manhattan_dist(self.start, self.goal)

    def calculate_value_function(self):
        '''
        Calculates a state-value function for every state in
        the grid world.
        :return: a state-value function for each state
        '''
        val_func = np.zeros(self.dim*self.dim)
        visited = np.zeros(self.dim*self.dim)
        priority_queue = []

        # put goal into the queue
        goal_ind = self.convert_coordinates_to_index(self.goal)
        priority_queue.append(goal_ind)
        visited[goal_ind] = 1
        val_func[goal_ind] = self.goal_reward

        while len(priority_queue) > 0:
            current = priority_queue.pop(0)
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if visited[neighbor] == 0:
                    val_func[neighbor] = val_func[current] + self.action_cost
                    visited[neighbor] = 1
                    priority_queue.append(neighbor)
        return val_func

    def convert_index_to_coordinates(self, index):
        '''
        Convertes the given index into the grid coordinates
        :param index: index of the field in range [0, dim * dim)
        '''
        x = index % self.dim
        y = int(np.floor(index / self.dim))

        return x, y

    def convert_coordinates_to_index(self, coordinates):
        '''
        Converted the given coordinates tuple into an index in range [0, dim*dim}
        :param coordinates: tuple of integers
        :return: index
        '''
        i = coordinates[0]
        j = coordinates[1]

        return j * self.dim + i

    def get_neighbors(self, field):
        '''
        Obtains a list of fields adjacent to the passed field
        :param field: index of field
        :return: list of fields adjacent to field
        '''
        possible_neighbors = [field - self.dim, field + self.dim]
        left = field - 1
        right = field + 1
        if right % self.dim != 0:
            possible_neighbors.append(right)
        if field % self.dim != 0:
            possible_neighbors.append(left)

        neighbors = filter(lambda f: self.is_valid_field(f), possible_neighbors)
        return neighbors

    def is_valid_field(self, field):
        '''
        Checks whether the provided field belongs to the board
        :param field: index of field
        :return: boolean indicating whether field belongs to the board
        '''
        if 0 <= field < self.dim * self.dim:
            return True
        else:
            return False

    def get_manhattan_dist(self, start, goal):
        '''
        Calculates the manhattan distance between the passed tuples
        :param start: (x, y) tuple of integers
        :param goal: (x, y) tuple of integers
        :return: a distance between start and goal
        '''
        x_dist = np.abs(start[0] - goal[0])
        y_dist = np.abs(start[1] - goal[1])
        manh_dist = x_dist + y_dist
        return manh_dist




