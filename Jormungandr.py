from math import sqrt
from queue import PriorityQueue

def manhattan_distance(v1, v2):
    """
    Computes manhattan distance
    Since there is no cost for traveling from node to node, we ignore it ( it's like if the cost was equal to zero).

    return dist: The manhattan distance between v1 and v2
    """
    dist = sum([abs(a - b) for a, b in zip(v1, v2)])
    return dist

def euclidean_distance(v1, v2):
    """
    Computes euclidean distance
    Since there is no cost for traveling from node to node, we ignore it ( it's like if the cost was equal to zero).

    return dist: The euclidean distance between v1 and v2
    """
    dist = sqrt(sum([(a - b)**2 for a, b in zip(v1, v2)]))
    return dist

class Jormungandr():
    """
    Why this name?  --> I could have called it Ouroboros but Jormungandr is a cooler name (see: https://en.wikipedia.org/wiki/J%C3%B6rmungandr)
    """

    def __init__(self, frame_size_x: int, frame_size_y: int, heuristic: str):
        self.MAX_X = frame_size_x
        self.MAX_Y = frame_size_y
        
        # Manhattan distance
        if heuristic == 'manhattan':
            self.heuristic_value = manhattan_distance
        
        # Euclidean distance
        elif heuristic == 'euclidean':
            self.heuristic_value = euclidean_distance

        else:
            raise ValueError('Unknown heuristic, avaliable values: manhattan, euclidean (dont use commas if you are on the command line')


    def can_expand(self, node: tuple, obstacles: dict, open: dict, closed: dict):
        """
        Checks if the node can actually be generated.

        Args:
            node (tuple): The current node we are at.
            obstacles (dict): Positions which we can not walk on (the snake's body).
            open (dict): Open nodes.
            closed (dict): Closed nodes.

        return:
            True if the node can be generated
            False in other case.
        """
        # First, we check if node it's not out of bounds.
        if node [0] [0] < 0 or node [0] [0] > self.MAX_X - 10 or node [0] [1] < 0 or node [0] [1] > self.MAX_Y - 10:
            return False

        try:

            #If node is a closed node it should not be generated.
            closed [node [0] ]
            return False

        except:

            try:

                #If node has already been generated (is an open node) it should not be generated.
                open [node [0] ]
                return False

            except:

                try:
                    # Here, we check if the number of moves we have to make in order to get to the obstacle is higher than
                    # the moves in which the obstacle will no longer be there.
                    if obstacles [ node [0] ] < node [1]:
                        return True

                    # The obstacle will still be there when we get there.
                    return False
                        
                except:
                    # Node it's neither a closed or open node nor an obstacle that will not be clear when we get there
                    # so, it can be generated.
                    return True


    def expand(self, node: tuple, obstacles: dict, open: dict, closed: dict, target: tuple):    
        """
        Expands the given node.
        
        Args:
            node (tuple): The current node we are at.
            obstacles (dict): Positions which we can not walk on (the snake's body).
            open (dict): Open nodes.
            closed (dict): Closed nodes.
            target (tuple): Target we want to reach (the food).

        Returns:
            List with new nodes to explore.
        """
        # A node is a tuple ( (X,Y), N) where X, Y are the coordinates and N is the number of moves the snake has to take to get there
        result = []

        # Node above.
        next_node = ( ( node [0] [0], node [0] [1] - 10) , node [1] + 1)

        if self.can_expand(next_node, obstacles, open, closed):
            result.append((self.heuristic_value (next_node [0], target), (1, node, next_node)))

        # Node below.
        next_node = ( ( node [0] [0], node [0] [1] + 10 ), node [1] + 1)

        if self.can_expand(next_node, obstacles, open, closed):
            result.append((self.heuristic_value (next_node [0], target), (2, node, next_node)))
            
        # Node to the left
        next_node = ( ( node [0] [0] - 10, node [0] [1] ), node [1] + 1 )

        if self.can_expand(next_node, obstacles, open, closed):
            result.append((self.heuristic_value (next_node [0], target), (3, node, next_node)))
            
        # Node to the right
        next_node = ( ( node [0] [0] + 10, node [0] [1] ), node [1] + 1 )

        if self.can_expand(next_node, obstacles, open, closed):
            result.append((self.heuristic_value (next_node [0], target), (4, node, next_node)))
            
        return result

    def backtrack (self, paths: dict, target: tuple) -> list:
        """
        Backtracks and obtains the path which we have to take
        Args:
        paths (dict): Dictionary with all posible nodes and paths.
        target (tuple): Target we want to reach (the food).

        return: List with a move sequence (the path)
        """
        path = []
        current = target
        while (True):

            # Repeat until we can't go backwards
            try:
                aux_value =  paths [ current ]
                path.insert(0, aux_value [0])
                current = aux_value [1]
            except:
                return path

    def pathfind(self, snake_body: list, food_pos: list):
        """
        A* algorithm

        Args:
            snake_body (list): List with snake's body, considered obstacle nodes.
            food_pos (list): Position of the food, considered the goal that we have to reach.

        return: List with the action sequence
        """
        # paths is a dictionary that contains the possible paths to food_pos.
        # Key: A node in format (X, Y), Value: A tuple (M, (Xf , Yf)) in which the second value is father node and M is the move we have to do
        # in order to get from the father node to the child node.

        paths = {}

        #Preprocessing
        snake_head = ( tuple( snake_body [0] ) , 0)
        snake_body = [ tuple (x) for x in snake_body [1:] ]
        obstacles = dict.fromkeys(snake_body)
        food_pos = tuple (food_pos)

        position = len (snake_body)
        for key in obstacles:
            obstacles [key] = position
            position -=1

        open_nodes = PriorityQueue()
        open_nodes_dict = {snake_head [0]: None}
        closed_nodes = {}
        current = snake_head

        # It is set to True when food_pos is reached.
        reached = False
        while not reached:

            # Expand, get possible nodes
            possible_nodes = self.expand(current, obstacles, open_nodes_dict, closed_nodes, food_pos)

            # Current is a closed node, remove also from open nodes
            try: 
                del open_nodes_dict [ current [0] ]
            except:
                pass
            closed_nodes [ current [0] ] = None

            # Put them in queue and in the open nodes list
            for node in possible_nodes:
                open_nodes.put(node)
                open_nodes_dict [ node [1] [2] [0] ] = None

            # Get node, and add to path dictionary, also, now it's a closed node
            current = open_nodes.get() [1]
            paths [ current [2] [0] ] = (current [0], current [1] [0] )

            current = current [2]

            # If we reached the food
            if current [0] [0] == food_pos [0] and current [0] [1] == food_pos [1]:
                reached = True

            if open_nodes.empty():
                print("No way!")

                # By default go up, but there's no possible path to the food
                return [1]

        # Backtrack to get the real path to food_pos.
        path = self.backtrack(paths, food_pos)
        return path