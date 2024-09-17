import random
from collections import defaultdict
from utils import manhattan_distance

import pygame

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)

# Action #0 -> Move Front
# Action #1 -> Move Left
# Action #2 -> Move Right

MAX_STEPS = 100000
MOVE_DICT = {
    'N': {
        0: [0, -1],
        1: [-1, 0],
        2: [1, 0],
    },

    'S': {
        0: [0, 1],
        1: [1, 0],
        2: [-1, 0],
    },

    'W': {
        0: [-1, 0],
        1: [0, 1],
        2: [0, -1],
    },

    'E': {
        0: [1, 0],
        1: [0, -1],
        2: [0, 1],
    }
}

DIRECTION_DICT = {
    'N': {
        0: 'N',
        1: 'W',
        2: 'E'
    },

    'E': {
        0: 'E',
        1: 'N',
        2: 'S'
    },

    'S': {
        0: 'S',
        1: 'E',
        2: 'W'
    },

    'W': {
        0: 'W',
        1: 'S',
        2: 'N'
    },
}

class SnakeGame():

    def __init__(self, game_window, width = 5, height = 5, display: bool = False):

        if width < 5 or height < 5:
            raise ValueError("Both height and width must be higher than 5")
        
        # Game window for visualization
        self.game_window = game_window

        # Max width of the board
        self.width = width

        # Max height of the board
        self.height = height
        
        # Whether to display the game in PyGame
        self.display = display

        # Snake starts at the middle of the board
        self.x = width // 2
        self.y = height // 2

        # Variable to control the resets
        self.food_before = False

        # Create the snake body, first the head, and the tail, by the default, the snake
        # starts with a body of length 3 growing towards south.
        self.snake_body = [(self.x, self.y), (self.x, self.y + 1), (self.x, self.y + 2)]

        # Spawn food and get the direction to it
        self.spawn_food()
        food_north, food_south, food_west, food_east = self.get_food_direction()
        
        self.food_eaten = 0
        self.deaths = 0
        self.steps = 0

        # Initialize state
        self.state = {
            'DangerFront': 1 if self.y + 1 > self.height else 0,
            'DangerLeft': 1 if self.x - 1 > self.width else 0,
            'DangerRight': 1 if self.x + 1 > self.width else 0,
            'GoingNorth': 1, # By default the snake starts by facing north
            'GoingSouth': 0,
            'GoingWest': 0,
            'GoingEast': 0,
            'FoodNorth': food_north,
            'FoodSouth': food_south,
            'FoodWest': food_west,
            'FoodEast': food_east,
        }

        # Internal variable to help V function computing
        self.current_direction = 'N'
        
        return

    
    def spawn_food(self):
        """
        Spawns food on the board.
        """
        self.food_x, self.food_y = random.randint(0, self.width), random.randint(0, self.height)

        while (self.food_x, self.food_y) in self.snake_body:
            self.food_x, self.food_y = random.randint(0, self.width), random.randint(0, self.height)

        return
    
    def check_danger(self, going_north, going_south, going_west):
        """
        Checks for new dangers, we look if the snake will get out of the board or run 
        into itself
        """

        # To avoid KeyError exceptions, we use a defaultdict with default value bool (False)
        # this is a little programming trick to optimize the if selection flow.
        snake_body =  defaultdict(bool,{tuple(key):True for key in self.snake_body})

        if going_north:
            danger_front = 1 if self.y - 1 < 0 or snake_body[(self.x, self.y - 1)] else 0
            danger_left = 1 if self.x - 1 < 0 or snake_body[(self.x - 1, self.y)] else 0
            danger_right = 1 if self.x + 1 > self.width or snake_body[(self.x + 1, self.y)] else 0

        # If the snake faces south
        elif going_south:
            danger_front = 1 if self.y + 1 > self.height or snake_body[(self.x, self.y + 1)] else 0
            danger_left = 1 if self.x + 1 > self.width or snake_body[(self.x + 1, self.y)] else 0
            danger_right = 1 if self.x - 1 < 0 or snake_body[(self.x - 1, self.y)] else 0

        # If the snake faces west
        elif going_west:
            danger_front = 1 if self.x - 1 < 0 or snake_body[(self.x - 1, self.y)] else 0
            danger_left = 1 if self.y + 1 > self.height or snake_body[(self.x, self.y + 1)] else 0
            danger_right = 1 if self.y - 1 < 0 or snake_body[(self.x, self.y - 1)] else 0 

        # If the snake faces east
        else:
            danger_front = 1 if self.x + 1 > self.width or snake_body[(self.x + 1, self.y)] else 0
            danger_left = 1 if self.y - 1 < 0 or snake_body[(self.x, self.y - 1)] else 0 
            danger_right = 1 if self.y + 1 > self.height or snake_body[(self.x, self.y + 1)] else 0 

        return danger_front, danger_left, danger_right
    
    def check_death(self) -> bool:
        """
        Checks if the head of the snake is in a valid position, returns True
        if the head is in a valid position, false otherwise

        Returns:
            bool
        """
        # Check if the head is inside the board
        if not (self.x <= self.width and self.x >= 0 and self.y <= self.height and self.y >= 0):
            return False
    
        # Check if the head is touching the snake body, if inside snake_body there are
        # two equals positions the dictionary will not have the same length and the list, 
        # which means that the snake has touched itself.
        if len(self.snake_body) != len(set([tuple(elem) for elem in self.snake_body])):
            return False
        
        return True
    
    def move(self, action):
        """
        Moves the snake given the action.
        """
        # If the snake faces north
        if self.state['GoingNorth']:

            # Get move increment
            incr_x, incr_y = MOVE_DICT['N'][action]

        # If the snake faces south
        elif self.state['GoingSouth']:

            # Get move increment
            incr_x, incr_y = MOVE_DICT['S'][action]

        # If the snake faces west
        elif self.state['GoingWest']:

            # Get move increment
            incr_x, incr_y = MOVE_DICT['W'][action]

        # If the snake faces east
        else:

            # Get move increment
            incr_x, incr_y = MOVE_DICT['E'][action]

        self.x += incr_x
        self.y += incr_y

        # Insert the new head, but don't delete the tail position yet
        self.snake_body.insert(0, [self.x, self.y])
        
        return 
    
    def get_direction(self, action):
        """
        Returns the new direction the snake is going.
        """

        going_north, going_south, going_west, going_east = 0, 0, 0, 0

        # If the snake faces north
        if self.state['GoingNorth']:
            direction = DIRECTION_DICT['N'][action]

        # If the snake faces south
        elif self.state['GoingSouth']:
            direction = DIRECTION_DICT['S'][action]

        # If the snake faces west
        elif self.state['GoingWest']:
            direction = DIRECTION_DICT['W'][action]

        # If the snake faces east
        else:
            direction = DIRECTION_DICT['E'][action]
        
        # Update internal variable
        self.current_direction = direction

        if direction == 'N':
            going_north = 1
        
        elif direction == 'S':
            going_south = 1
        
        elif direction == 'W':
            going_west = 1

        else:
            going_east = 1

        return going_north, going_south, going_west, going_east
    
    def get_food_direction(self):
        """
        Returns the direction of the food. That is, North, South, West or East.
        """

        food_north, food_south, food_west, food_east = 0, 0, 0, 0

        # We do not check the case where food_x == x since in this case the food is neither west nor east,
        # same for y axis.
        if self.food_x < self.x:
            food_west = 1
        
        elif self.food_x > self.x:
            food_east = 1
        
        if self.food_y > self.y:
            food_south = 1

        elif self.food_y < self.y:
            food_north = 1

        return food_north, food_south, food_west, food_east
    
    def step(self, action):
        """
        Performs one step in the environment

        Args:
            action (str): The action to take
        
        Returns:
            terminal (bool): Whether the state is a terminal state or not
            reward (int): Reward for step
            new_state (dict): The new state
        """

        # Move the snake
        self.move(action)

        # Check the new direction the snake is heading
        going_north, going_south, going_west, going_east = self.get_direction(action)

        # Now, check for danger and the new direction of the food
        danger_front, danger_left, danger_right = self.check_danger(going_north, going_south, going_west)
        food_north, food_south, food_west, food_east = self.get_food_direction()
        

        # Create new state
        next_state = {
            'DangerFront': danger_front,
            'DangerLeft': danger_left,
            'DangerRight': danger_right,
            'GoingNorth': going_north,
            'GoingSouth': going_south,
            'GoingWest': going_west,
            'GoingEast': going_east,
            'FoodNorth': food_north,
            'FoodSouth': food_south,
            'FoodWest': food_west,
            'FoodEast': food_east,
        }
        
        self.steps += 1

        if self.display:
            # Draw the next frame
            self.refresh_screen()

        # Check if we are not dead
        if not self.check_death():
            self.deaths += 1
            return True, -10, next_state
        
        # Check if we can eat food
        if self.x == self.food_x and self.y == self.food_y:
            self.food_eaten += 1
            self.food_before = True
            return True, +10, next_state
        
        if self.steps == MAX_STEPS:
            return True, -1, next_state
        
        # The last position of the list no longer exists, since the snake moves
        self.snake_body.pop()

        #return False, -0.01 * manhattan_distance((self.x, self.y), (self.food_x, self.food_y)), next_state
        return False, -1, next_state
    
    def reset(self):
        """
        Resets the game a creates a clean new instance (episode)
        """ 

        # In the case we have eaten food in the previous episode, the state and the body
        # is left the same.
        if not self.food_before:

            # Snake starts at the middle of the board
            self.x = self.width // 2
            self.y = self.height // 2

            # Spawn food and get the direction to it
            self.spawn_food()
            food_north, food_south, food_west, food_east = self.get_food_direction()
        
            # Initialize state
            self.state = {
                'DangerFront': 1 if self.y + 1 > self.height else 0,
                'DangerLeft': 1 if self.x - 1 > self.width else 0,
                'DangerRight': 1 if self.x + 1 > self.width else 0,
                'GoingNorth': 1, # By default the snake starts by facing north
                'GoingSouth': 0,
                'GoingWest': 0,
                'GoingEast': 0,
                'FoodNorth': food_north,
                'FoodSouth': food_south,
                'FoodWest': food_west,
                'FoodEast': food_east,
            }

            self.snake_body = [(self.x, self.y), (self.x, self.y + 1), (self.x, self.y + 2)]
            
        else:

            # Spawn food and get the direction to it
            self.spawn_food()
            food_north, food_south, food_west, food_east = self.get_food_direction()

            self.state['FoodNorth'] = food_north
            self.state['FoodSouth'] = food_south
            self.state['FoodWest'] = food_west
            self.state['FoodEast'] = food_east

            self.food_before = False
        # Reset number of episode steps
        self.steps = 0

        return
    
    def show_scores(self):

        score_font = pygame.font.SysFont('times new roman', 17)
        food_eaten = score_font.render('Food eaten : ' + str(self.food_eaten), True, GREEN)
        deaths = score_font.render('Deaths : ' + str(self.deaths), True, RED)

        food_eaten_rect = food_eaten.get_rect()
        deaths_rect = deaths.get_rect()

        food_eaten_rect.midtop = (self.width / 10, 15)
        deaths_rect.midtop = (self.width / 10, 35)

        self.game_window.blit(food_eaten, food_eaten_rect)
        self.game_window.blit(deaths, deaths_rect)

        return
    
    def debug_info(self):

        debug_font = pygame.font.SysFont('times new roman', 10)

        danger_front = debug_font.render('Danger front', True, GREEN if self.state['DangerFront'] else RED)
        danger_left = debug_font.render('Danger left', True, GREEN if self.state['DangerLeft'] else RED)
        danger_right = debug_font.render('Danger right', True, GREEN if self.state['DangerRight'] else RED)

        food_north = debug_font.render('Food north', True, GREEN if self.state['FoodNorth'] else RED)
        food_south = debug_font.render('Food south', True, GREEN if self.state['FoodSouth'] else RED)
        food_west = debug_font.render('Food west', True, GREEN if self.state['FoodWest'] else RED)
        food_east = debug_font.render('Food east', True, GREEN if self.state['FoodEast'] else RED)

        going_north = debug_font.render('Going north', True, GREEN if self.state['GoingNorth'] else RED)
        going_south = debug_font.render('Going south', True, GREEN if self.state['GoingSouth'] else RED)
        going_west = debug_font.render('Going west', True, GREEN if self.state['GoingWest'] else RED)
        going_east = debug_font.render('Going east', True, GREEN if self.state['GoingEast'] else RED)

        food_x = debug_font.render('Food X: ' + str(self.food_x), True, BLUE)
        food_y = debug_font.render('Food Y: ' + str(self.food_y), True, BLUE)

        x = debug_font.render('X: ' + str(self.x), True, BLUE)
        y = debug_font.render('Y: ' + str(self.y), True, BLUE)

        danger_front_rect = danger_front.get_rect()
        danger_left_rect = danger_left.get_rect()
        danger_right_rect = danger_right.get_rect()

        food_north_rect = food_north.get_rect()
        food_south_rect = food_south.get_rect()
        food_west_rect = food_west.get_rect()
        food_east_rect = food_east.get_rect()

        going_north_rect = going_north.get_rect()
        going_south_rect = going_south.get_rect()
        going_west_rect = going_west.get_rect()
        going_east_rect = going_east.get_rect()

        food_x_rect = food_x.get_rect()
        food_y_rect = food_y.get_rect()
        x_rect = x.get_rect()
        y_rect = y.get_rect()

        food_north_rect.midtop = (self.width - 35, 15)
        food_south_rect.midtop = (self.width - 35, 30)
        food_west_rect.midtop = (self.width - 35, 45)
        food_east_rect.midtop = (self.width - 35, 60)

        going_north_rect.midtop = (self.width - 35, 75)
        going_south_rect.midtop = (self.width - 35, 90)
        going_west_rect.midtop = (self.width - 35, 105)
        going_east_rect.midtop = (self.width - 35, 120)

        food_x_rect.midtop = (self.width - 35, 135)
        food_y_rect.midtop = (self.width - 35, 150)
        x_rect.midtop = (self.width - 35, 165)
        y_rect.midtop = (self.width - 35, 180)

        danger_front_rect.midtop = (self.width - 35, 195)
        danger_left_rect.midtop = (self.width - 35, 210)
        danger_right_rect.midtop = (self.width - 35, 225)

        self.game_window.blit(danger_front, danger_front_rect)
        self.game_window.blit(danger_left, danger_left_rect)
        self.game_window.blit(danger_right, danger_right_rect)

        self.game_window.blit(food_north, food_north_rect)
        self.game_window.blit(food_south, food_south_rect)
        self.game_window.blit(food_west, food_west_rect)
        self.game_window.blit(food_east, food_east_rect)

        self.game_window.blit(going_north, going_north_rect)
        self.game_window.blit(going_south, going_south_rect)
        self.game_window.blit(going_west, going_west_rect)
        self.game_window.blit(going_east, going_east_rect)

        self.game_window.blit(food_x, food_x_rect)
        self.game_window.blit(food_y, food_y_rect)
        self.game_window.blit(x, x_rect)
        self.game_window.blit(y, y_rect)


        return
    
    def refresh_screen(self):
        """
        Updates (draws) the screen after one environment step
        """

        self.reset_screen()
        self.draw_snake_and_food()
        self.show_scores()
        self.debug_info()

        # Refresh game screen
        pygame.display.update()

        return

    def reset_screen(self):
        """
        Fills the screen window in black
        """
        self.game_window.fill(BLACK)

        return

    def draw_snake_and_food(self):
        """
        Draws the snake body and food on the screen
        """
        for pos in self.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))

        # Snake food
        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(self.food_x, self.food_y, 10, 10))

        return