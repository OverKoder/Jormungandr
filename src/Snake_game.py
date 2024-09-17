import pygame, sys, time, random, argparse
from numpy import cumsum

from Jormungandr import Jormungandr
from utils import game_over
from rl.environments import SnakeGame
from rl.action_selection import EpsilonGreedy
from rl.solvers import SARSA, QLearning, n_step_SARSA, n_step_OffPolicy
from rl.agent import Jormungandr as RL_Jormungandr
from rl.planner import DynaQ, DynaQWithPriority

from tqdm import tqdm
import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser(description = "Start the game!")

# Algorithm and RL are mutually exclusive
group = parser.add_mutually_exclusive_group()

# Display the game on screen
parser.add_argument("-d","--display",dest = 'display', required = False, action='store_true', default = False, help = "Whether to display on PyGame the training / game. Default is False.")

# Train or test
parser.add_argument("-t","--test",dest = 'test', required = False, action='store_true', default = False, help = "Whether to train (False) or test (True). Default is False")

# Heuristic
parser.add_argument("-hh","--heuristic",dest = 'heuristic', type = str, required = False, default = 'manhattan', help = "Which heuristic to use, 1 .- Manhattan Distance. 2.- Euclidean Distance")

# Whether to run using A* or not
group.add_argument("-a","--algorithm",dest = 'algorithm', required = False, action='store_true', default = False, help = "Whether to run A* algorithm or not (mutually exclusive with -r)")

# Whether to use RL
group.add_argument("-r","--reinforcement",dest = 'reinforcement',required = False, action='store_true', default = False,help = "Whether to train with RL or not (mutually exclusive with -a)")

# Checkpoint path to Q function
parser.add_argument("-c","--checkpoint",dest = 'checkpoint', type = str, required = False, default = None, help = "Checkpoint path for Q function (-r only).")

# Learning algorithm
parser.add_argument("-la","--learning_algorithm",dest = 'learning_algorithm', type = str, required = False, default = 'SARSA', help = "Learning algorithm to use (RL only): 'sarsa', 'qlearning', 'nstep'")

# Size of the screen
parser.add_argument("--width",dest = 'width', type = int, required = False, default = 500, help = "Width of the display screen.")
parser.add_argument("--height",dest = 'height', type = int, required = False, default = 500, help = "Height of the display screen.")

# Epsilon for policy
parser.add_argument("-e", "--epsilon",dest = 'epsilon', type = float, required = False, default = 0, help = "Epsilon for epsilon-greedy policy.")

# Number of episodes
parser.add_argument("-ep", "--episodes",dest = 'episodes', type = int, required = False, default = 1000, help = "Number of episodes to run")

# Path to save Q
parser.add_argument("-s", "--save_path",dest = 'save_path', type = str, required = False, default = 'output.npy', help = "Save path for Q function.")

# Steps of nStep Sarsa
parser.add_argument("-ns", "--n_steps",dest = 'n_steps', type = int, required = False, default = 3, help = "Number of steps in nStepSarsa and nStepQLearning.")

# Planning steps
parser.add_argument("-ps", "--planning_steps",dest = 'planning_steps', type = int, required = False, default = 0, help = "Number of planning steps.")

# Type of planner
parser.add_argument("-p", "--planner",dest = 'planner', type = str, required = False, default = 'dynaq', help = "Type of planner, can be either 'DynaQ' or 'DynaQPriority'.")

# Threshold for priority sweeping
parser.add_argument("-th", "--threshold",dest = 'threshold', type = float, required = False, default = 0.1, help = "Threshold for priority sweeping (DynaQ planning).")

args = parser.parse_args()

# Checks for errors encountered
check_errors = pygame.init()

if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


if args.display:
    # Initialise game window
    pygame.display.set_caption('Jormungandr')
    game_window = pygame.display.set_mode((args.width, args.height))

    # FPS (frames per second) controller
    fps_controller = pygame.time.Clock()

    # Refresh rate
    fps_controller.tick(30)

else:
    game_window = None


if args.reinforcement:

    total_rewards = []

    # Initialize environment
    environment = SnakeGame(game_window, width = args.width, height = args.height, display = args.display)

    # Action selector
    action_selector = EpsilonGreedy(args.epsilon, reduce_on_steps = False)

    # Agent
    agent = RL_Jormungandr(alpha = 0.1, action_selector = action_selector, environment = environment, checkpoint = args.checkpoint)

    planner_name = args.planner.lower()
    planner = None

    solver = args.learning_algorithm.lower()


    # Planner
    if args.planning_steps != 0:

        if planner_name == 'dynaq':
            planner = DynaQ(agent = agent, environment = environment, planning_steps = args.planning_steps, learning_algorithm = solver)

        elif planner_name == 'dynaqpriority':
            planner = DynaQWithPriority(agent = agent, environment = environment, planning_steps = args.planning_steps, learning_algorithm = solver, threshold = args.threshold)

    # Solver
    if solver == 'sarsa':
        solver = SARSA(agent, environment, planner = planner, save_path = args.save_path, test = args.test)

    elif solver == 'qlearning':
        solver = QLearning(agent = agent, environment = environment, planner = planner, save_path = args.save_path, test = args.test)

    elif solver == 'nstepsarsa':
        solver = n_step_SARSA(n_steps = args.n_steps, agent = agent, environment = environment, planner = planner, save_path = args.save_path, test = args.test)

    elif solver == 'nstepoffpolicy':
        solver = n_step_OffPolicy(n_steps = args.n_steps, agent = agent, environment = environment, planner = planner, save_path = args.save_path, test = args.test)

    else:
        print("ERROR Solver not found")
        sys.exit(-1)
 
    for episode in tqdm(range(1, args.episodes + 1), desc = 'Training...'):
        
        # Run episode
        solver.run_episode()



    print("Food eaten:", environment.food_eaten)
    print("Deaths:", environment.deaths)

    """plt.plot(list(range(1,MAX_EPISODES)), cumsum(cum_rewards), label = 'Reward sum = ' + str(cumsum(cum_rewards)[-1]) + ' (ε = ' + str(epsilon) + ')')
    plt.title("Evolution of agent learning (Off-policy SARSA)")
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative sum of rewards')
    plt.legend()
    plt.savefig('plots/QLearning/Rewards_QLearning.png')
    plt.clf()"""
if args.algorithm:

    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)
    # Default snake position
    snake_pos = [100, 50]

    # Snake starts with length == 3
    snake_body = [[100, 50], [90, 50], [80, 50]]

    # We generate the first piece of food
    food_pos = [random.randrange(1, (args.width//10)) * 10, random.randrange(1, (args.height//10)) * 10]
    food_spawn = True

    direction = 'RIGHT'

    score = 0


    # Score
    def show_score(choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(score), True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (args.width/10, 15)
        else:
            score_rect.midtop = (args.width/2, args.height/1.25)
        game_window.blit(score_surface, score_rect)
        # pygame.display.flip()

    snake = Jormungandr(args.width, args.height, args.heuristic)
    # Main logic
    while True:
    
        move_sequence = snake.pathfind(snake_body, food_pos)
        for move in move_sequence:

            # Moving the snake

            # Move up
            if move == 1:
                snake_pos[1] -= 10

            # Move down
            if move == 2:
                snake_pos[1] += 10

            # Move left
            if move == 3:
                snake_pos[0] -= 10

            # Move right
            if move == 4:
                snake_pos[0] += 10
            
            # Snake body growing mechanism
            snake_body.insert(0, list(snake_pos))
            if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
                score += 1
                food_spawn = False
            else:
                snake_body.pop()

            # Spawning food on the screen
            while not food_spawn:
                food_pos = [random.randrange(1, (args.width//10)) * 10, random.randrange(1, (args.height//10)) * 10]

                #We check that food does not spawn inside the snake, otherwise it could crash
                if food_pos not in snake_body and food_pos not in snake_pos:
                    food_spawn = True

            

            # GFX
            game_window.fill(black)
            for pos in snake_body:
                # Snake body
                # .draw.rect(play_surface, color, xy-coordinate)
                # xy-coordinate -> .Rect(x, y, size_x, size_y)
                pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

            # Snake food
            pygame.draw.rect(game_window, white, pygame.Rect(food_pos[0], food_pos[1], 10, 10))
            
            # Game Over conditions
            # Getting out of bounds
            if snake_pos[0] < 0 or snake_pos[0] > args.width - 10:
                print("My planet needs me! Bye!")
                game_over()
            if snake_pos[1] < 0 or snake_pos[1] > args.height - 10:
                print("My planet needs me! Bye!")
                game_over()

            # Touching the snake body
            for block in snake_body[1:]:
                if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                    print("Oopsie! I ate a wall!")
                    game_over()

            show_score(1, white, 'Console', 20)

            # Refresh game screen
            pygame.display.update()

            # Refresh rate
            fps_controller.tick(25)
            
    



   



   