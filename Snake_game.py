"""
This file has the necessary code to start the snake game.
The code has been slightly modified from its original state, see (https://github.com/rajatdiptabiswas/snake-pygame) for the original code.
"""
from Jormungandr import jormungandr
import pygame, sys, time, random, argparse


parser = argparse.ArgumentParser(description = "Start the game!")
parser.add_argument("-d","--difficulty",dest = 'difficulty', type = int, required = False, default = 2, help = "Difficulty of the snake game: 1.- Easy  2.- Medium  3.- Hard 4.- Very Hard  5.- Impossible.   Default is 2")

args = parser.parse_args()

difficulty_dict = {
    # Difficulty settings
    # Easy      ->  10
    # Medium    ->  25
    # Hard      ->  40
    # Harder    ->  60
    # Impossible ->  120
    1: 10,
    2: 25,
    3: 40,
    4: 60,
    5: 120
}

#Translation of the difficulty
difficulty = difficulty_dict[args.difficulty]

# Window size
frame_size_x = 720
frame_size_y = 480

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))


# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()


# Game variables

#Default snake position
snake_pos = [100, 50]
#Snake starts with length == 3
snake_body = [[100, 50], [100-10, 50], [100-(2*10), 50]]

#We generate the first piece of food
food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
food_spawn = True

direction = 'RIGHT'
change_to = direction

score = 0


# Game Over
def game_over():
    my_font = pygame.font.SysFont('times new roman', 90)
    game_over_surface = my_font.render('YOU DIED', True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x/2, frame_size_y/4)
    game_window.fill(black)
    game_window.blit(game_over_surface, game_over_rect)
    show_score(0, red, 'times', 20)
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    sys.exit()


# Score
def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_size_x/10, 15)
    else:
        score_rect.midtop = (frame_size_x/2, frame_size_y/1.25)
    game_window.blit(score_surface, score_rect)
    # pygame.display.flip()

snake = jormungandr(frame_size_x, frame_size_y)
# Main logic
while True:
   
    move_sequence = snake.pathfind(snake_body, food_pos)

    for move in move_sequence:
        # Moving the snake

        #UP
        if move == 1:
            snake_pos[1] -= 10

        #DOWN
        if move == 2:
            snake_pos[1] += 10

        #LEFT
        if move == 3:
            snake_pos[0] -= 10

        #RIGHT
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
            food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]

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
        if snake_pos[0] < 0 or snake_pos[0] > frame_size_x - 10:
            print("My planet needs me! Bye!")
            game_over()
        if snake_pos[1] < 0 or snake_pos[1] > frame_size_y - 10:
            print("My planet needs me! Bye!")
            game_over()
        # Touching the snake body
        for block in snake_body[1:]:
            if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                print("Oopsie! I ate a wall!")
                game_over()

        show_score(1, white, 'consolas', 20)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        fps_controller.tick(difficulty)
            
    



   



   