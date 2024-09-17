from math import sqrt
import time, sys
import pygame

import numpy as np

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)

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

def game_over(frame_size_x, frame_size_y, game_window):
    my_font = pygame.font.SysFont('times new roman', 90)
    game_over_surface = my_font.render('YOU DIED', True, RED)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x / 2, frame_size_y / 4)
    game_window.fill(BLACK)
    game_window.blit(game_over_surface, game_over_rect)
    show_score(0, RED, 'times', 20)
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    sys.exit()

# Score
def show_score(color, font, size, frame_size_x, score, game_window):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    score_rect.midtop = (frame_size_x / 10, 15)
    game_window.blit(score_surface, score_rect)

    return

def show_reward(font, size, frame_size_x, reward_total, game_window):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Reward : ' + str(reward_total), True, BLUE)
    score_rect = score_surface.get_rect()
    score_rect.midtop = (frame_size_x / 10, 15)
    game_window.blit(score_surface, score_rect)

    return

def softmax(array):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(array - np.max(array))
    return e_x / e_x.sum()
