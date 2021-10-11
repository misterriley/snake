'''
Created on Sep 18, 2021

@author: bleem
'''

import random

from gym import spaces
from pygame import Rect
import gym
import pygame

import numpy as np

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

MS_PER_TICK = .750
GAME_OVER_SLEEP_TIME = 2

GRID_WIDTH = 4
GRID_HEIGHT = 4
GRID_PAD = 50

SCORE_X = 10
SCORE_Y = 10

CELL_WIDTH = 25
CELL_PAD = 1

SNAKE_COLOR = (0, 0, 100)
SNAKE_HEAD_COLOR = (0, 0, 255)
FOOD_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (0, 0, 0)
GRID_COLOR = (20, 20, 20)
TEXT_COLOR = (255, 255, 255)

EMPTY = 0
FULL = 1

COLLISION_REWARD = -10
FOOD_REWARD = 1
MOVE_CLOSER_REWARD = .1
MOVE_FARTHER_REWARD = -.1


class SnakeEnvironment(gym.Env):

    def __init__(self, mode):

        self.snake_head_x = None
        self.snake_head_y = None
        self.food_x = None
        self.food_y = None

        self.score = None
        self.snake_len = None
        self.game_over = None
        self.won_game = None
        self.action_space = spaces.Discrete(4)
        self.snake_head_grid = self.new_grid()
        self.snake_body_grid = self.new_grid()
        self.food_grid = self.new_grid()
        self.oob_grid = self.new_grid()

        # mark out top and bottom boundaries
        self.oob_grid[0] = FULL
        self.oob_grid[-1] = FULL

        # mark out left and right boundaries
        for n in range(GRID_HEIGHT + 2):
            self.oob_grid[n][0] = FULL
            self.oob_grid[n][-1] = FULL

        self.data = np.asarray([self.snake_head_grid, self.snake_body_grid, self.food_grid, self.oob_grid])
        self.observation_space = spaces.Box(low = 0, high = 1, shape = self.data.shape, dtype = np.int32)
        self.display = SnakeGameDisplay()
        self.mode = mode

    def render(self):
        if self.mode == 'human':
            frame_rate = 1 / MS_PER_TICK
            end_screen = True
        else:
            frame_rate = 60
            end_screen = False
        self.display.display_state(self, frame_rate, end_screen)

    def get_new_food_location(self):

        while True:
            food_x = random.randint(1, GRID_WIDTH)
            food_y = random.randint(1, GRID_HEIGHT)

            if self.snake_body_grid[food_x][food_y] == EMPTY:
                self.food_grid[food_x][food_y] = FULL
                self.food_x = food_x
                self.food_y = food_y
                # if not, keep searching for an empty spot
                break

    def new_grid(self):
        # add two to each dimension for the out-of-bounds markers
        return np.array([[EMPTY for _ in range(GRID_WIDTH + 2)] for _ in range(GRID_HEIGHT + 2)], dtype = np.int32)

    def reset(self):
        self.game_over = False
        self.won_game = None
        self.score = 0
        self.snake_len = 1
        self.snake_head_grid = self.new_grid()
        self.snake_body_grid = self.new_grid()
        self.food_grid = self.new_grid()
        self.data = np.asarray([self.snake_head_grid, self.snake_body_grid, self.food_grid, self.oob_grid])

        self.snake_head_x = int(GRID_WIDTH / 2 + 1)
        self.snake_head_y = int(GRID_HEIGHT / 2 + 1)
        self.snake_head_grid[self.snake_head_x][self.snake_head_y] = FULL
        self.snake_body_grid[self.snake_head_x][self.snake_head_y] = self.snake_len
        self.get_new_food_location()
        self.render()
        return self.data

    def step(self, action):

        reward = None

        self.snake_head_grid[self.snake_head_x][self.snake_head_y] = EMPTY

        if action == NORTH:
            self.snake_head_y -= 1
        elif action == EAST:
            self.snake_head_x += 1
        elif action == SOUTH:
            self.snake_head_y += 1
        elif action == WEST:
            self.snake_head_x -= 1

        self.snake_head_grid[self.snake_head_x][self.snake_head_y] = FULL
        if self.food_grid[self.snake_head_x][self.snake_head_y] == FULL:
            # got a food
            self.score += 1
            self.snake_len += 1
            self.food_grid[self.snake_head_x][self.snake_head_y] = EMPTY
            if not self.snake_has_filled_grid():
                self.get_new_food_location()
            reward = FOOD_REWARD
        else:
            self.snake_body_grid = np.maximum(self.snake_body_grid - 1, 0)

        if not self.head_is_in_bounds():
            self.game_over = True
            self.won_game = False
            reward = COLLISION_REWARD

        if not self.game_over:
            if self.snake_body_grid[self.snake_head_x][self.snake_head_y] != EMPTY:
                # head collided with tail
                self.game_over = True
                self.won_game = False
                reward = COLLISION_REWARD
            else:
                self.snake_body_grid[self.snake_head_x][self.snake_head_y] = self.snake_len
                if self.snake_len == GRID_HEIGHT * GRID_WIDTH:
                    self.game_over = True
                    self.won_game = True
                elif reward is None:
                    reward = 0

        self.render()
        self.data = np.asarray([self.snake_head_grid, self.snake_body_grid, self.food_grid, self.oob_grid])
        return (self.data, reward, self.game_over, {})

    def snake_has_filled_grid(self):
        return self.snake_len == GRID_HEIGHT * GRID_WIDTH

    def head_is_in_bounds(self):
        return self.oob_grid[self.snake_head_x][self.snake_head_y] == EMPTY

    def get_score(self):
        return self.score

    def is_over(self):
        return self.game_over

    def agent_has_won(self):
        return self.won_game

    def is_food(self, x, y):
        return x == self.food_x and y == self.food_y

    def is_snake_head(self, x, y):
        return x == self.snake_head_x and y == self.snake_head_y

    def is_snake_body(self, x, y):
        return self.snake_body_grid[x][y] != EMPTY


class SnakeGameDisplay:

    def __init__(self):
        pygame.init()

        pygame.event.set_allowed(pygame.KEYDOWN)

        total_width_px = GRID_PAD * 2 + GRID_WIDTH * CELL_WIDTH + (GRID_WIDTH - 1) * CELL_PAD
        total_height_px = GRID_PAD * 2 + GRID_HEIGHT * CELL_WIDTH + (GRID_HEIGHT - 1) * CELL_PAD

        self.surface = pygame.display.set_mode((total_width_px, total_height_px))
        self.clock = pygame.time.Clock()

    def draw_grid(self):
        top_y = self.get_px_from_grid_coordinate(1)
        bottom_y = self.get_px_from_grid_coordinate(GRID_HEIGHT + 1)
        for grid_x in range(1, GRID_WIDTH + 2):
            x = self.get_px_from_grid_coordinate(grid_x)
            pygame.draw.line(self.surface, GRID_COLOR, (x, top_y), (x, bottom_y))

        left_x = self.get_px_from_grid_coordinate(1)
        right_x = self.get_px_from_grid_coordinate(GRID_WIDTH + 1)
        for grid_y in range(1, GRID_HEIGHT + 2):
            y = self.get_px_from_grid_coordinate(grid_y)
            pygame.draw.line(self.surface, GRID_COLOR, (left_x, y), (right_x, y))

    def display_score(self, game):
        score_text = pygame.font.Font("freesansbold.ttf", 32).render("SCORE: " + str(game.get_score()), True, TEXT_COLOR)
        # must overwrite previous score text
        pygame.draw.rect(self.surface, BACKGROUND_COLOR, Rect(SCORE_X, SCORE_Y, 400, score_text.get_height()))
        self.surface.blit(score_text, (SCORE_X, SCORE_Y))

    def display_state(self, game, frame_rate, display_end_screen = False):
        if game.is_over():
            if display_end_screen:
                displaytext = "YOU WIN!" if game.agent_has_won() else "GAME OVER"
                self.do_end_screen(displaytext, game.get_score())
        else:
            self.draw_play_state(game)
        # print(game.grid)
        pygame.display.update()
        self.clock.tick(frame_rate)

    def draw_play_state(self, game):
        self.surface.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.display_score(game)
        for x in range(1, GRID_WIDTH + 1):
            for y in range(1, GRID_HEIGHT + 1):
                color = BACKGROUND_COLOR
                if game.is_food(x, y):
                    color = FOOD_COLOR
                elif game.is_snake_head(x, y):
                    color = SNAKE_HEAD_COLOR
                elif game.is_snake_body(x, y):
                    color = SNAKE_COLOR
                self.color_cell(color, x, y)

    def do_end_screen(self, displaytext, score):
        self.surface.fill(BACKGROUND_COLOR)
        game_over_text = pygame.font.Font("freesansbold.ttf", 32).render(displaytext, True, TEXT_COLOR)
        x = (self.surface.get_width() - game_over_text.get_width()) / 2
        y = (self.surface.get_height() - game_over_text.get_height()) / 2
        self.surface.blit(game_over_text, (x, y))
        score_text = pygame.font.Font("freesansbold.ttf", 32).render("SCORE: " + str(score), True, TEXT_COLOR)
        score_x = (self.surface.get_width() - score_text.get_width()) / 2
        score_y = y + game_over_text.get_height() + GRID_PAD
        self.surface.blit(score_text, (score_x, score_y))

    def color_cell(self, color, grid_x, grid_y):
        left_x = self.get_px_from_grid_coordinate(grid_x) + 1
        top_y = self.get_px_from_grid_coordinate(grid_y) + 1

        pygame.draw.rect(self.surface, color, Rect(left_x, top_y, CELL_WIDTH, CELL_WIDTH))

    # returns the px location of the left or top edge of the cell
    def get_px_from_grid_coordinate(self, coordinate):
        return GRID_PAD + (coordinate - 1) * (CELL_WIDTH + CELL_PAD)


class Controller:

    def __init__(self):
        self.direction = WEST

    def get_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.direction = NORTH
                elif event.key == pygame.K_RIGHT:
                    self.direction = EAST
                elif event.key == pygame.K_DOWN:
                    self.direction = SOUTH
                elif event.key == pygame.K_LEFT:
                    self.direction = WEST

        # if no direction chosen, continue in same direction
        return self.direction

    def play_game(self):
        game = SnakeEnvironment('human')
        game.reset()
        state = None
        while True:
            if game.is_over():
                for i in range(state.shape[0]):
                    print(state[i])
                game.reset()
            else:
                action = self.get_action()
                state, _, _, _ = game.step(action)


if __name__ == "__main__":
    controller = Controller()
    controller.play_game()
