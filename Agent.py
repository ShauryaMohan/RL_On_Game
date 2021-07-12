from Constants import *
import numpy as np
from collections import deque
import random
import torch

class Agent:
    def __init__(self,env,model,trainer,file_name) -> None:
        self.env = env
        self.model = model
        self.trainer = trainer
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = EPSILON
        self.file_name = file_name
        self.games = 0
    
    def get_state(self):
        pacman_x,pacman_y = self.env.make_2D(self.env.pacman)
        food_x, food_y = self.env.make_2D(self.env.food)
        wall_left = (pacman_y == 0)
        wall_up = (pacman_x == 0)
        wall_right = (self.env.size  - 1 ==  pacman_y)
        wall_down = (self.env.size - 1 == pacman_x)
        food_up = int(food_x < pacman_x)
        food_down = int(food_x > pacman_x)
        food_left = int(food_y < pacman_y)
        food_right = int(food_y > pacman_y)
        state = [wall_up,wall_left,wall_down,wall_right,food_up,food_left,food_down,food_right]
        if (self.env.is_legal(pacman_x-1,pacman_y-1)):
            state += [1 if (self.env.grid[pacman_x-1][pacman_y-1] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x+1,pacman_y-1)):
            state += [1 if (self.env.grid[pacman_x+1][pacman_y-1] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x-1,pacman_y+1)):
            state += [1 if (self.env.grid[pacman_x-1][pacman_y+1] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x+1,pacman_y+1)):
            state += [1 if (self.env.grid[pacman_x+1][pacman_y+1] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x-2,pacman_y)):
            state += [1 if (self.env.grid[pacman_x-2][pacman_y] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x+2,pacman_y)):
            state += [1 if (self.env.grid[pacman_x+2][pacman_y] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x,pacman_y-2)):
            state += [1 if (self.env.grid[pacman_x][pacman_y-2] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x,pacman_y+2)):
            state += [1 if (self.env.grid[pacman_x][pacman_y+2] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x,pacman_y+1)):
            state += [1 if (self.env.grid[pacman_x][pacman_y+1] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x,pacman_y-1)):
            state += [1 if (self.env.grid[pacman_x][pacman_y-1] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x-1,pacman_y)):
            state += [1 if (self.env.grid[pacman_x-1][pacman_y] == Code.GHOST) else 0]
        else:
            state += [0]
        if (self.env.is_legal(pacman_x+1,pacman_y)):
            state += [1 if (self.env.grid[pacman_x+1][pacman_y] == Code.GHOST) else 0]
        else:
            state += [0]
        return state

    def get_move(self,state):
        chance = np.random.randint(0,EPSILON_FACTOR*EPSILON)
        move = [0,0,0,0]
        if chance < self.epsilon:
            move[np.random.randint(0,4)] = 1
        else:
            pred = self.model(state)
            # pred = pred[0]
            maximum = pred[0]
            i = 0
            for idx in range(len(pred)):
                if maximum < pred[idx]:
                    maximum = pred[idx]
                    i = idx
            move[i] = 1
        return move

    def get_locations(self,move):
        pacman_x, pacman_y = self.env.make_2D(self.env.pacman)
        new_ghosts = []
        new_pacman = self.env.pacman
        if move[0]:
            if self.env.is_legal(pacman_x+1,pacman_y):
                new_pacman = self.env.flatten(pacman_x+1,pacman_y)
        elif move[1]:
            if self.env.is_legal(pacman_x,pacman_y+1):
                new_pacman = self.env.flatten(pacman_x,pacman_y+1)
        elif move[2]:
            if self.env.is_legal(pacman_x-1,pacman_y):
                new_pacman = self.env.flatten(pacman_x-1,pacman_y)
        else:
            if self.env.is_legal(pacman_x,pacman_y-1):
                new_pacman = self.env.flatten(pacman_x,pacman_y-1)

        # Ghosts Locations
        for ghost in self.env.ghosts:
            new_ghosts.append(self.get_new_ghost_location(new_pacman,ghost))
        if self.env.food == new_pacman:
            found_empty_space = False
            while not found_empty_space:
                location = np.random.randint(0,self.env.size*self.env.size)
                found_empty_space = (location != new_pacman)
            food = location
        else:
            food = self.env.food
        return new_pacman, new_ghosts, food

    def get_new_ghost_location(self,pacman,ghost):
        pacman_x, pacman_y = self.env.make_2D(pacman)
        ghost_x, ghost_y = self.env.make_2D(ghost[0])
        if (pacman_x == ghost_x - 1 and pacman_y == ghost_y):
            return [pacman,0]
        elif (pacman_x == ghost_x + 1 and pacman_y == ghost_y):
            return [pacman,2]
        elif (pacman_x == ghost_x and pacman_y == ghost_y + 1):
            return [pacman,1]
        elif (pacman_x == ghost_x and pacman_y == ghost_y - 1):
            return [pacman,3]
        else:
            new_ghost = ghost[0]
            changed_direction = ghost[1]
            chance = np.random.randint(0,100)
            if chance < 20:
                changed_direction = np.random.randint(0,4)
            
            if(ghost[1] == 0):
                if(ghost_x > 0):
                    new_ghost = self.env.flatten(ghost_x-1,ghost_y)
                else:
                    arr = [1,2,3]
                    changed_direction = arr[np.random.randint(0,3)]
            elif(ghost[1] == 1):
                if(ghost_y != self.env.size-1):
                    new_ghost = self.env.flatten(ghost_x,ghost_y+1)
                else:
                    arr = [0,2,3]
                    changed_direction = arr[np.random.randint(0,3)]
            elif(ghost[1] == 2):
                if(ghost_x != self.env.size - 1):
                    new_ghost = self.env.flatten(ghost_x+1,ghost_y)
                else:
                    arr = [0,1,3]
                    changed_direction = arr[np.random.randint(0,3)]
            else:
                if(ghost_y > 0):
                    new_ghost = self.env.flatten(ghost_x,ghost_y-1)
                else:
                    arr = [0,1,2]
                    changed_direction = arr[np.random.randint(0,3)]
            return [new_ghost,changed_direction]
                

    def remember(self,old_state,move,reward,new_state,game_over):
        self.memory.append((old_state,move,reward,new_state,game_over))
    
    def train(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory

        states, moves, rewards, next_states, game_overs = zip(*batch)
        self.trainer.train_step(states, moves, rewards, next_states, game_overs)

    def train_short(self,old_state,move,reward,new_state,game_over):
        self.trainer.train_step(old_state,move,reward,new_state,game_over)
    
    def decrease_epsilon(self):
        self.epsilon -= 1

    def increment_games(self):
        self.games += 1

    def save_model(self):
        MODEL_NAME = self.file_name + '.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ghosts': len(self.env.ghosts),
            'size' : self.env.size,
            'discount' : self.trainer.discount,
            'lr' : self.trainer.lr,
        },'Models/' + MODEL_NAME)


