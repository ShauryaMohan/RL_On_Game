from Constants import Code, Reward
import numpy as np
class Environment:
    def __init__(self,size,ghost_count) -> None:
        self.size = size
        self.grid = [[Code.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.ghost_count = ghost_count
        self.ghosts = [[-1,-1] for _ in range(ghost_count)]
        self.pacman = -1
        self.food = -1
        self.score = 0
        

    def clear_grid(self):
        self.grid = [[Code.EMPTY for _ in range(self.size)] for _ in range(self.size)]

    def reset(self):
        self.clear_grid()
        shuffled = list(range(self.size*self.size))
        np.random.shuffle(shuffled)
        for idx in range(self.ghost_count):
            direction = np.random.randint(0,4)
            self.add_to_grid([shuffled[idx],direction],Code.GHOST,idx)
        self.add_to_grid(shuffled[self.ghost_count],Code.FOOD,-1)
        self.add_to_grid(shuffled[self.ghost_count+1],Code.PACMAN,-1)
        self.score = 0



    def next_step(self,pacman,ghosts,food):
        self.clear_grid()
        game_over = False
        reward = Reward.MOVE.value
        if (self.food != food):
            self.increment_score()
            reward = Reward.FOOD.value
        if (self.pacman == pacman):
            reward = Reward.STAY.value
        self.add_to_grid(food,Code.FOOD,-1)
        self.add_to_grid(pacman,Code.PACMAN,-1)
        idx = 0
        for ghost in ghosts:
            self.add_to_grid(ghost,Code.GHOST,idx)
            idx += 1
        for ghost in self.ghosts:
            if(self.pacman == ghost[0]):
                reward = Reward.DEAD.value
                game_over = True
        return self.score, reward, game_over

    
    def make_2D(self,location):
        x = location//self.size
        y = location%self.size
        return x,y

    def flatten(self,x,y):
        return x*self.size + y

    def increment_score(self):
        self.score += 1

    def get_score(self):
        return self.score

    def add_to_grid(self,location,code,idx):
        if idx != -1:
            x,y = self.make_2D(location[0])
            self.grid[x][y] = code
            self.ghosts[idx] = location
        else:
            x,y = self.make_2D(location)
            self.grid[x][y] = code
            if(code == Code.FOOD):
                self.food = location
            else:
                self.pacman = location
    def is_legal(self,x,y):
        return x >= 0 and y >= 0 and x < self.size and y < self.size