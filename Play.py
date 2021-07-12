from Environment import Environment
from Constants import *
from Model import Model
from Model import Trainer
from Agent import Agent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import torch

parser = argparse.ArgumentParser(prog='Play.py',usage='%(prog)s File_name [options]',description='Get the configuration for training')
parser.add_argument('File_name', metavar='',help="Enter the model name you want to play")
args = parser.parse_args()


def load_game(path):
    game = torch.load(path)
    model = Model()
    model.load_state_dict(game['model_state_dict'])
    discount = game['discount']
    lr = game['lr']
    ghosts = game['ghosts']
    size = game['size']
    return model, size, ghosts, lr, discount

file_name = args.File_name if(args.File_name[len(args.File_name)-4:] == '.pth') else args.File_name + '.pth'
model, size, ghosts, lr, discount = load_game('Models/' + file_name)
env = Environment(size,ghosts)
trainer = Trainer(model,lr,discount)
agent = Agent(env,model,trainer,file_name)
agent.env.reset()
agent.epsilon = 0


def create_grid():
    rows = len(agent.env.grid)
    cols = len(agent.env.grid[0])
    grid = [[agent.env.grid[i][j].value for j in range(cols)] for i in range(rows)]
    return grid

fig = plt.figure()
im = plt.imshow(create_grid(),animated = True)

time = 0
def main_loop(*args):
    global time
    global size
    state_old = agent.get_state()
    move = agent.get_move(state_old)
    pacman, ghosts, food = agent.get_locations(move)
    score, reward, game_over = agent.env.next_step(pacman,ghosts, food)
    if(reward == Reward.FOOD.value):
        time = 0
    time+=1
    if time > 10*size:
        game_over = True
    im.set_array(create_grid())
    if game_over:
        ani.event_source.stop()
        print("Score : " + str(score))
    
    

ani = animation.FuncAnimation(fig,main_loop,interval=40)

plt.show()

