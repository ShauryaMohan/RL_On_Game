from Configuration import Configuration
from Environment import Environment
from Constants import *
from Model import Model
from Model import Trainer
from Agent import Agent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

parser = argparse.ArgumentParser(prog='Train.py',usage='%(prog)s file_name [options]',description='Get the configuration for training')
parser.add_argument('-gh','--ghosts', type=int, metavar='',help="Enter the number ghosts for the training")
parser.add_argument('-s','--size', type=int, metavar='',help="Enter the grid size you want to use")
parser.add_argument('-ani','--animate',help="use this argument to animate the training process", action="store_true")
parser.add_argument('-d','--discount',type=float, metavar='',help='The discount factor to be used in the model')
parser.add_argument('-lr','--learning',type = float, metavar='',help='Enter the learning rate for the model')
parser.add_argument('file_name', metavar='',help="Enter a name for the model, you may use your name")
args = parser.parse_args()

ghosts = args.ghosts if (args.ghosts) else DefaultConfiguration.GHOSTS.value
size = args.size if (args.size) else DefaultConfiguration.SIZE.value
animate = args.animate
lr = args.learning if (args.learning) else DefaultConfiguration.LR.value
discount = args.discount if (args.discount) else DefaultConfiguration.DISCOUNT.value
configuration = Configuration(discount,lr,ghosts,size)

record = 0
time = 0

def train(configuration,animate,file_name):
    env = Environment(configuration.size,configuration.ghosts)
    model = Model()
    trainer = Trainer(model,configuration.lr,configuration.discount)
    agent = Agent(env,model,trainer,file_name)
    agent.env.reset()

    if not animate:
        time = 0
        total_time = 0
        record = 0
        total_score = 0
        while True:
            state_old = agent.get_state()
            move = agent.get_move(state_old)
            pacman, ghosts, food = agent.get_locations(move)
            score, reward, game_over = agent.env.next_step(pacman,ghosts, food)
            if (reward == Reward.FOOD.value):
                time = 0
            if (time  >= 10*configuration.size):
                reward = Reward.DEAD.value
                game_over = True
            state_new = agent.get_state()
            agent.remember(state_old,move,reward,state_new,game_over)
            agent.train_short(state_old,move,reward,state_new,game_over)
            time += 1
            total_time += 1
            if total_time > 1000000:
                print("Ended the training with score : " + str(score))
                agent.save_model()
                break
            if game_over:
                agent.env.reset()
                agent.decrease_epsilon()
                agent.increment_games()
                time = 0
                total_time = 0
                total_score += score
                agent.train()
                if record < score:
                    record = score
                    agent.save_model()
                if agent.games%100 == 0:
                    average_score = total_score/100
                    total_score = 0
                    print("Stats between game " + str(agent.games-100) + " and " + str(agent.games))
                    print("Average Score : " + str(average_score) + " Highest Score : " + str(record))
    else:

        def create_grid():
            rows = len(agent.env.grid)
            cols = len(agent.env.grid[0])
            grid = [[agent.env.grid[i][j].value for j in range(cols)] for i in range(rows)]
            return grid

    
        fig = plt.figure()
        im = plt.imshow(create_grid(),animated = True)


        def main_loop(*args):
            global record
            global time
            state_old = agent.get_state()
            move = agent.get_move(state_old)
            pacman, ghosts, food = agent.get_locations(move)
            score, reward, game_over = agent.env.next_step(pacman,ghosts, food)
            if (reward == Reward.FOOD.value):
                time = 0
            if (time  == 10*configuration.size):
                reward = Reward.DEAD.value
                game_over = True
            state_new = agent.get_state()
            agent.remember(state_old,move,reward,state_new,game_over)
            agent.train_short(state_old,move,reward,state_new,game_over)
            time += 1
            im.set_array(create_grid())
            if game_over:
                agent.env.reset()
                agent.decrease_epsilon()
                agent.increment_games()
                time = 0
                agent.train()
                print("Game number : " + str(agent.games))
                if record < score:
                    record = score
                    agent.save_model()
                    print("Congratulations New Record made! " + str(record))
                else: 
                    print("Score : " + str(score))
        ani = animation.FuncAnimation(fig,main_loop,interval=40)
        plt.show()

if __name__ == "__main__":
    train(configuration,animate,args.file_name)
            