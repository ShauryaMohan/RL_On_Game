# About this project
In this project I made a pacman like game where the pacman (yellow cell) has to eat a food (black cell) on the grid while avoiding ghosts (red cells). All the animation has been done using Matplotlib.animation. Next, I trained my pacman to play this game using Reinforcement learning algorithms. In this case I used a combination of TD(0) algorithm and Monte-Carlo algorithm. Initially I used a neural network to approximate Q(s,a) values but later switched to linear function as they trained faster without much compromise on quality of training.

# How to use this repository:
## Setting up the project
You can clone this project to your computer. To get started with training and playing with your trained agent, first you will require to install pytorch. You can do this by going to the root directery and typing `pip install -r requirements.txt`. This will install pytorch version 1.7.0. But if you have pytorch insall, this project will work on most of the versions so you don't have to worry.  

## Playing right away:
You can observe the already trained agent play without training one yourself. This can be done by going the following command at the root directory:
```
python Play.py sample_model
```
This will begin an animation where you can see the already trained pacman avoiding the ghosts and eating food.

## Training the pacman
To begin with training you will have to use Train.py file. Here you can train the pacman in two ways, one in which you can observe the animation of pacman getting trained and the faster one where all the training takes place in the background and you can later play with the trained model. For training with default values you can type the following command. (The default values are Learning Rate: 0.01, Discount (gamma): 0.9, Grid Size: 30, Number of ghosts: 8)
 ```
 python Train.py <a-name-for-the-model>
 ```
To change the default values you can use the optional arguments as follows:
```
positional arguments:
                     Name the file, you may use your name

optional arguments:
  -h, --help         show this help message and exit
  -gh , --ghosts     Enter the number ghosts for the training
  -s , --size        Enter the grid size you want to use
  -ani, --animate    use this argument to animate the training process
  -d , --discount    The discount factor to be used in the model
  -lr , --learning   Enter the learning rate for the model
```
These flags are optional and can be used to change the default values, for example if you want to have 12 ghosts in a grid size of 40, with learning rate 0.1 and discount 0.8 you will have to type the folling command:
```
python Train.py model_2 -gh 12 -s 40 -lr 0.1 -d 0.8
```
  
On training you will observe the following lines on command line (for without animation) this will give you stats of your game and you can check if it is training or not:  
> Stats between game 400 and 500  
> Average Score : 0.37 Highest Score : 4  
> Stats between game 500 and 600  
> Average Score : 0.67 Highest Score : 6  

### Training with animation
You can also train your model and observe the training (long process) using the optional argument -ani or --animate, this is shown as follows:
```
python Train.py <a-name-for-the-model> -ani -gh 10
```
This will train your model with 10 ghosts and animation on, that is you can watch the pacman being trained. This is not recommended if you don't have lot of time, but is interesting to understand how the pacman is learning.  
  
If you forget the arguments or want to see how to use them you can always type the following command to get help on how to use. 
```
python Train.py -h
```

## Playing a trained model
To play with a trained model (that you just trained or maybe the sample_model or one you trained a while back) you can use the name of the model after the command Python Play.py to watch the trained pacman playing the game. Again this command has to by typed at the root directory of this repository. An example:
```
python Play.py <name-of-the-model-you-trained>
```
You can also change the speed (frames per second) of the game by using an optional argument -t or --time which denotes the time it takes to change on frame on milli seconds, below is an example on how to use this argument (the default value is 40ms that is 25fps):
```
python Play.py <name-of-the-model-you-trained> -t 100
```

