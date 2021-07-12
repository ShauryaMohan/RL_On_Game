from enum import Enum
class Code(Enum):
    EMPTY = [255,255,255]
    PACMAN = [255,255,0]
    GHOST = [255,0,0]
    FOOD = [0,0,0]

class Reward(Enum):
    FOOD = 40
    DEAD = -30
    MOVE = 1
    STAY = -5

class DefaultConfiguration(Enum):
    DISCOUNT = 0.9
    LR = 0.01
    GHOSTS = 8
    SIZE = 30

EPSILON = 2000
EPSILON_FACTOR = 2
BATCH_SIZE = 1000

INPUT_SIZE = 20
OUTPUT_SIZE = 4
HIDDEN_SIZE = 10
MAX_MEMORY = 10000







