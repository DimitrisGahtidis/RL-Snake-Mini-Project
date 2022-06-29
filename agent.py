import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # append a tuples worth of information to the memory

    def train_long_memory(self):
            if len(self.memorty) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a batch size number of tuples
            else: # if the memorty is lower than the batch size take the whole memory
                mini_sample = self.memory
            
            states, actions, rewards, next_states, dones = zip(*mini_sample)

def train():
    pass
        
if __name__ == '__main__':
    train()
