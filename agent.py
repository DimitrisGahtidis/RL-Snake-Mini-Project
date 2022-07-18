import torch
import random
import numpy as np
from statistics import mean
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, save_plot

MAX_MEMORY = 100_000
BATCH_SIZE = 2**10
LR = 0.001

class Agent:
    def __init__(self, device):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=device)
        self.device = device

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def train_short_memory(self, state, action, reward, next_state, done):
     # The code below is using the train_step function inside the Qtrainer class
        self.trainer.train_step(state, action, reward, next_state, done)
    pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # append a tuples worth of information to the memory

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a batch size number of tuples
        else: # if the memorty is lower than the batch size take the whole memory
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def get_action(self, state):
        # random moves have a tradeoff: exploration vs. exploitation
        # the probability of executing a random move is inversely 
        # proportional to the number of games played by the agent

        final_move = [0,0,0]
        if np.random.random(1)[0] < self.epsilon:
            # then define a random move
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # else get the next move as defined by the model
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(file_names, epsilon_rate, gamma_rate, response, i):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_scores = []
    plot_mean_scores = []

    plot_reward = []
    total_rewards = 0
    plot_mean_reward = []

    plot_loss = []
    plot_mean_loss = []

    plot_gamma = []

    record = 0
    n_episodes = 2000
    
    def save_plot_info(file_names):
        for tuple in enumerate(file_names):
            save_plot(*tuple)
    
    if response == 'y':
        load = True
    elif response == 'n':
        load = False
    else:
        exit()

    
    agent = Agent(device)

    if load:
        agent.model.load()
        
    game = SnakeGameAI()
    while (agent.n_games <= n_episodes):
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        total_rewards += reward

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(file_name=f'model_{i}.pth')

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Return:', total_rewards, 'Loss:', round(agent.trainer.loss,4))

            gamma = agent.gamma

            plot_reward.append(total_rewards)
            total_rewards = 0
            plot_loss.append(agent.trainer.loss)
            plot_scores.append(score)
            plot_gamma.append(gamma)
            #total_score += score
            #mean_score = total_score / agent.n_games
            #plot_mean_scores.append(mean_score)

            if agent.n_games >= 10:
                plot_mean_reward.append(mean([plot_reward[-i-1] for i in np.arange(10)]))
                plot_mean_loss.append(mean([plot_loss[-i-1] for i in np.arange(10)]))
                plot_mean_scores.append(mean([plot_scores[-i-1] for i in np.arange(10)]))
            else:
                plot_mean_reward.append(mean([plot_reward[-i-1] for i in np.arange(agent.n_games)]))
                plot_mean_loss.append(mean([plot_loss[-i-1] for i in np.arange(agent.n_games)]))
                plot_mean_scores.append(mean([plot_scores[-i-1] for i in np.arange(agent.n_games)]))


            #plot(plot_scores, plot_mean_scores, plot_return, plot_mean_return, plot_loss, plot_mean_loss)

            if (agent.n_games+1) %100 == 0:
                plot(plot_scores, 'Score', 0, xlabel='Number of Games', ylabel='Score', delay=0.1, clear=True)
                plot(plot_mean_scores, 'Mean Score', 0)
                plot(plot_reward, 'Reward', 1, xlabel='Number of Games', ylabel='Reward', delay=0.1, clear=True)
                plot(plot_mean_reward, 'Mean Reward', 1)
                plot(plot_loss, 'Loss', 2, xlabel='Number of Games', ylabel='Loss', delay=0.1, clear=True, round_num=4)
                plot(plot_mean_loss, 'Mean Loss', 2, round_num=4)
                plot(plot_gamma, 'Discount value', 3, xlabel='Number of Games', ylabel='Discount Value', delay=0.1, clear=True, round_num=3)
                save_plot_info(file_names)
            if agent.n_games == n_episodes:
                plot(plot_scores, 'Score', 0, xlabel='Number of Games', ylabel='Score', delay=0.1, clear=True)
                plot(plot_mean_scores, 'Mean Score', 0)
                plot(plot_reward, 'Reward', 1, xlabel='Number of Games', ylabel='Reward', delay=0.1, clear=True)
                plot(plot_mean_reward, 'Mean Reward', 1)
                plot(plot_loss, 'Loss', 2, xlabel='Number of Games', ylabel='Loss', delay=0.1, clear=True, round_num=4)
                plot(plot_mean_loss, 'Mean Loss', 2, round_num=4)
                plot(plot_gamma, 'Discount value', 3, xlabel='Number of Games', ylabel='Discount Value', delay=0.1, clear=True, round_num=3)
                save_plot_info(file_names)
            
            agent.epsilon = epsilon_rate(agent.n_games)
            agent.gamma = gamma_rate(agent.n_games)
        
        
if __name__ == '__main__':
    response = input("Would you like to load the most recent model [y/n]:")

    gamma_values = [0.9, 0.925, 0.99, 1, 1.01]
    def epsilon_rate(t):
        return 1*(0.5)**(t/50)

    for i, gamma in enumerate(gamma_values):
        if i > 1:
            def gamma_rate(t):
                return gamma

            file_names = [f'score_{i}.png', f'reward_{i}.png', f'loss_{i}.png', f'discount_{i}.png']

            train(file_names, epsilon_rate, gamma_rate, response, i)
    
    # def gamma_rate(t):
    #     r = (0.5)**(t/400)
    #     return 0.9*r + (1-r)*1.01
    
    
