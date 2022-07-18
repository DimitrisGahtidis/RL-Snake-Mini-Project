import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size))
    
    def forward(self, x):
        return self.layers(x)

    def save(self, file_name='model.pth'):
        print("Record reached, saving QNet...")
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_directory = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_directory)
    
    def load(self, file_name='model.pth'):
        print("Loading QNet...")
        model_folder_path = './model'
        file_directory = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_directory))
        self.eval() 
        

class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.device = device
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.float).to(self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: r + y * max(next_predicted Q value) -> only if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        clone = pred.clone()
        for index in range(len(done)):
            Q_new = reward[index]
            if not done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            clone[index][torch.argmax(action).item()] = Q_new

        # empty gradients and update with back prop
        self.optimizer.zero_grad()
        loss = self.criterion(clone, pred)
        self.loss = loss.item()
        loss.backward()

        self.optimizer.step()
