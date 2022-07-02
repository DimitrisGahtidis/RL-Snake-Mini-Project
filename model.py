import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    pass

class QTrainer:
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

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
        loss.backward()

        self.optimizer.step()
