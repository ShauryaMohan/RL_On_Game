from Constants import INPUT_SIZE, OUTPUT_SIZE
import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(INPUT_SIZE,OUTPUT_SIZE),
        )

    def forward(self,xb):
        xb = torch.tensor(xb,dtype=torch.float)
        out = self.linear(xb)
        return out

class Trainer:
    def __init__(self,model,lr,discount) -> None:
        self.lr = lr
        self.model = model
        self.discount = discount
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,game_over):
        
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            action = action.unsqueeze(0)
            game_over = (game_over,)
        

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(state)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.discount*torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            # print("Target : ", target)
            # print("Predictions : ", pred)

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()
        


