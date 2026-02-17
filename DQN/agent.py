import torch
import torch.nn.functional as F
import random

from network import QNetwork

class PursuerAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-5)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995


        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.q_net(state_tensor)
            return torch.argmax(q_vals).item()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        next_q = self.target_net(next_states).max(1)[0]
        target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update
        for target_param, param in zip(self.target_net.parameters(),
                                       self.q_net.parameters()):
            target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)

        self.epsilon = max(self.epsilon_min,
                   self.epsilon * self.epsilon_decay)

