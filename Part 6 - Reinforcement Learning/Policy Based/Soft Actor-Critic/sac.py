import webbrowser
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import cv2

def preprocess_state(state):
    # Convert to grayscale and normalize
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, IMAGE_SIZE)  # Downsample to reduce complexity
    state = state.astype(np.float32) / 255.0  # Normalize
    return state.flatten()

# Setting the hyperparameters
HIDDEN_DIM = 256
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 1e6
BATCH_SIZE = 128
ALPHA = 0.2  # Entropy coefficient
IMAGE_SIZE = (64, 64)  # Size to which the images will be resized

ENV_NAME = "CarRacing-v3"
env = gym.make(ENV_NAME)
state, _ = env.reset()
state_dim = preprocess_state(state).shape[0]
action_dim = env.action_space.shape[0]

# Implementing the Replay Buffer
class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, BATCH_SIZE))
        state = torch.from_numpy(np.stack(state)).float()
        action = torch.from_numpy(np.stack(action)).float()
        reward = torch.from_numpy(np.stack(reward)).float().unsqueeze(1)
        next_state = torch.from_numpy(np.stack(next_state)).float()
        done = torch.from_numpy(np.stack(done)).float().unsqueeze(1)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# Building the Actor Network
class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state, return_log_prob=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        if return_log_prob:
            log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)
            return action, log_prob
        return action

# Building the Critic Network
class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Building the SAC Agent
class SACAgent:

    def __init__(self):
        self.actor = Actor()
        self.critic_1 = Critic()
        self.critic_2 = Critic()
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR, weight_decay=0)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=CRITIC_LR, weight_decay=0)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=CRITIC_LR, weight_decay=0)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor.sample(state, return_log_prob=False)
        return action.detach().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample()

        with torch.no_grad():
            action, log_prob = self.actor.sample(state)
            target_q1 = self.target_critic_1(next_state, action)
            target_q2 = self.target_critic_2(next_state, action)
            target_q_min = torch.min(target_q1, target_q2) - ALPHA * log_prob
            target_q = reward + (1 - done) * GAMMA * target_q_min

        # Update of the Critic 1 network
        current_q1 = self.critic_1(state, action)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        # Update of the Critic 2 network
        current_q2 = self.critic_2(state, action)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Update of the Actor network
        action, log_prob = self.actor.sample(state)
        q_val = self.critic_1(state, action)
        actor_loss = (ALPHA * log_prob - q_val).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update of the Critic Target networks
        soft_update(self.target_critic_1, self.critic_1)
        soft_update(self.target_critic_2, self.critic_2)

def soft_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# Creating the agent
agent = SACAgent()

# Implementing the Training Loop
num_episodes = 5
for episode in range(num_episodes):
    frames = []
    state, _ = env.reset()
    state = preprocess_state(state)
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = preprocess_state(next_state)
        done = terminated or truncated
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        episode_reward += reward
        
    print(f"\rEpisode {episode}: Total Reward: {episode_reward}", end="")

# Saving the model
torch.save(agent.actor.state_dict(), 'sac_actor.pth')
torch.save(agent.critic_1.state_dict(), 'sac_critic_1.pth')
torch.save(agent.critic_2.state_dict(), 'sac_critic_2.pth')

# Loading the model
def load_model(agent, actor_path, critic_1_path, critic_2_path):
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic_1.load_state_dict(torch.load(critic_1_path))
    agent.critic_2.load_state_dict(torch.load(critic_2_path))
    
# load_model(agent, 'sac_actor.pth', 'sac_critic_1.pth', 'sac_critic_2.pth')

# Visualizing the results
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode = 'rgb_array')
    
    state, _ = env.reset()
    state = preprocess_state(state)
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.select_action(state)
        state, _, terminated, truncated, _= env.step(action)
        state = preprocess_state(state)
        done = terminated or truncated
    env.close()
    imageio.mimsave('video.mp4', frames, fps=60)

show_video_of_model(agent, ENV_NAME)
webbrowser.open('video.mp4')