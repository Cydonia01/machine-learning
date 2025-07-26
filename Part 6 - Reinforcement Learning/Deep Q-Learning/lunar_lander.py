import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import gymnasium as gym
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers import RecordVideo
import webbrowser

# Building the AI
# Creating the architecture of the neural network
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed = 42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# Training the AI
# Setting up the environment
env = gym.make('LunarLander-v3')
state_shape = env.observation_space.shape
state_size = state_shape[0]
number_actions = env.action_space.n

# Initializing the hyperparameters
learning_rate = 5e-4
minibatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3

# Implementing Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones
    
# Implementing the DQN class
class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.memory) > minibatch_size:
            experiences = self.memory.sample(minibatch_size)
            self.learn(experiences, discount_factor)
        
    def act(self, state, epsilon = 0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)
        
    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)
            
# Initializing the agent
agent = Agent(state_size, number_actions)

# Training the agent
# number_episodes = 2000
# maximum_number_timesteps_per_episode = 1000
# epsilon_starting_value = 1.0
# epsilon_ending_value = 0.01
# epsilon_decay_value = 0.995
# epsilon = epsilon_starting_value
# scores_on_100_episodes = deque(maxlen = 100)

# record_training = True
# record_every_n_episodes = 40
# all_training_frames = []

# for episode in range(1, number_episodes + 1):
#     env = gym.make('LunarLander-v3', render_mode='rgb_array')
#     state, _ = env.reset()
#     score = 0
#     episode_frames = []
    
#     for t in range(maximum_number_timesteps_per_episode):
#         if record_training and episode % record_every_n_episodes == 0:
#             episode_frames.append(env.render())
        
#         action = agent.act(state, epsilon)
#         next_state, reward, done, _, _ = env.step(action)
#         agent.step(state, action, reward, next_state, done)
#         state = next_state
#         score += reward
#         if done:
#             break
        
#     if record_training and episode % record_every_n_episodes == 0:
#         all_training_frames.extend(episode_frames)
        
#     scores_on_100_episodes.append(score)
#     epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
#     print('\rEpisode {}\tAverage Score: {:.2f}\t'.format(episode, np.mean(scores_on_100_episodes)), end="")
    
#     if episode % 100 == 0:
#         print('\rEpisode {}\tAverage Score: {:.2f}\t'.format(episode, np.mean(scores_on_100_episodes)))
    
#     if np.mean(scores_on_100_episodes) >= 200.0:
#         print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
#         torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
#         break

# if record_training and all_training_frames:
#     imageio.mimsave('training_video.mp4', all_training_frames, fps=90)
#     webbrowser.open('training_video.mp4')

agent.local_qnetwork.load_state_dict(torch.load('checkpoint.pth', map_location=agent.device))
    
# Visualizing the results
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, _, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)
    
show_video_of_model(agent, 'LunarLander-v3')
webbrowser.open('video.mp4')