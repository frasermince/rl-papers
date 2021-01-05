import gym
import gym.wrappers as wrappers
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule
import random
import numpy as np
from pl_bolts.datamodules.experience_source import Experience, ExperienceSourceDataset
from torch.utils.data import DataLoader
# env = gym.make("Pong-v0")
# observation = env.reset()
# for _ in range(10000):
#   env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)

#   if done:
#     observation = env.reset()
# env.close()

class GameNet(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.hidden_linear = nn.Linear(32 * 9 * 9, 256)
        self.output_layer = nn.Linear(256, action_count)

    def forward(self, x):
        print("SHAPE", x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.hidden_linear(x))
        return self.output_layer(x)

class DQN(LightningModule): 
    def __init__(self, lr):
        super().__init__()
        self.env = gym.make("Pong-v0")
        self.env = wrappers.FrameStack(wrappers.ResizeObservation(wrappers.GrayScaleObservation(self.env), 84), 4)
        #print("space", self.env.observation_space)
        self.lr = 0.01
        self.observation = self.env.reset()
        self.epsilon = 0.05
        self.discount_factor = 0.9
        self.network = GameNet(self.env.action_space.n)

    def forward(self, observation):
        return torch.max(self.network(observation), 1)

    def training_step(self, data, batch_idx):
        (observation, action, reward, is_done, next_observation) = data
        if is_done:
            yj = reward
        else:
            with torch.no_grad():
                predicted_reward, action = self(next_observation)
                yj = reward + (self.discount_factor * predicted_reward)
        #print("YJ", yj)
        predicted_reward, action = self(observation)
        loss = yj - predicted_reward

        return loss

    def find_new_history(self, observation):
        self.history_frames.append(observation)
        if self.history_frames.length > 4:
            self.history_frames.pop(0)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def train_batch(self):
        self.env.render()
        first_observation = torch.tensor(self.observation.__array__(np.float32))
        print("FIRST SPACE", first_observation.shape)
        first_observation = first_observation.permute(3, 0, 1, 2)
        if random.uniform(0, 1) <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            _, action = self(first_observation)

        second_observation, reward, is_done, info = self.env.step(action)
        first_observation = torch.squeeze(first_observation)
        second_observation_collated = torch.squeeze(torch.tensor(second_observation.__array__(np.float32)))
        print("OBS SPACE", second_observation.shape)
        #second_observation = second_observation.permute(3, 0, 1, 2)
        #print("OBS SPACE", first_observation.shape)
        state_tuple = (first_observation, action, reward, is_done, second_observation_collated)
        yield state_tuple
        self.observation = second_observation

    def train_dataloader(self):
        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=1)

Trainer().fit(DQN(0.01))
