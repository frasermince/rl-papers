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
from pytorch_lightning.callbacks import LearningRateMonitor
# env = gym.make("Pong-v0")
# observation = env.reset()
# for _ in range(10000):
#   env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)

#   if done:
#     observation = env.reset()
# env.close()
#torch.set_printoptions(profile="full")
class Memory:
    def __init__(self, length):
        self.length = length
        self.items = []

    def append(self, item):
        self.items.append(item)
        if (len(self.items) > self.length):
            self.items.pop(0)

    def sample(self, n):
        return random.sample(self.items, n)

class GameNet(nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.hidden_linear = nn.Linear(32 * 9 * 9, 256)
        self.output_layer = nn.Linear(256, action_count)

    def forward(self, x):
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
        self.lr = 0.001
        self.observation = self.env.reset()
        self.epsilon = 1.0
        self.discount_factor = 0.85
        self.network = GameNet(self.env.action_space.n)

    def forward(self, observation):
        result = self.network(observation).to(dtype=torch.float64)
        #print("RESULT", result)
        return torch.max(result, 1)

    def training_step(self, data, batch_idx):
        (observation, reward, is_done, next_observation) = data
        with torch.no_grad():
            predicted_reward, action = self(next_observation)
            yj = reward + ((1 - is_done.long()) * (self.discount_factor * predicted_reward))
        predicted_reward, action = self(observation)
        loss = nn.MSELoss()(yj, predicted_reward)
        print("LOSS", loss)

        return loss

    def find_new_history(self, observation):
        self.history_frames.append(observation)
        if self.history_frames.length > 4:
            self.history_frames.pop(0)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1)
        return [optimizer], [scheduler]


    def play_step(self):
        self.env.render()
        first_observation = torch.tensor(self.observation.__array__(np.float32))
        first_observation = first_observation.permute(3, 0, 1, 2)
        if random.uniform(0, 1) <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            _, action = self(first_observation)

        second_observation, reward, is_done, info = self.env.step(action)
        first_observation = torch.squeeze(first_observation)
        second_observation_collated = torch.squeeze(torch.tensor(second_observation.__array__(np.float32)))
        state_tuple = (first_observation, reward, is_done, second_observation_collated)
        print("EPSILON", self.epsilon)
        if self.epsilon > 0.1:
            self.epsilon -= 0.0000009
        if is_done:
            self.observation = self.env.reset()
        self.memory.append(state_tuple)
        self.observation = second_observation


    def train_batch(self):
        while(True):
            self.play_step()
            tuples = self.memory.sample(32)
            for t in tuples:
                yield t

    def train_dataloader(self):
        self.dataset = ExperienceSourceDataset(self.train_batch)
        self.memory = Memory(1000000)
        for i in range(128):
            self.play_step()

        return DataLoader(dataset=self.dataset, batch_size=32)

lr_monitor = LearningRateMonitor(logging_interval='step')
Trainer(callbacks=[lr_monitor]).fit(DQN(0.01))
