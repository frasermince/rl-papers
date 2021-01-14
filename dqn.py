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
        #print("ACTION COUNT", action_count)
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        # self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        # self.conv2_bn = nn.BatchNorm2d(32)
        self.hidden_linear = nn.Linear(32 * 9 * 9, 256)
        self.output_layer = nn.Linear(256, action_count)

    def forward(self, x):
        x = F.relu(self.conv1(x / 255))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.hidden_linear(x))
        return self.output_layer(x)

class DQN(LightningModule):
    def __init__(self, learning_rate= 1e-4):
        super().__init__()
        self.env = gym.make("Pong-v0")
        self.env = wrappers.FrameStack(wrappers.ResizeObservation(wrappers.GrayScaleObservation(self.env), 84), 4)
        self.observation = self.env.reset()
        self.epsilon = 1
        self.discount_factor = 0.99
        self.network = GameNet(self.env.action_space.n)
        self.target_network = GameNet(self.env.action_space.n)
        self.target_update_rate = 1000
        self.reward = 0
        self.game_reward = 0
        self.games = 0
        self.epoch_length = 50000
        self.learning_rate = learning_rate

    def forward(self, observation, net):
        result = net(observation).to(dtype=torch.float64)
        m, action = torch.max(result, 1)
        return m, action

    def training_step(self, data, batch_idx):
        self.log("reward", self.reward, prog_bar=True, on_step=True)
        self.log("games", self.games, prog_bar=True, on_step=True)
        self.log("epsilon", self.epsilon, prog_bar=True, on_step=True)
        self.log("lr", self.learning_rate, prog_bar=True, on_step=True)
        self.log("game_reward", self.game_reward, prog_bar=True, on_step=True)

        (observation, reward, is_done, next_observation) = data
        with torch.no_grad():
            predicted_reward, action = self(next_observation, self.target_network)
        #print("IS ONGOING", 1 - is_done.long())
        #print("PREDICTED REWARD", ((1 - is_done.long()) * (self.discount_factor * predicted_reward)))
        #print("CURRENT REWARD", reward)
        yj = reward + ((1 - is_done.long()) * (self.discount_factor * predicted_reward))
        #print("VALUE", yj)
        predicted_reward, action = self(observation, self.network)
        if self.global_step % self.target_update_rate == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        loss = nn.MSELoss()(yj, predicted_reward)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1)
        return optimizer


    def play_step(self):
        self.env.render()
        first_observation = torch.tensor(self.observation.__array__(np.float32))
        first_observation = first_observation.permute(3, 0, 1, 2)
        if random.uniform(0, 1) <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            values, action = self(first_observation, self.network)
            action = action.item()
            #print("ACTION", action)

        second_observation, reward, is_done, info = self.env.step(action)
        self.reward += reward
        self.game_reward += reward
        state_tuple = (self.observation, reward, is_done, second_observation)
        if self.epsilon > 0.1:
            self.epsilon -= 0.0000009
        if is_done:
            self.logger.experiment.add_scalar("game_score", self.game_reward, self.games)
            self.observation = self.env.reset()
            self.game_reward = 0
            self.games += 1

        self.memory.append(state_tuple)
        self.observation = second_observation


    def prepare_for_batch(self, observation):
      observation = torch.tensor(observation.__array__(np.float32))
      observation = observation.permute(3, 0, 1, 2)
      return torch.squeeze(observation)

    def train_batch(self):
        i = 0
        while(i < self.epoch_length):
            i+= 1
            self.play_step()
            tuples = self.memory.sample(32)
            for t in tuples:
                (first_observation, reward, is_done, second_observation) = t
                yield (self.prepare_for_batch(first_observation), reward, is_done, self.prepare_for_batch(second_observation))

    def training_epoch_end(self, outputs):
        if self.games != 0:
            self.logger.experiment.add_scalar("average_game_reward", self.reward / self.games, self.current_epoch)
        self.logger.experiment.add_scalar("total_game_reward", self.reward, self.current_epoch)
        self.logger.experiment.add_scalar("game_count", self.games, self.current_epoch)
        self.games = 0
        self.reward = 0

    def train_dataloader(self):
        self.dataset = ExperienceSourceDataset(self.train_batch)
        self.memory = Memory(1000000)
        for i in range(10000):
            self.play_step()

        return DataLoader(dataset=self.dataset, batch_size=32)

lightning_module = DQN()#.load_from_checkpoint("./lightning_logs/version_82/checkpoints/epoch=26-step=1316096.ckpt")
trainer = Trainer(progress_bar_refresh_rate=50, max_epochs=40, auto_lr_find=True)
#trainer.tune(lightning_module)
trainer.fit(lightning_module)
