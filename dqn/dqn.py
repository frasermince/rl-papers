import gym
import gym.wrappers as wrappers

# import torch.nn as nn
# import torch
# import torch.nn.functional as F

# from pytorch_lightning import Trainer, LightningModule

# import random
# import numpy as np

# from pl_bolts.datamodules.experience_source import Experience, ExperienceSourceDataset
# from torch.utils.data import DataLoader
# from pytorch_lightning.callbacks import LearningRateMonitor
# import sys
# import time

# class Memory:
#     def __init__(self, length):
#         self.length = length
#         self.items = []

#     def append(self, item):
#         self.items.append(item)
#         if (len(self.items) > self.length):
#             self.items.pop(0)

#     def sample(self, n):
#         return random.sample(self.items, n)

#     def average(self):
#         total = 0
#         for i in self.items:
#             total += i
#         return total / self.length

# class StandardGameNet(nn.Module):
#     def __init__(self, action_count, use_gpus):
#         super().__init__()
#         #print("ACTION COUNT", action_count)
#         self.use_gpus = use_gpus
#         self.conv_layers = nn.Sequential(
#                 nn.Conv2d(4, 32, kernel_size=8, stride=4),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
#                 nn.ReLU(),
#         )
#         self.fully_connected = nn.Sequential(
#                 nn.Linear(64 * 7 * 7, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, action_count)

#         )

#     def forward(self, x):
#         if self.use_gpus:
#             x = x.cuda()
#         x = self.conv_layers(x / 255)
#         x = x.view(-1, 64 * 7 * 7)
#         return self.fully_connected(x)

# class DuelingGameNet(nn.Module):
#     def __init__(self, action_count, use_gpus):
#         super().__init__()
#         #print("ACTION COUNT", action_count)
#         self.use_gpus = use_gpus
#         self.conv_layers = nn.Sequential(
#                 nn.Conv2d(4, 32, kernel_size=8, stride=4),
#                 nn.ReLU(),
#                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
#                 nn.ReLU(),
#         )
#         self.advantage = nn.Sequential(
#                 nn.Linear(64 * 7 * 7, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, action_count)

#         )
#         self.value = nn.Sequential(
#                 nn.Linear(64 * 7 * 7, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 1)

#         )

#     def forward(self, x):
#         if self.use_gpus:
#             x = x.cuda()
#         x = self.conv_layers(x / 255)
#         x = x.view(-1, 64 * 7 * 7)
#         value = self.value(x)
#         advantage = self.advantage(x)
#         return value + (advantage - advantage.mean(dim=1, keepdim=True))

# class DQN(LightningModule):
#     def __init__(self, variant="dueling", use_gpus=False, learning_rate= 0.00025):
#         super().__init__()
#         #self.env = gym.make("Pong-v0")
#         self.env = gym.make("CartPole-v1")
#         self.env = wrappers.FrameStack(wrappers.ResizeObservation(wrappers.GrayScaleObservation(self.env), 84), 4)
#         self.observation = self.env.reset()
#         self.epsilon = 1
#         self.discount_factor = 0.99
#         self.network = DuelingGameNet(self.env.action_space.n, use_gpus) if variant == "dueling" else StandardGameNet(self.env.action_space.n, use_gpus)
#         self.target_network = DuelingGameNet(self.env.action_space.n, use_gpus) if variant == "dueling" else StandardGameNet(self.env.action_space.n, use_gpus)
#         self.target_update_rate = 1000
#         self.reward = 0
#         self.game_reward = 0
#         self.games = 0
#         self.epoch_length = 50000
#         self.learning_rate = learning_rate
#         self.steps_to_train = 4
#         self.last_ten = Memory(10)
#         self.variant = variant

#     def forward(self, observation, net):
#         return net(observation).to(dtype=torch.float64)

#     def test_step(self, batch, x):
#         time.sleep(.05)
#         return batch

#     def training_step(self, data, batch_idx):
#         self.log("reward", self.reward, prog_bar=True, on_step=True)
#         self.log("games", self.games, prog_bar=True, on_step=True)
#         self.log("epsilon", self.epsilon, prog_bar=True, on_step=True)
#         self.log("lr", self.learning_rate, prog_bar=True, on_step=True)
#         self.log("game_reward", self.game_reward, prog_bar=True, on_step=True)
#         self.log("last_ten_reward", self.last_ten.average(), prog_bar=True, on_step=True)

#         (observation, reward, actions, is_done, next_observation) = data
#         with torch.no_grad():
#             if self.variant == "standard":
#                 value_result = self(next_observation, self.target_network)
#                 predicted_reward, action = torch.max(value_result, 1)
#                 predicted_reward = predicted_reward.detach()
#             else:
#                 value_result = self(next_observation, self.network)
#                 value_actions = torch.max(value_result, 1)[1].unsqueeze(-1)
#                 target_result = self(next_observation, self.target_network)
#                 predicted_reward = target_result.gather(1, value_actions).squeeze(-1)
#                 predicted_reward = predicted_reward.detach()
#         yj = reward + ((1 - is_done.long()) * (self.discount_factor * predicted_reward))
#         current_predicted_reward = self(observation, self.network).gather(1, actions.unsqueeze(-1)).squeeze(-1)
#         if self.global_step % self.target_update_rate == 0:
#             self.target_network.load_state_dict(self.network.state_dict())
#         loss = nn.MSELoss()(current_predicted_reward, yj)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
#         #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1)
#         return optimizer


#     def play_step(self, epsilon):
#         self.env.render()
#         first_observation = torch.tensor(self.observation.__array__(np.float32))
#         first_observation = first_observation.permute(3, 0, 1, 2)
#         if random.uniform(0, 1) <= epsilon:
#             action = self.env.action_space.sample()
#         else:
#             values, action = torch.max(self(first_observation, self.network), 1)
#             action = action.item()
#             #print("ACTION", action)

#         second_observation, reward, is_done, info = self.env.step(action)
#         self.reward += reward
#         if reward > 0:
#           self.game_reward += reward
#         state_tuple = (self.observation, reward, action, is_done, second_observation)
#         if is_done:
#             self.last_ten.append(self.game_reward)
#             self.logger.experiment.add_scalar("game_score", self.game_reward, self.games)
#             self.observation = self.env.reset()
#             self.game_reward = 0
#             self.games += 1

#         if epsilon != 0:
#             self.memory.append(state_tuple)
#         self.observation = second_observation


#     def prepare_for_batch(self, observation):
#       observation = torch.tensor(observation.__array__(np.float32))
#       observation = observation.permute(3, 0, 1, 2)
#       return torch.squeeze(observation)

#     def test_batch(self):
#         i = 0
#         while(i < self.epoch_length):
#             i+= 1
#             self.play_step(0)
#             if i % self.steps_to_train == 0:

#                 if self.epsilon > 0.1:
#                     self.epsilon -= 0.000009
#                 tuples = self.memory.sample(32 * self.steps_to_train)
#                 # print("TUPLES", tuples)
#                 for t in tuples:
#                     (first_observation, reward, action, is_done, second_observation) = t
#                     yield (self.prepare_for_batch(first_observation), reward, action, is_done, self.prepare_for_batch(second_observation))

#     def train_batch(self):
#         i = 0
#         while(i < self.epoch_length):
#             i+= 1
#             self.play_step(self.epsilon)
#             if i % self.steps_to_train == 0:

#                 if self.epsilon > 0.1:
#                     self.epsilon -= 0.000009
#                 tuples = self.memory.sample(32 * self.steps_to_train)
#                 # print("TUPLES", tuples)
#                 for t in tuples:
#                     (first_observation, reward, action, is_done, second_observation) = t
#                     yield (self.prepare_for_batch(first_observation), reward, action, is_done, self.prepare_for_batch(second_observation))

#     def training_epoch_end(self, outputs):
#         if self.games != 0:
#             self.logger.experiment.add_scalar("average_game_reward", self.reward / self.games, self.current_epoch)
#         self.logger.experiment.add_scalar("total_game_reward", self.reward, self.current_epoch)
#         self.logger.experiment.add_scalar("game_count", self.games, self.current_epoch)
#         self.games = 0
#         self.reward = 0

#     def test_dataloader(self):
#         return self.dataloader(self.test_batch, 32 * self.steps_to_train)

#     def train_dataloader(self):
#         return self.dataloader(self.train_batch, 10000)

#     def dataloader(self, batch_iterator, pre_steps):
#         self.dataset = ExperienceSourceDataset(batch_iterator)
#         self.memory = Memory(500000)
#         for i in range(pre_steps):
#             self.play_step(self.epsilon)

#         return DataLoader(dataset=self.dataset, batch_size=32 * self.steps_to_train)
# # standard, dueling, or double
# use_gpus = False
# lightning_module = DQN("dueling", use_gpus)
# if len(sys.argv) > 2:
#     epochs = int(sys.argv[2])
# else:
#     epochs = 40

# if len(sys.argv) > 3:
#     refresh_rate = int(sys.argv[3])
# else:
#     refresh_rate = 50



# if use_gpus:
#     trainer = Trainer(progress_bar_refresh_rate=refresh_rate, max_epochs=epochs, gpus=1)
# else:
#     trainer = Trainer(progress_bar_refresh_rate=refresh_rate, max_epochs=epochs)
# stage = sys.argv[1]
# if stage == "train":
#     trainer.fit(lightning_module)
#     trainer.save_checkpoint("final.ckpt")
# elif stage == "test":
#     lightning_module = lightning_module.load_from_checkpoint("./dqn/dueling.ckpt")
#     trainer.test(lightning_module)
