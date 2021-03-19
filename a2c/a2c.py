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
import time
from torch.distributions import Categorical


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

    def average(self):
        total = 0
        for i in self.items:
            total += i
        return total / self.length

class A2CNetCartPole(nn.Module):
    def __init__(self, action_count, use_gpus):
        super().__init__()
        self.use_gpus = use_gpus
        self.ln1 = nn.LayerNorm(4)
        self.entry = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
            nn.ReLU(),
            nn.Softmax()
        )
        self.value = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)

        )

    def forward(self, x):
        if self.use_gpus:
            x = x.cuda()
        x = self.ln1(x)
        x = self.entry(x)
        value = self.value(x)
        policy = self.policy(x)
        return (policy, value)

class A2CNet(nn.Module):
    def __init__(self, action_count, use_gpus):
        super().__init__()
        self.use_gpus = use_gpus
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
            nn.ReLU(),
            nn.Softmax()
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)

        )

    def forward(self, x):
        if self.use_gpus:
            x = x.cuda()
        x = self.conv_layers(x / 255)
        x = x.view(-1, 64 * 7 * 7)
        value = self.value(x)
        policy = self.policy(x)
        return (policy, value)


class A2C(LightningModule):
    def __init__(self, use_gpus=False, learning_rate=0.001, n_steps=128):
        super().__init__()
        self.env = gym.make("CartPole-v1")
        # self.env = wrappers.FrameStack(wrappers.ResizeObservation(
            # wrappers.GrayScaleObservation(self.env), 84), 4)
        self.n_steps = n_steps
        self.observation = self.env.reset()
        self.discount_factor = 0.99
        self.action_count = self.env.action_space.n
        self.network = A2CNetCartPole(self.action_count, use_gpus)
        self.reward = 0
        self.game_reward = 0
        self.games = 0
        self.epoch_length = 50000
        self.learning_rate = learning_rate
        self.last_ten = Memory(10)
        self.entropy_scaling = 0.01

    def forward(self, observation, net):
        policy, value = net(observation)
        return policy.to(dtype=torch.float64), value.to(dtype=torch.float64)

    def test_step(self, batch, x):
        time.sleep(.05)
        return batch

    def training_step(self, data, batch_idx):
        # advantages, log_probs, entropy
        self.log("entropy", self.entropy, prog_bar=True, on_step=True)
        self.log("reward", self.reward, prog_bar=True, on_step=True)
        self.log("games", self.games, prog_bar=True, on_step=True)
        self.log("lr", self.learning_rate, prog_bar=True, on_step=True)
        self.log("game_reward", self.game_reward, prog_bar=True, on_step=True)
        self.log("last_ten_reward", self.last_ten.average(),
                 prog_bar=True, on_step=True)

        (q_vals, observations) = data

        policy, values = self(observations.squeeze(), self.network)
        log_probs = torch.log(policy)
        advantages = q_vals.detach() - values.squeeze()
        critic_loss = 0.5 * advantages.pow(2).mean()
        actor_loss = (advantages.detach() @ -log_probs).mean()
        # distribution = Categorical(policy)
        # entropy = distribution.entropy().mean()

        # loss = -log π( a | s) * (monte_carlo + bootstrap - predicted_value) + TD(monte_carlo + boostrap,  predicted_value)
        loss = actor_loss + critic_loss + self.entropy_scaling * self.entropy
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1)
        return optimizer

    def monte_carlo_play_step(self):
        i = 0
        qval_for_bootstrap = None
        rewards = []
        observations = []
        done_indices = []
        self.entropy = 0
        with torch.no_grad():
            while(True):
                self.env.render()
                first_observation = self.observation.__array__(np.float32)
                first_observation = torch.tensor(first_observation)
                # first_observation = first_observation.permute(3, 0, 1, 2)

                policy, value = self(first_observation, self.network)

                distribution = Categorical(policy)
                sampled_action = distribution.sample()
                self.entropy += distribution.entropy().mean()
                second_observation, reward, is_done, info = self.env.step(sampled_action.item())

                if is_done:
                    self.last_ten.append(self.game_reward)
                    self.logger.experiment.add_scalar(
                        "game_score", self.game_reward, self.games)
                    self.observation = self.env.reset()
                    self.game_reward = 0
                    self.games += 1
                    done_indices.append(i)

                if i == self.n_steps:
                    qval_for_bootstrap = value
                    break
                else:
                    self.reward += reward
                    rewards.append(reward)
                    observations.append(first_observation)

                    if reward > 0:
                        self.game_reward += reward

                    self.observation = second_observation
                    i += 1

            q_vals = torch.zeros(len(rewards))

            for i in reversed(range(len(rewards))):
                if i in done_indices:
                    qval_for_bootstrap = 0
                else:
                    qval_for_bootstrap = rewards[i] + self.discount_factor * qval_for_bootstrap
                q_vals[i] = qval_for_bootstrap

        return (q_vals.detach(), observations)

    def prepare_for_batch(self, observation):
        observation = torch.tensor(observation.__array__(np.float32))
        # observation = observation.permute(3, 0, 1, 2)
        return torch.squeeze(observation)

    def iterate_batch(self):
        for i in range(int(self.epoch_length)):
            (q_vals, observations) = self.monte_carlo_play_step()
            for j in range(len(q_vals)):
                yield (q_vals[j], observations[j])

    def training_epoch_end(self, outputs):
        if self.games != 0:
            self.logger.experiment.add_scalar(
                "average_game_reward", self.reward / self.games, self.current_epoch)
        self.logger.experiment.add_scalar(
            "total_game_reward", self.reward, self.current_epoch)
        self.logger.experiment.add_scalar(
            "game_count", self.games, self.current_epoch)
        self.games = 0
        self.reward = 0

    def test_dataloader(self):
        return self.dataloader(self.iterate_batch)

    def train_dataloader(self):
        return self.dataloader(self.iterate_batch)

    def dataloader(self, batch_iterator):
        self.dataset = ExperienceSourceDataset(batch_iterator)
        # for i in range(pre_steps):

        return DataLoader(dataset=self.dataset, batch_size=self.n_steps)


# standard, dueling, or double
use_gpus = False
lightning_module = A2C(use_gpus, n_steps=256)
if use_gpus:
    trainer = Trainer(progress_bar_refresh_rate=1, max_epochs=40, gpus=1)
else:
    trainer = Trainer(progress_bar_refresh_rate=1, max_epochs=40)
# trainer.tune(lightning_module)
trainer.fit(lightning_module)
trainer.save_checkpoint("a2c.ckpt")
# trainer.test(lightning_module)
