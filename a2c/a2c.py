import gym
import gym.wrappers as wrappers
import torch.nn as nn
import torch
from pytorch_lightning import Trainer, LightningModule
import random
import numpy as np
from pl_bolts.datamodules.experience_source import ExperienceSourceDataset
from torch.utils.data import DataLoader
import time
from torch.distributions import Categorical
import inspect

def assert_equals(a, b):
  assert a == torch.Size(b), f"{a} != {b}"

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

    def items(self):
        return self.items

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
            nn.Softmax(),
        )
        self.value = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)

        )

    def forward(self, x):
        if self.use_gpus:
            x = x.cuda()
        # x = self.ln1(x)
        x = self.entry(x)
        value = self.value(x)
        policy = self.policy(x)
        return (Categorical(policy), value)

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
            nn.Softmax(),
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
        policy = Categorical(policy)
        return (policy, value)


class A2C(LightningModule):
    def __init__(self, use_gpus=True, learning_rate=1e-3, n_steps=5, env_name="Pong", environment_count=8, normalize_advantages=False):
        super().__init__()
        self.normalize_advantages = normalize_advantages
        self.environment_count = environment_count
        self.envs = []
        self.observations = []
        self.game_reward = []
        self.eps = np.finfo(np.float32).eps.item()
        self.environment_name = env_name
        for i in range(self.environment_count):
          if self.environment_name == "Pong":
            # env = gym.make("Pong-v4")
            env = gym.make("Breakout-v0")
            env = wrappers.FrameStack(wrappers.ResizeObservation(
                wrappers.GrayScaleObservation(env), 84), 4)
          elif self.environment_name == "CartPole":
            env = gym.make("CartPole-v1")

        #   env = wrappers.Monitor(env, "~/Programming/recordings", video_callable=False ,force=True)
          self.envs.append(env) 
          self.observations.append(env.reset())
          self.action_count = env.action_space.n
          self.game_reward.append(0)
        self.n_steps = n_steps
        self.discount_factor = 0.99
        if self.environment_name == "Pong":
            self.network = A2CNet(self.action_count, False)
        elif self.environment_name == "CartPole":
            self.network = A2CNetCartPole(self.action_count, False)
        self.reward = 0
        self.games = 0
        self.epoch_length = 20000
        self.learning_rate = learning_rate
        self.last_ten = Memory(10)
        self.last_hundred = Memory(100)
        self.entropy_scaling = 0.01
        self.automatic_optimization = False

    def forward(self, observation, net):
        policy, value = net(observation)
        return policy, value

    def test_step(self, data, x):
        time.sleep(0.01)

    def training_step(self, data, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        q_vals, observations, actions = data
        sch = self.lr_scheduler()
        actions = actions.squeeze()
        assert_equals(observations.squeeze().shape, [self.n_steps * self.environment_count, 4, 84, 84])
        policy, values = self(observations.squeeze(), self.network)
        values = values.squeeze()

        advantages = q_vals - values.squeeze()
        if self.normalize_advantages:
            with torch.no_grad():
                # print("first", advantages)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # print("second", advantages)

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        self.log("game_reward", sum(self.game_reward) / len(self.game_reward), prog_bar=True, on_step=True)
        self.log("last_ten_reward", self.last_ten.average(),
                  prog_bar=True, on_step=True)
        self.log("last_hundred_reward", self.last_hundred.average(),
                  prog_bar=True, on_step=True)
        print("STATE DICT", sch.state_dict())

        #print("QVALS", q_vals.shape)
        #print("observations", observations.shape)
        assert_equals(actions.shape, [self.n_steps * self.environment_count])
        log_probs = policy.log_prob(actions.squeeze())
        #print("ADVANTAGES", q_vals.dtype, values.dtype)
        #print("LOG_PROBS", log_probs)
        #print("ADVANTAGES", advantages)
        assert_equals(q_vals.shape, [self.n_steps * self.environment_count])
        assert_equals(values.shape, [self.n_steps * self.environment_count])
        assert_equals(log_probs.shape, [self.n_steps * self.environment_count])
        assert_equals(advantages.shape, [self.n_steps * self.environment_count])

        critic_loss = nn.MSELoss()(q_vals, values.squeeze())
        actor_loss = - (log_probs * advantages.detach()).mean()

        #print("ENTROPY", policy.entropy())
        entropy = policy.entropy().mean()
        #print("ENTROPY MEAN", entropy)
        loss = actor_loss + critic_loss - (self.entropy_scaling * entropy)
        self.log("entropy", entropy, prog_bar=True, on_step=True)
        self.log("actor_loss", actor_loss, prog_bar=True, on_step=True)
        self.log("critic_loss", critic_loss, prog_bar=True, on_step=True)
        self.log("entropy_coff", self.entropy_scaling * entropy, prog_bar=True, on_step=True)
        self.manual_backward(loss)
        print(inspect.getsource(sch.step))
        opt.step()
        sch.step()
        #print("STEP", self.epoch + batch_idx / self.epoch_length)
        #self.epoch + batch_idx / self.epoch_length)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, last_epoch=100)
        return [optimizer], [scheduler]#{"scheduler": scheduler, "optimizer": optimizer}

    def monte_carlo_play_step(self, env, environment_index):
    #   with torch.no_grad():
        rewards = torch.zeros(self.n_steps)
        actions = []
        observations = []
        done_indices = []
        in_progress_mask = torch.ones(self.n_steps + 1)
        for n_step_index in range(self.n_steps):
            # env.render()
            observation = self.transform_observation(self.observations[environment_index])
            observations.append(observation)

            policy, _ = self(observation, self.network)
            sampled_action = policy.sample().detach()
            actions.append(sampled_action)
            new_observation, reward, is_done, info = env.step(sampled_action.item())

            if reward > 0:
                self.game_reward[environment_index] += reward
                
            rewards[n_step_index] = reward
            if is_done:
                self.last_ten.append(self.game_reward[environment_index])
                self.last_hundred.append(self.game_reward[environment_index])
                self.logger.experiment.add_scalar(
                    "game_score", self.game_reward[environment_index], self.games)
                self.observations[environment_index] = env.reset()
                self.game_reward[environment_index] = 0
                self.games += 1
                done_indices.append(n_step_index)
                in_progress_mask[n_step_index] = 0
            else:
                self.observations[environment_index] = new_observation
            self.reward += reward

        observation = self.transform_observation(self.observations[environment_index])
        _, qval_for_bootstrap = self(observation, self.network)
        qval_for_bootstrap = qval_for_bootstrap.detach()

        q_vals = torch.zeros(len(rewards))
        next_value = qval_for_bootstrap
        #print("BOOTSTRAP", qval_for_bootstrap)
        #print("REWARDS SHAPE", rewards.shape, in_progress_mask.shape, q_vals.shape)
        for i in reversed(range(len(rewards))):
            next_value = rewards[i] + in_progress_mask[i + 1] * self.discount_factor * next_value
            q_vals[i] = next_value
            #print("REWARDS", i, rewards[i], in_progress_mask[i + 1], self.discount_factor, q_vals[i + 1], q_vals[i])
        # print("REWARDS", rewards, in_progress_mask, q_vals)
        #print(q_vals.shape, len(observations))
        #print("QVALS", q_vals)

        return (q_vals.detach(), observations, actions)
      #return (q_vals, observations, actions)

    def transform_observation(self, observation):
      observation = torch.tensor(observation.__array__(np.float32))
      if self.environment_name == "Pong":
        observation = observation.permute(3, 0, 1, 2)
      return observation

    def prepare_for_batch(self, observation):
        observation = torch.tensor(observation.__array__(np.float32))
        if self.environment_name == "Pong":
            observation = observation.permute(3, 0, 1, 2)
        return torch.squeeze(self.transform_observation(observation))

    def iterate_batch(self):
        for i in range(int(self.epoch_length)):
            for j in range(len(self.envs)):
              (q_vals, observations, actions) = self.monte_carlo_play_step(self.envs[j], j)
              for k in range(self.n_steps):
                  yield (q_vals[k], observations[k], actions[k])

    def test_iterated_batch(self):
        for i in range(int(self.epoch_length)):
            self.envs[0].render()
            observation = self.transform_observation(self.observations[0])
            policy, _ = self(observation, self.network)
            action = torch.argmax(policy.logits)
            new_observation, reward, is_done, info = self.envs[0].step(action.item())
            if is_done:
                self.observations[0] = self.envs[0].reset()
            else:
                self.observations[0] = new_observation
            yield (action)


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
        return self.dataloader(self.test_iterated_batch)

    def train_dataloader(self):
        return self.dataloader(self.iterate_batch)

    def dataloader(self, batch_iterator):
        self.dataset = ExperienceSourceDataset(batch_iterator)
        # for i in range(pre_steps):

        return DataLoader(dataset=self.dataset, batch_size=self.n_steps * self.environment_count)

# standard, dueling, or double
use_gpus = False
lightning_module = A2C(use_gpus=False, n_steps=5, env_name="Pong", environment_count=16)
# lightning_module = lightning_module.load_from_checkpoint("./epoch=21-step=1100000.ckpt").cpu()
if use_gpus:
    trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=100, gpus=1)
else:
    trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=100)
# trainer.tune(lightning_module)
trainer.fit(lightning_module)
# trainer.save_checkpoint("a2c.ckpt")
# trainer.test(lightning_module)
