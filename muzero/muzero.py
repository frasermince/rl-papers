import gym
import gym.wrappers as wrappers
from importlib_metadata import DistributionFinder, distribution
import torch.nn as nn
import torch
from pytorch_lightning import Trainer, LightningModule
import random
import numpy as np
from pl_bolts.datamodules.experience_source import ExperienceSourceDataset
from torch.utils.data import DataLoader
import time
from torch.distributions import Categorical
import math

def assert_size(a, b):
  assert a == torch.Size(b), f"{a} != {b}"

def assert_equals(a, b):
  assert a == b, f"{a} != {b}"

def one_hot_encode(action, input_shape, action_output_shape, device):
    result = torch.zeros((input_shape[0], action_output_shape[0] * action_output_shape[1]), device=device)
    result = result.scatter_(1, action.unsqueeze(1), 1)
    assert_size(result.shape, (input_shape[0], action_output_shape[0] * action_output_shape[1]))
    result = result.view(input_shape[0], action_output_shape[0], action_output_shape[1])
    assert_size(result.shape, (input_shape[0], action_output_shape[0], action_output_shape[1]))
    # result[math.floor(action / size[0])][action % size[0]] = 1
    # one hot encode the action
    return result 
class Node():
    def __init__(self, policy, prediction_network, dynamics_network, device, simulations = 10):
        self.device = device
        self.simulations = simulations
        self.prediction_network = prediction_network
        self.dynamics_network = dynamics_network
        #self.state = state
        self.policy = policy
        self.visit_count = 0
        self.accumulated_value = 0
        self.reward = 0
        self.children = []
        self.discount_factor = 0.99
        self.c1 = 1.25
        self.c2 = 19652

    def value(self) -> float:
      if self.visit_count == 0:
        return 0
      return self.accumulated_value / self.visit_count

    def set_state(self, state, policy):
      self.state = state
      self.policy = policy

    def visit_policy(self):
      result = torch.zeros(18)
      for action, child in enumerate(self.children):
        result[action] = child.policy / self.visit_count

      return result.softmax(0)

    def initial_expand(self, state, policy):
      self.state = state
      #policy = policy.cpu()
      policy = {a: math.exp(policy[a]) for a in range(18)}
      policy_sum = sum(policy)
      for action, p in policy.items():
        self.children.append(Node(p / policy_sum, self.prediction_network, self.dynamics_network, self.device))

    def expand(self, action, state):
        action = torch.tensor(action, device=self.device).unsqueeze(0)
        action = one_hot_encode(action, action.shape, state.shape[2:], self.device)
        new_state, reward = self.dynamics_network(state, action)
        value, policy = self.prediction_network(state)
        policy = policy.squeeze()
        #policy = policy.cpu()
        self.state = new_state
        policy = {a: math.exp(policy[a]) for a in range(18)}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
          self.children.append(Node(p / policy_sum, self.prediction_network, self.dynamics_network, self.device))
        return reward, value

    def choose_child(self):
      results = ((self.children[i].ucb_count(self), self.children[i], i) for i in range(18))
      _, child, action = max(results)
      #print("RESULTS", results)
      #print("ACTION", results.argmax())
      return child, action
      
    def ucb_count(self, parent):
        upper_confidence_bound = self.policy * (math.sqrt(parent.visit_count) / (1 + self.visit_count)) * (self.c1 + math.log((parent.visit_count + self.c2 + 1) / self.c2))
        if self.visit_count > 0:
          upper_confidence_bound += self.value() + self.reward
        #print(self.value(), self.reward, self.policy, (math.sqrt(parent.visit_count) / (1 + self.visit_count)), (self.c1 + math.log((parent.visit_count + self.c2 + 1) / self.c2)))
        # import code; code.interact(local=dict(globals(), **locals()))
        return upper_confidence_bound

    def children_count(self):
        total = 0
        for _, child in self.children:
            # import code; code.interact(local=dict(globals(), **locals()))
            total += child.children_count() + 1
        return total

    def monte_carlo_tree_search(self, parent):
        # TODO: the following:
        # 1. Normalize Q values for UCB
        self.visit_count += 1
        if self.children != []:
            #print("RECURSE")
            child, action = self.choose_child()
            #print("ACTION", action, self.children[action])
            self.action = action
            # TODO set new current node
            previous_cumulative = child.monte_carlo_tree_search(self)
            cumulative_discounted_reward = self.reward + self.discount_factor * previous_cumulative
            #self.q_vals[action] = (self.visit_count[action] * self.q_vals[action] + cumulative_discounted_reward) / (self.visit_count[action] + 1)
            self.accumulated_value += self.discount_factor * previous_cumulative
            return cumulative_discounted_reward
        else:
            #print("BASE")
            reward, value = self.expand(parent.action, parent.state)
            # import code; code.interact(local=dict(globals(), **locals()))
            self.accumulated_value = value.item()
            self.reward = reward.item()
            return self.accumulated_value#, action
                
    def perform_simulations(self):
        for simulation in range(self.simulations):
            value = self.monte_carlo_tree_search(self)
        return self.visit_policy(), value
class MuZeroMemory:
    def __init__(self, length):
        self.length = length
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.policies = []

    def last_n(self, n):
        return {"observations": self.observations[-n:], "actions": self.actions[-n:], "rewards": self.rewards[-n:], "values": self.values[-n:], "policies": self.policies[-n:]}

    def append(self, observation, action, policy, value, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.policies.append(policy)
        if (len(self.observations) > self.length):
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.policies.pop(0)

    def sample(self, n):
        available_indices = list(enumerate(self.observations))
        indices = random.sample(available_indices[32:], n)
        samples = []
        for i, _ in indices:
            samples.append(
                {
                    "observations": self.observations[i - 32 : i],
                    "actions": self.actions[i - 32 : i],
                    "rewards": self.rewards[i - 32 : i],
                    "values": self.values[i - 32 : i],
                    "policies": self.policies[i - 32 : i],
                }
            )
        return samples

    def item_count(self):
        return len(self.observations)

class Memory:
    def __init__(self, length):
        self.length = length
        self.items = []

    def last_n(self, n):
        return self.items[-n:]

    def append(self, item):
        self.items.append(item)
        if (len(self.items) > self.length):
            self.items.pop(0)

    def sample(self, n):
        return random.sample(self.items, n)

    def item_count(self):
        return len(self.items)

    def items(self):
        return self.items

    def remove_first_item(self):
        self.items.pop(0)

    def average(self):
        total = 0
        for i in self.items:
            total += i
        return total / self.length

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.planes = planes
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        assert_equals(x.shape[1], self.inplanes)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        assert_equals(out.shape[1], self.planes)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# still need to folow the recomendations found on page 14 of the paper under the "Network Architecture" section under reference number 30
class PredictionNet(nn.Module):
    def __init__(self):
        super(PredictionNet, self).__init__()
        self.starting_layers = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 18),
            nn.Softmax(),
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.starting_layers(x)
        x = x.view(-1, 64 * 6 * 6)
        value = self.value(x)
        policy = self.policy(x)
        return (value, policy)

class RepresentationNet(nn.Module):
    def __init__(self):
        super().__init__()
        # directly from the paper
        self.first_residual_blocks = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            ResBlock(128, 128),
            # ResBlock(128, 128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            ResBlock(256, 256),
            nn.AvgPool2d(2, stride=2),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            ResBlock(256, 256),
            nn.AvgPool2d(2, stride=2),
        )

    def forward(self, inputs, actions):
        assert_size(inputs.shape, (inputs.shape[0], 32, 96, 96, 3))
        #print("FLATTEN and PERMUTE")
        inputs = inputs.permute(0, 4, 1, 2, 3).flatten(1, 2)
        #print("SCALE ACTIONS")
        actions = actions.unsqueeze(-1).unsqueeze(-1) / 18
        actions = actions.repeat(1, 1, 96, 96)
        #print("AFTER ACTION CHANGE")
        assert_size(actions.shape, (inputs.shape[0], 32, 96, 96))
        assert_size(inputs.shape, (inputs.shape[0], 96, 96, 96))
        #print(actions.shape, inputs.shape)
        #print("CAT INPUTS")
        inputs = torch.cat((actions.float(), inputs.float()), 1)
        assert_size(inputs.shape, (inputs.shape[0], 128, 96, 96))
        #print("FIRST RESIDUAL", inputs.shape)
        inputs = self.first_residual_blocks(inputs)
        #print("AFTER FIRST RESIDUAL", inputs.shape)
        #print("AFTER SECOND RESIDUAL")
        assert_size(inputs.shape, (inputs.shape[0], 256, 6, 6))
        #print("AFTER ASSERT", inputs.shape)
        return inputs

class DynamicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Paper calls for 16 blocks, but we use 10 to make it train faster
        # Currently assuming that we append the one hot encoded action to the planes of the hidden state
        self.residual_blocks = nn.Sequential(
            ResBlock(257, 256, downsample=nn.Conv2d(257, 256, 1, stride=1)),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            # ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.reward_conv_layers = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.reward_linear_layers = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, hidden_state, action):
        action = action.unsqueeze(1)
        assert_size(action.shape, (hidden_state.shape[0], 1, 6, 6))
        assert_size(hidden_state.shape, (hidden_state.shape[0], 256, 6, 6))
        inputs = torch.cat((action, hidden_state), dim=1)
        assert_size(inputs.shape, (hidden_state.shape[0], 257, 6, 6))
        new_hidden = self.residual_blocks(inputs)
        assert_size(new_hidden.shape, (hidden_state.shape[0], 256, 6, 6))
        # reward = self.reward_conv_layers1(new_hidden)
        reward = self.reward_conv_layers(new_hidden)
        reward = reward.view(-1, 64 * 6 * 6)
        reward = self.reward_linear_layers(reward).squeeze()
        return (new_hidden, reward)

class MuZeroNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.representation = RepresentationNet()
        self.dynamics = DynamicsNet()
        self.prediction = PredictionNet()

    def set_device(self, device):
      if (device != self.device):
        self.device = device

    def representation_net(self, inputs, actions):
        inputs = inputs.to(self.device)
        #print("ACTION SHAPE", actions.shape)
        actions = actions.to(self.device)
        #print("AFTER")
        return self.representation(inputs, actions)

    def dynamics_net(self, hidden_state, action):
        hidden_state = hidden_state.to(self.device)
        action = action.to(self.device)
        return self.dynamics(hidden_state, action)

    def prediction_net(self, hidden_state):
        hidden_state = hidden_state.to(self.device)
        return self.prediction(hidden_state)

    def forward(self, inputs, actions):
        inputs = inputs.to(self.device)
        actions = actions.to(self.device)
        hidden_state = self.representation(inputs, actions)
        value, policy = self.prediction(hidden_state)
        distribution = Categorical(policy)
        new_actions = distribution.sample().to(self.device)
        assert_size(new_actions.shape, [inputs.shape[0]])
        new_actions = one_hot_encode(new_actions, new_actions.shape, hidden_state.shape[2:], self.device)
        new_state, reward = self.dynamics_net(hidden_state, new_actions)
        return (value, policy, reward)


class MuZero(LightningModule):
    def __init__(self, use_gpus=True, learning_rate=1e-4, env_name="Pong", normalize_advantages=False, batch_size=128, render_mode="human"):
        super().__init__()
        self.normalize_advantages = normalize_advantages
        self.game_reward = 0
        self.eps = np.finfo(np.float32).eps.item()
        self.environment_name = env_name
        self.steps_to_train = 4
        if self.environment_name == "Pong":
        # env = gym.make("Pong-v4")
            # self.env = gym.make("ALE/Pong-v5", render_mode="human", full_action_space=True)
            self.env = gym.make("ALE/Pong-v5", render_mode=render_mode, full_action_space=True)
            # self.env = wrappers.FrameStack(wrappers.ResizeObservation(self.env, 96), 4)
            self.env = wrappers.ResizeObservation(self.env, 96)
        elif self.environment_name == "CartPole":
            self.env = gym.make("CartPole-v1")

        #   env = wrappers.Monitor(env, "~/Programming/recordings", video_callable=False ,force=True)
        self.action_count = self.env.action_space.n
        self.discount_factor = 0.99
        # print("DEVICE", self.device)
        self.network = MuZeroNet("cpu")
        self.target_network = MuZeroNet("cpu")
        self.reward = 0
        self.games = 0
        self.epoch_length = 20000
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.last_ten = Memory(10)
        self.last_hundred = Memory(100)
        self.memory = MuZeroMemory(500000)
        initial_observation = self.env.reset()

        initial_observation = torch.tensor(initial_observation)
        starting_policy = torch.ones(18) / 18
        for i in range(32):
            self.memory.append(initial_observation, 0, starting_policy, 1, 0)
        self.entropy_scaling = 0.01
        self.automatic_optimization = False
        self.target_update_rate = 1000

    def forward(self, observations, actions, net):
        policy, value, reward = net(observations.to(self.device), actions.to(self.device))
        return policy, value, reward

    def test_step(self, data, x):
        time.sleep(0.01)

    def training_step(self, data, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        observations, actions, search_policy, search_value, simulator_reward = data
        search_policy = search_policy[:, -2:-1, :].squeeze()
        sch = self.lr_schedulers()

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        self.log("game_reward", self.game_reward, prog_bar=True, on_step=True)
        self.log("last_ten_reward", self.last_ten.average(),
                  prog_bar=True, on_step=True)
        self.log("last_hundred_reward", self.last_hundred.average(),
                  prog_bar=True, on_step=True)

        value, policy, reward = self(observations, actions.float(), self.target_network)
        policy_loss = nn.MSELoss()(policy, search_policy)
        value_loss = nn.MSELoss()(value, search_value[:, -2:-1])
        reward_loss = nn.MSELoss()(reward, simulator_reward[:, -2:-1].squeeze())
        loss = policy_loss + value_loss + reward_loss
        #print("LOSS", loss.device, loss.dtype, loss)
        self.log("loss", loss, prog_bar=True, on_step=True)

        self.manual_backward(loss)
        opt.step()
        #sch.step(self.current_epoch + batch_idx / self.epoch_length)

        if self.global_step % self.target_update_rate == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, T_mult=2)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, last_epoch=100)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}#{"scheduler": scheduler, "optimizer": optimizer}
        #return [optimizer], [scheduler]#{"scheduler": scheduler, "optimizer": optimizer}
 
    def monte_carlo_tree_search(self, observation, action, network):
        # import code; code.interact(local=dict(globals(), **locals()))
        hidden_state = network.representation_net(observation, action)
        # new_state, reward = network.dynamics_net(hidden_state)
        value, policy = network.prediction_net(hidden_state)
    # def __init__(self, state, policy, prediction_network, dynamics_network):

        node = Node(0, network.prediction_net, network.dynamics_net, self.device)
        node.initial_expand(hidden_state, policy.squeeze())
        policy, value = node.perform_simulations()

            # select a random action
        # probs = torch.tensor([0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556, 0.05555555556 ])
        return policy, value

 
    def play_step(self):
        #self.network.set_device(self.device)
        #self.target_network.set_device(self.device)
        past_memory = self.memory.last_n(32)
        # import code; code.interact(local=dict(globals(), **locals()))
        past_observations = torch.stack(past_memory["observations"]).unsqueeze(0).to(self.device)
        # print("observations shape", past_observations.shape)
        past_actions = torch.tensor(past_memory["actions"]).to(self.device)

        policy, value = self.monte_carlo_tree_search(past_observations, past_actions, self.network)
        action = Categorical(policy).sample().item()

        second_observation, reward, is_done, info = self.env.step(action)
        self.reward += reward
        if reward > 0:
            self.game_reward += reward
        if is_done:
            self.last_ten.append(self.game_reward)
            self.logger.experiment.add_scalar("game_score", self.game_reward, self.games)
            self.observation = self.env.reset()
            self.game_reward = 0
            self.games += 1

        self.memory.append(torch.tensor(second_observation) / 255, action, policy, value, reward)
        self.observation = second_observation

    def transform_observation(self, observation):
      observation = torch.tensor(observation.__array__(np.float32))
    #   if self.environment_name == "Pong":
    #     observation = observation.permute(3, 0, 1, 2)
      return observation

    def prepare_for_batch(self, observation):
        observation = torch.tensor(observation.__array__(np.float32))
        # if self.environment_name == "Pong":
        #     observation = observation.permute(3, 0, 1, 2)
        print("PREP STEP", observation.shape)
        return torch.squeeze(self.transform_observation(observation))

    def iterate_batch(self):
        i = 0
        while(i < self.epoch_length):
            i+= 1
            self.play_step()
            if (self.memory.item_count() > self.batch_size + 32 and self.batch_size % 32 == 0):
                # TODO add prioritized experience replay. Read appendix G in the paper.
                memories = self.memory.sample(self.batch_size)
                for memory in memories:
                    #print(memory["observations"][0].device)
                    #print(memory["actions"])
                    #print(memory["policies"][0].device)
                    #print(memory["values"])
                    #print(memory["rewards"])
                    yield (
                        torch.stack(memory["observations"]),
                        torch.tensor(memory["actions"]),
                        torch.stack(memory["policies"]),
                        torch.tensor(memory["values"]),
                        torch.tensor(memory["rewards"])
                    )
                    # yield (self.prepare_for_batch(first_observation), reward, action, is_done, self.prepare_for_batch(second_observation))

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

        # TODO determine batch size
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)
device="CPU"
# lightning_module = lightning_module.load_from_checkpoint("./epoch=21-step=1100000.ckpt").cpu()
if device == "TPU":
    trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=100, accelerator="tpu", devices=1, callbacks=[ModelCheckpoint(dirpath="/content/drive/My Drive/colab")])
elif device == "GPU":
  trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=100, gpus=1, callbacks=[ModelCheckpoint(dirpath="/content/drive/My Drive/colab")])
else:
    trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=100)
lightning_module = MuZero(use_gpus=True, env_name="Pong", batch_size=64)
trainer.fit(lightning_module)
# trainer.save_checkpoint("a2c.ckpt")
# trainer.test(lightning_module)