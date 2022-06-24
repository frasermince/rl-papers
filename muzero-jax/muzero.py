import jax.numpy as jnp
import gym
import gym.wrappers as wrappers
from jax import random
from absl import flags, app
import sys
from experiment import MuzeroExperiment
from jaxline import platform

from model import MuZeroNet
from experience_replay import MuZeroMemory
from self_play import play_game  

# class MuZero(LightningModule):
#     def __init__(self, use_gpus=True, learning_rate=1e-4, env_name="Pong", normalize_advantages=False, batch_size=128, rollout_size=5):
#         super().__init__()
#         self.normalize_advantages = normalize_advantages
#         self.game_reward = 0
#         self.eps = np.finfo(np.float32).eps.item()
#         self.environment_name = env_name
#         self.steps_to_train = 4
#         self.rollout_size = 5
#         if self.environment_name == "Pong":
#         # env = gym.make("Pong-v4")
#             self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array", full_action_space=True)
#             # self.env = wrappers.FrameStack(wrappers.ResizeObservation(self.env, 96), 4)
#             self.env = wrappers.ResizeObservation(self.env, 96)
#         elif self.environment_name == "CartPole":
#             self.env = gym.make("CartPole-v1")

#         #   env = wrappers.Monitor(env, "~/Programming/recordings", video_callable=False ,force=True)
#         self.action_count = self.env.action_space.n
#         self.discount_factor = 0.99
#         #print("DEVICE", self.device)
#         self.network = MuZeroNet(self.device)
#         self.target_network = MuZeroNet(self.device)
#         self.reward = 0
#         self.games = 0
#         self.epoch_length = 20000
#         self.batch_size = batch_size
#         self.lr_init = 0.05
#         self.lr_decay_rate = 0.1
#         self.lr_decay_steps = 350e3
#         self.n_step = 10
#         self.last_ten = Memory(10)
#         self.last_hundred = Memory(100)
#         self.memory = MuZeroMemory(125000, rollout_size=self.rollout_size)
#         initial_observation = self.env.reset()

#         initial_observation = torch.tensor(initial_observation)
#         starting_policy = torch.ones(18) / 18
#         for i in range(32):
#             self.memory.append(initial_observation, 0, starting_policy, torch.tensor(1.0).unsqueeze(0), torch.tensor(0).unsqueeze(0))
#         self.entropy_scaling = 0.01
#         self.automatic_optimization = False
#         self.target_update_rate = 500

#     def forward(self, actions, net, observations=None, hidden_state=None):
#         if observations != None:
#           policy, value, reward, hidden_state = net(observations = observations.to(self.device), actions = actions.to(self.device))
#         elif hidden_state != None:
#           policy, value, reward, hidden_state = net(hidden_state = hidden_state.to(self.device), actions = actions.to(self.device))
#         return policy, value, reward, hidden_state

#     def test_step(self, data, x):
#         time.sleep(0.01)
  
#     def training_step(self, data, batch_idx):
#         #print(self.device)
#         opt = self.optimizers()
#         opt.zero_grad()
#         observations, actions, search_policy, search_value, simulator_reward, indices, priorities = data
#         #print(actions)
#         support_value = scalar_to_support(search_value, self.device)
#         support_reward = scalar_to_support(simulator_reward, self.device)
#         sch = self.lr_schedulers()

#         cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
#         self.log("lr", cur_lr, prog_bar=True, on_step=True)
#         self.log("game_reward", self.game_reward, prog_bar=True, on_step=True)
#         self.log("last_ten_reward", self.last_ten.average(),
#                   prog_bar=True, on_step=True)
#         self.log("last_hundred_reward", self.last_hundred.average(),
#                   prog_bar=True, on_step=True)
#         print(actions)
#         #print("ACTIONS SIZE", actions.shape)
#         values = []
#         policies = []
#         rewards = []
#         value, policy, reward, hidden_state = self(observations=observations, actions=actions[:, 0, :].float(), net=self.target_network)
#         values.append(value)
#         policies.append(policy)
#         rewards.append(reward)
#         policy_loss = 0
#         value_loss = 0
#         reward_loss = 0
#         for i in range(self.rollout_size):
#           value, policy, reward, hidden_state = self(hidden_state=hidden_state, actions=actions[:, i + 1, :].float(), net=self.target_network)
#           hidden_state.register_hook(lambda grad: grad * 0.5)
#           values.append(value)
#           policies.append(policy)
#           rewards.append(reward)
#         #print("SHAPE", support_value.shape, support_reward.shape)
#         for i in range(len(values)):
#           #print(rewards[i].squeeze().shape, support_reward.shape)
#           current_policy_loss = nn.CrossEntropyLoss(reduction="none")(policies[i], search_policy.permute(1, 0, 2)[i, :, :])
#           current_value_loss = nn.CrossEntropyLoss(reduction="none")(values[i].squeeze(), support_value[:, i, :])
#           current_reward_loss = nn.CrossEntropyLoss(reduction="none")(rewards[i], support_reward[:, i, :])
#           current_value_loss.register_hook(
#             lambda grad: grad / self.rollout_size
#           )
#           current_reward_loss.register_hook(
#             lambda grad: grad / self.rollout_size
#           )
#           current_policy_loss.register_hook(
#             lambda grad: grad / self.rollout_size
#           )

#           value_loss += current_value_loss
#           reward_loss += current_reward_loss
#           policy_loss += current_policy_loss

#         loss = policy_loss + value_loss + reward_loss
#         loss /= (self.memory.item_count() * priorities)
#         loss = loss.mean()
#         self.log("loss", loss, prog_bar=True, on_step=True)
#         self.log("play_step", self.steps, prog_bar=True, on_step=True)
#         self.manual_backward(loss)
#         opt.step()
#         with torch.no_grad():
#           #print("BEFORE", torch.stack(values).shape)
#           scalar_values = support_to_scalar(torch.stack(values).permute(1, 0, 2), self.device)
#           #print("AFTER", scalar_values.shape)
#           #print("BEFORE SEARCH SCALAR", search_value.shape, search_value)
#           #print("AFTER SEARCH SCALAR", search_scalar_values)
#           #print("AFTER SCALAR", scalar_values)
#           #print(torch.stack(search_value).shape, scalar_values.shape)
#           value_difference = torch.abs(search_value - scalar_values.to(self.device))
#           #print("DIFFERENCE", value_difference)
#           #TODO make sure this is right
#           self.memory.update_priorities(value_difference[:, -1], indices[-1], self.device)
#         sch.step()

#         if self.global_step % self.target_update_rate == 0:
#             self.target_network.load_state_dict(self.network.state_dict())

#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr_init)
#         decay_lambda = lambda step:  self.lr_init * self.lr_decay_rate ** (
#             step / self.lr_decay_steps
#         )
#         scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[decay_lambda])
#         return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}#{"scheduler": scheduler, "optimizer": optimizer}
 
#     def transform_observation(self, observation):
#       observation = torch.tensor(observation.__array__(np.float32))
#     #   if self.environment_name == "Pong":
#     #     observation = observation.permute(3, 0, 1, 2)
#       return observation

#     def prepare_for_batch(self, observation):
#         observation = torch.tensor(observation.__array__(np.float32))
#         # if self.environment_name == "Pong":
#         #     observation = observation.permute(3, 0, 1, 2)
#         return torch.squeeze(self.transform_observation(observation))

#     def iterate_batch(self):
#         i = 0
#         while(i < self.epoch_length):
#             i+= 1
#             self.steps = i
#             self.play_step()

#             if (self.memory.item_count() > self.batch_size + 32 + self.rollout_size + self.n_step and i % 2 == 0):
#                 memories = self.memory.sample(self.batch_size)
#                 for memory in memories:
#                     yield (
#                         torch.stack(memory["observations"]),
#                         torch.tensor(memory["actions"]),
#                         torch.stack(memory["policies"]),
#                         torch.tensor(memory["values"]),
#                         torch.tensor(memory["rewards"]),
#                         memory["indices"],
#                         memory["priority"],
#                     )

#     def test_iterated_batch(self):
#         for i in range(int(self.epoch_length)):
#             self.envs[0].render()
#             observation = self.transform_observation(self.observations[0])
#             policy, _ = self(observation, self.network)
#             action = torch.argmax(policy.logits)
#             new_observation, reward, is_done, info = self.envs[0].step(action.item())
#             if is_done:
#                 self.observations[0] = self.envs[0].reset()
#             else:
#                 self.observations[0] = new_observation
#             yield (action)


#     def training_epoch_end(self, outputs):
#         if self.games != 0:
#             self.logger.experiment.add_scalar(
#                 "average_game_reward", self.reward / self.games, self.current_epoch)
#         self.logger.experiment.add_scalar(
#             "total_game_reward", self.reward, self.current_epoch)
#         self.logger.experiment.add_scalar(
#             "game_count", self.games, self.current_epoch)
#         self.games = 0
#         self.reward = 0

#     def test_dataloader(self):
#         return self.dataloader(self.test_iterated_batch)

#     def train_dataloader(self):
#         return self.dataloader(self.iterate_batch)

#     def dataloader(self, batch_iterator):
#         self.dataset = ExperienceSourceDataset(batch_iterator)
#         return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

def main(argv, experiment_class):
    # rollout_size = 5
    # key = random.PRNGKey(0)
    # network = MuZeroNet()
    # memory = MuZeroMemory(125000, rollout_size=rollout_size)
    # key, representation_params, dynamics_params, prediction_params = network.initialize_networks(key)
    # params = (representation_params, dynamics_params, prediction_params)
    # game_memory = MuZeroMemory(20000, rollout_size=rollout_size)
    # env = gym.make("ALE/Pong-v5", render_mode="human", full_action_space=True)
    # env = wrappers.ResizeObservation(env, 96)
    # initial_observation = env.reset()
    # initial_observation = jnp.array(initial_observation)
    # starting_policy = jnp.ones(18) / 18
    # for _ in range(32):
    #     memory.append((initial_observation, 0, starting_policy, jnp.broadcast_to(jnp.array(1.0), (1)), jnp.broadcast_to(jnp.array(0), (1))))
    #     game_memory.append((initial_observation, 0, starting_policy, jnp.broadcast_to(jnp.array(1.0), (1)), jnp.broadcast_to(jnp.array(0), (1))))

    # def self_play(key, params, memory, env, starting_memory):
    #     key, game_buffer = play_game(key, params, memory, env, starting_memory)
    #     memory.append_multiple(game_buffer)
    #     return key

    # self_play(key, params, memory, env, game_memory)
    # print(memory)

    flags.mark_flag_as_required('config')
    print(sys.argv[1])
    platform.main(experiment_class, argv)

app.run(lambda argv: main(argv, MuzeroExperiment))