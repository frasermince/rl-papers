import jax.numpy as np
import jax.lax as lax
from jax import random
import jax
from model import scatter
from collections.abc import Sequence
from functools import partial

# TODO Add per game memory for chess and go

def game_memory_flatten(memory):
    return ((memory.observations, memory.actions, memory.rewards, memory.values, memory.policies, memory.priorities),
        (memory.rollout_size, memory.n_step, memory.discount_rate))

def game_memory_unflatten(aux, children):
    (observations, actions, rewards, values, policies, priorities) = children
    (rollout_size, n_step, discount_rate) = aux
    return GameMemory(observations=observations, actions=actions, rewards=rewards, values=values, policies=policies, priorities=priorities, n_step=n_step, discount_rate=discount_rate)

def self_play_flatten(memory):
    return ((memory.observations, memory.actions, memory.rewards, memory.values, memory.policies),
        (memory.games))

def self_play_unflatten(aux, children):
    (games) = aux
    memory = SelfPlayMemory(games)
    (observations, actions, rewards, values, policies) = children
    memory.observations = observations
    memory.actions = actions
    memory.rewards = rewards
    memory.values = values
    memory.policies = policies
    return memory


class SelfPlayMemory(Sequence):
    def __init__(self, games):
        self.games = games

    def populate(self):
        self.observations = np.zeros((self.games, 232, 96, 96, 3))
        self.actions = np.zeros((self.games, 232, 1))
        self.rewards = np.zeros((self.games, 232, 1))
        self.values = np.zeros((self.games, 232, 1))
        self.policies = np.zeros((self.games, 232, 18))

    def set_steps(self, i, data):
        all_steps, finished_steps = data
        return all_steps.at[finished_steps[i]].set(32)

    @jax.jit
    def output_game_buffer(self, finished_steps, all_steps, starting_observations):
        finished_steps = finished_steps[0 : 8]
        finished_buffer = SelfPlayMemory(self.games)
        finished_buffer.observations = jax.vmap(lambda index: self.observations[index])(finished_steps)
        finished_buffer.actions = jax.vmap(lambda index: self.actions[index])(finished_steps)
        finished_buffer.rewards = jax.vmap(lambda index: self.rewards[index])(finished_steps)
        finished_buffer.values = jax.vmap(lambda index: self.values[index])(finished_steps)
        finished_buffer.policies = jax.vmap(lambda index: self.policies[index])(finished_steps)
        cpus = jax.devices("cpu")
        finished_buffer = jax.device_put(finished_buffer, cpus[0])

        self.observations = self.observations.at[finished_steps].set(starting_observations[0])
        self.actions = self.actions.at[finished_steps].set(0)
        self.rewards = self.rewards.at[finished_steps].set(0)
        self.values = self.observations.at[finished_steps].set(0)
        self.policies = self.policies.at[finished_steps].set(0)

        # self.actions = vmap(lambda index: self.actions[index])(finished_steps)
        # self.rewards = vmap(lambda index: self.rewards[index])(finished_steps)
        # self.values = vmap(lambda index: self.values[index])(finished_steps)
        # self.policies = vmap(lambda index: self.policies[index])(finished_steps)

        for i in range(8):
            all_steps = self.set_steps(i, (all_steps, finished_steps))
        # all_steps, _ = lax.fori_loop(0, 8, self.set_steps, (all_steps, finished_steps))
        # all_steps = jax.vmap(lambda index: all_steps.at[index].set(32))(finished_steps)
        
        return finished_buffer, all_steps
        

    def __getitem__(self, i):
        # import code; code.interact(local=dict(globals(), **locals()))
        return (self.observations[i], self.actions[i], self.rewards[i], self.values[i], self.policies[i])

    def __len__(self):
        return self.actions.shape[0]



class MuZeroMemory:
    def __init__(self, length, games=[], rollout_size=5, n_step=10, discount_rate=0.995):
        self.length = length
        self.games = games
        self.rollout_size = rollout_size
        self.n_step = n_step
        self.discount_rate = discount_rate


    def append(self, self_play_memory):
        for i in range(len(self_play_memory)):
            game_memory = GameMemory(rollout_size=self.rollout_size, n_step=self.n_step, discount_rate=self.discount_rate)
            game_memory.add_from_self_play(self_play_memory[i])
            self.games.append(game_memory)
            if len(self.games) > self.length:
                self.games.pop(0)

    def item_count(self):
        return len(self.games)

    def update_priorities(self, priorities, game_indices, step_indices):
        return self.games[game_indices].update_priorities(priorities, step_indices)

    def compute_nstep_value(self, i, data):
        (starting_index, value, rewards, n_step) = data
        update_value = lambda: value + rewards[starting_index + n_step + 1] * self.discount_rate ** n_step
        value = lax.cond( starting_index + n_step + 1 < rewards.shape[0], update_value, lambda: value)
        return (starting_index, value, rewards, n_step)


    def sample_from_game(self, observations, actions, values, policies, priorities, rewards, game_index, rollout_size, n_step, key):
        available_indices = list(range(0, observations.shape[0]))
        starting_index = 32
        available_indices = np.stack(available_indices[starting_index : -(rollout_size + n_step)])
        priorities = np.stack(priorities[starting_index : -(rollout_size + n_step)])
        priorities = np.where(priorities == 0 , 1, priorities)
        sum = np.sum(priorities)
        # TODO check paper to understand why this is happening
        priorities = lax.cond(np.all(sum == 0), lambda: priorities + (1 / priorities.shape[0]), lambda: priorities / sum)
        # if sum != 0:
        #   priorities /= sum
        # else:
        #   priorities += 1 / len(priorities)

        index = random.choice(key, available_indices, p=priorities).squeeze()

        # result["observations"].append observation_result.append(result_dict["observations"])
        # action_result.append(result_dict["actions"])
        # reward_result.append(result_dict["rewards"])
        # value_result.append(result_dict["values"])
        # policy_result.append(result_dict["policies"])
        # step_index_result.append(result_dict["index"])
        # game_index_result.append(random_game)
        # priority_result.append(result_dict["priority"])
        k_step_actions = []
        k_step_rewards = []
        k_step_values = []
        k_step_policies = []
        for k_step in range(rollout_size + 1):
            k_step_actions.append(lax.dynamic_slice_in_dim(actions, index - 32 + k_step, 32))
            k_step_rewards.append(rewards[index + k_step])
            _, k_step_value, _, _= lax.fori_loop(0, n_step - 1, self.compute_nstep_value, (index + k_step, rewards[index + k_step] - values[index + k_step], rewards, n_step))
            k_step_values.append(k_step_value)
            # k_step_values.append(compute_nstep_value((index + k_step, rewards[index + k_step]) - values[index + k_step], rewards, n_step))
            k_step_policies.append(policies[index + k_step])

        
        observations = lax.dynamic_slice_in_dim(observations, index - 32, 32)
        return (np.array(observations), np.stack(k_step_actions), np.array(k_step_rewards), np.array(k_step_values), np.array(k_step_policies), game_index, index, priorities[index])
        # return key, {
        #     "observations": np.array(self.observations[index - 32 : index]),
        #     "actions": np.stack(k_step_actions),
        #     "rewards": np.array(k_step_rewards),
        #     "values": np.array(k_step_values),
        #     "policies": np.array(k_step_policies),
        #     "index": index,
        #     "priority": priorities[index],
        # }


    def sample(self, key, n):
                
        key, subkey = random.split(key)
        random_indices = random.choice(subkey, np.array(range(len(self.games))), shape=(1, n)).squeeze()
        observations = []
        actions = []
        values = []
        policies = []
        priorities = []
        rewards = []
        step_indices = []
        game_indices = []
        for i in random_indices:
            observations.append(self.games[i].observations)
            actions.append(self.games[i].actions)
            values.append(self.games[i].values)
            policies.append(self.games[i].policies)
            priorities.append(self.games[i].priorities)
            rewards.append(self.games[i].rewards)
            game_indices.append(i)

        # observation_result = []
        # action_result = []
        # reward_result = []
        # value_result = []
        # policy_result = []
        # step_index_result = []
        # game_index_result = []
        # priority_result = []
        keys = random.split(key, num=n + 1)
        key = keys[0]
        keys = keys[1:]

        game_sample = jax.vmap(self.sample_from_game, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, 0))
        observations, actions, rewards, values, policies, game_index, step_indices, priorities = game_sample(
            np.stack(observations),
            np.stack(actions),
            np.stack(values),
            np.stack(policies),
            np.stack(priorities),
            np.stack(rewards),
            np.stack(game_indices),
            self.rollout_size,
            self.n_step,
            keys
        )

        # for random_game in random_games:
        #     subkey, result_dict = self.games[random_game].sample(subkey, 1)

        #     observation_result.append(result_dict["observations"])
        #     action_result.append(result_dict["actions"])
        #     reward_result.append(result_dict["rewards"])
        #     value_result.append(result_dict["values"])
        #     policy_result.append(result_dict["policies"])
        #     step_index_result.append(result_dict["index"])
        #     game_index_result.append(random_game)
        #     priority_result.append(result_dict["priority"])

        return key, {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "policies": policies,
            "step_indices": step_indices,
            "game_indices": np.stack(game_indices),
            "priority": priorities,
        }



class GameMemory:
    def __init__(self, observations = [], actions = [], rewards = [], values = [], policies = [], priorities = [], rollout_size=5, n_step=10, discount_rate=0.995):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.policies = policies
        self.priorities = priorities
        self.rollout_size = rollout_size
        self.n_step = n_step
        self.discount_rate = discount_rate

    def last_n(self, n):
        return {"observations": self.observations[-n:], "actions": self.actions[-n:], "rewards": self.rewards[-n:], "values": self.values[-n:], "policies": self.policies[-n:]}

    def add_from_self_play(self, data):
        observations, actions, rewards, values, policies = data
        length = observations.shape[0]
        self.observations = observations
        self.actions = actions.squeeze()
        self.rewards = rewards.squeeze()
        self.values = values.squeeze()
        self.policies = policies.squeeze()
        # self.priorities.append(abs(values[i] - self.compute_nstep_value(length, values[i])))
        self.priorities = (abs(values - self.compute_nstep_value(length, values))).squeeze()

    def append(self, item):
        (observation, action, policy, value, reward) = item
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.policies.append(policy)
        # priority = abs(value.item() - self.compute_nstep_value(len(self.priorities), value).item())
        priority = abs(value - self.compute_nstep_value(len(self.priorities), value))
        self.priorities.append(priority)
        
    def update_priorities(self, priorities, indices):
      pr = np.array(self.priorities)
      scatter(pr, 0, indices, priorities.squeeze())
      self.priorities = np.asarray(pr)


    # TODO confirm value targets are correct. See make_target in psuedocode
    def compute_nstep_value(self, starting_index, starting_value):
      value = starting_value
      for n_step in (range(self.n_step - 1)):
        if starting_index + n_step + 1 < len(self.rewards):
          value += self.rewards[starting_index + n_step + 1] * self.discount_rate ** n_step
      return value


    # TODO Sample on a game basis (for atari "games" of 200 steps)

    def access_slice(self, array, start, end):
        return array[start:end]

    # TODO investigate vmapping
    def sample(self, key, n):
        available_indices = list(range(0, len(self.observations)))
        starting_index = 32
        available_indices = np.stack(available_indices[starting_index : -(self.rollout_size + self.n_step)])
        priorities = np.stack(self.priorities[starting_index : -(self.rollout_size + self.n_step)])
        priorities = priorities.at[priorities==0].set(1)
        sum = np.sum(priorities)
        # TODO check paper to understand why this is happening
        if sum != 0:
          priorities /= sum
        else:
          priorities += 1 / len(priorities)

        key, subkey = random.split(key)
        # import code; code.interact(local=dict(globals(), **locals()))
        index = random.choice(subkey, available_indices, p=priorities).squeeze()
        
        # for count, i in enumerate(indices):
        k_step_actions = []
        k_step_rewards = []
        k_step_values = []
        k_step_policies = []
        for k_step in range(self.rollout_size + 1):
            k_step_actions.append(np.array(self.actions[index - 32 + k_step : index + k_step]))
            k_step_rewards.append(self.rewards[index + k_step])

            k_step_values.append(self.compute_nstep_value(index + k_step, self.rewards[index + k_step]) - self.values[index + k_step])
            k_step_policies.append(self.policies[index + k_step])

        return key, {
            "observations": np.array(self.observations[index - 32 : index]),
            "actions": np.stack(k_step_actions),
            "rewards": np.array(k_step_rewards),
            "values": np.array(k_step_values),
            "policies": np.array(k_step_policies),
            "index": index,
            "priority": priorities[index],
        }

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
