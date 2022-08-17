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
        (memory.games, memory.halting_steps))

def muzero_flatten(memory):
    return ((memory.games, memory.priorities), (memory.length, memory.rollout_size, memory.n_step, memory.discount_rate))

def muzero_unflatten(aux, children):
    memory = MuZeroMemory(1)
    memory.games, memory.priorities = children
    memory.length, memory.rollout_size, memory.n_step, memory.discount_rate = aux
    return memory


def self_play_unflatten(aux, children):
    (games, halting_steps) = aux
    memory = SelfPlayMemory(games, halting_steps)
    (observations, actions, rewards, values, policies) = children
    memory.observations = observations
    memory.actions = actions
    memory.rewards = rewards
    memory.values = values
    memory.policies = policies
    return memory


class SelfPlayMemory:
    def __init__(self, games, halting_steps=232):
        self.games = games
        self.halting_steps = halting_steps

    def populate(self):
        self.observations = np.zeros((self.games, self.halting_steps, 96, 96, 3))
        self.actions = np.zeros((self.games, self.halting_steps, 1))
        self.rewards = np.zeros((self.games, self.halting_steps, 1))
        self.values = np.zeros((self.games, self.halting_steps, 1))
        self.policies = np.zeros((self.games, self.halting_steps, 18))

    def set_steps(self, i, data):
        all_steps, finished_steps = data
        return all_steps.at[finished_steps[i]].set(32)

    def update_memory(self, observations, actions, rewards, values, policies):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.policies = policies

    @jax.jit
    def output_game_buffer(self, finished_steps, all_steps, starting_observations, amount_to_add = 8):
        if amount_to_add:
            finished_steps = finished_steps[0 : amount_to_add]
        finished_buffer = SelfPlayMemory(self.games, self.halting_steps)
        finished_buffer.observations = jax.vmap(lambda index: self.observations[index])(finished_steps)
        finished_buffer.actions = jax.vmap(lambda index: self.actions[index])(finished_steps)
        finished_buffer.rewards = jax.vmap(lambda index: self.rewards[index])(finished_steps)
        finished_buffer.values = jax.vmap(lambda index: self.values[index])(finished_steps)
        finished_buffer.policies = jax.vmap(lambda index: self.policies[index])(finished_steps)

        self.observations = self.observations.at[finished_steps].set(starting_observations[0])
        self.actions = self.actions.at[finished_steps].set(0)
        self.rewards = self.rewards.at[finished_steps].set(0)
        self.values = self.observations.at[finished_steps].set(0)
        self.policies = self.policies.at[finished_steps].set(0)


        for i in range(8):
            all_steps = self.set_steps(i, (all_steps, finished_steps))
        
        return finished_buffer, all_steps
        

    def __getitem__(self, i):
        return (self.observations[i], self.actions[i], self.rewards[i], self.values[i], self.policies[i])

    def __len__(self):
        return self.actions.shape[0]



class MuZeroMemory:
    def __init__(self, length, games=[], priorities=[], rollout_size=5, n_step=10, discount_rate=0.995):
        self.length = length
        self.games = games
        self.priorities = priorities
        self.rollout_size = rollout_size
        self.n_step = n_step
        self.discount_rate = discount_rate


    def append(self, self_play_memory):
        print("BEFORE APPEND", len(self.games))
        for i in range(len(self_play_memory)):
            game_memory = GameMemory(rollout_size=self.rollout_size, n_step=self.n_step, discount_rate=self.discount_rate)
            game_memory.add_from_self_play(self_play_memory[i])
            self.games.append(game_memory)
            self.priorities.append(np.max(self.games[-1].priorities))
            if len(self.games) > self.length:
                self.games.pop(0)
                self.priorities.pop(0)

        print("AFTER APPEND", len(self.games))
    def item_count(self):
        return len(self.games)

    def update_priorities(self, priorities, game_indices, step_indices):
        self.games[game_indices].update_priorities(priorities, step_indices)
        self.priorities[game_indices] = np.max(self.games[game_indices].priorities)

    def compute_nstep_value(self, i, data):
        (starting_index, value, rewards, n_step) = data
        update_value = lambda: value + rewards[starting_index + n_step + 1] * self.discount_rate ** n_step
        value = lax.cond( starting_index + n_step + 1 < rewards.shape[0], update_value, lambda: value)
        return (starting_index, value, rewards, n_step)


    def choice(self, priorities, key):
        available_indices = list(range(0, priorities.shape[0]))
        starting_index = 32
        available_indices = np.stack(available_indices[starting_index : -(self.rollout_size + self.n_step)])
        priorities = np.stack(priorities[starting_index : -(self.rollout_size + self.n_step)])
        priorities = np.where(priorities == 0 , 1, priorities)
        sum = np.sum(priorities)
        # TODO check paper to understand why this is happening
        priorities = lax.cond(np.all(sum == 0), lambda: priorities + (1 / priorities.shape[0]), lambda: priorities / sum)
        index = random.choice(key, available_indices, p=priorities).squeeze()
        return index, priorities

    
    # @partial(jax.jit, static_argnums=(2))
    def fetch_games(self, key, n):
                
        key, subkey = random.split(key)
        game_indices = random.choice(subkey, np.array(range(len(self.games))), shape=(1, n), p=np.array(self.priorities[0: len(self.games)])).squeeze()

        keys = random.split(key, num=n + 1)
        key = keys[0]
        keys = keys[1:]

        observations = []
        actions = []
        values = []
        policies = []
        priorities = []
        rewards = []
        step_indices = []
        priority_result = []
        for i in game_indices:
            priorities.append(self.games[i].priorities)
            observations.append(self.games[i].observations)
            actions.append(self.games[i].actions)
            values.append(self.games[i].values)
            policies.append(self.games[i].policies)
            rewards.append(self.games[i].rewards)

        priorities = np.array(priorities)
        choices, priorities = jax.vmap(self.choice, (0, 0))(priorities, keys)

        return key, (np.stack(observations), np.stack(actions), np.stack(values), np.stack(policies), np.stack(rewards), priorities, game_indices, choices)

def compute_nstep_value(i, data):
  (starting_index, value, rewards, n_step, discount_rate) = data
  update_value = lambda: value + rewards[starting_index + n_step + 1] * discount_rate ** n_step
  value = lax.cond( starting_index + n_step + 1 < rewards.shape[0], update_value, lambda: value)
  return (starting_index, value, rewards, n_step, discount_rate)


# TODO move to fetching out of bounds blank states on sample
@partial(jax.jit, static_argnums=(7, 8, 9))
def sample_from_game(observations, actions, values, policies, rewards, game_index, step_index, rollout_size, n_step, discount_rate):
  k_step_actions = []
  k_step_rewards = []
  k_step_values = []
  k_step_policies = []
  for k_step in range(rollout_size + 1):
      k_step_actions.append(lax.dynamic_slice_in_dim(actions, step_index - 32 + k_step, 32))
      k_step_rewards.append(rewards[step_index + k_step])
      _, k_step_value, _, _, _ = lax.fori_loop(0, n_step - 1, compute_nstep_value, (step_index + k_step, rewards[step_index + k_step] - values[step_index + k_step], rewards, n_step, discount_rate))
      k_step_values.append(k_step_value)
      k_step_policies.append(policies[step_index + k_step])

  
  observations = lax.dynamic_slice_in_dim(observations, step_index - 32, 32)
  return (np.array(observations), np.stack(k_step_actions), np.array(k_step_rewards), np.array(k_step_values), np.array(k_step_policies), game_index, step_index)


@partial(jax.jit, static_argnums=(1,2,3,))
def memory_sample(data, rollout_size, n_step, discount_rate):
    (observations, actions, values, policies, rewards, priorities, game_indices, choices) = data
    game_sample = jax.vmap(sample_from_game, (0, 0, 0, 0, 0, 0, 0, None, None, None))
    observations, actions, rewards, values, policies, _, step_indices = game_sample(
      observations,
      actions,
      values,
      policies,
      rewards,
      game_indices,
      choices,
      rollout_size,
      n_step,
      discount_rate
    )

    priority_result = jax.vmap((lambda i, j: priorities[i][j]), (0, 0))(game_indices, step_indices)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "policies": policies,
        "step_indices": step_indices,
        "game_indices": np.stack(game_indices),
        "priority": np.stack(priority_result),
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
        self.priorities = (abs(values - self.compute_nstep_value(length, values))).squeeze()

    def append(self, item):
        (observation, action, policy, value, reward) = item
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.policies.append(policy)
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

    def access_slice(self, array, start, end):
        return array[start:end]

    # TODO investigate vmapping
    def sample(self, key, n, device):
        available_indices = list(range(0, len(self.observations)))
        starting_index = 32
        available_indices = np.stack(available_indices[starting_index : -(self.rollout_size + self.n_step)])
        priorities = np.stack(self.priorities[starting_index : -(self.rollout_size + self.n_step)])
        priorities = priorities.at[priorities==0].set(1)
        priorities = jax.device_put(priorities, device)
        sum = np.sum(priorities)
        # TODO check paper to understand why this is happening
        if sum != 0:
          priorities /= sum
        else:
          priorities += 1 / len(priorities)

        key, subkey = random.split(key)
        index = random.choice(subkey, available_indices, p=priorities).squeeze()
        
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
