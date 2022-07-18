import jax.numpy as np
from jax import random
from model import scatter

# TODO Add per game memory for chess and go

def flatten(memory):
    return ((memory.observations, memory.actions, memory.rewards, memory.values, memory.policies, memory.priorities),
        (memory.rollout_size, memory.n_step, memory.discount_rate, memory.length))

def unflatten(aux, children):
    print(children)
    (observations, actions, rewards, values, policies, priorities) = children
    (rollout_size, n_step, discount_rate, length) = aux
    return MuZeroMemory(length, observations=observations, actions=actions, rewards=rewards, values=values, policies=policies, priorities=priorities, n_step=n_step, discount_rate=discount_rate)

class MuZeroMemory:
    def __init__(self, length, observations = [], actions = [], rewards = [], values = [], policies = [], priorities = [], rollout_size=5, n_step=10, discount_rate=0.995):
        self.length = length
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

    def append_multiple(self, array):
        for i in range(len(array.observations)):
            self.observations.append(array.observations[i])
            self.actions.append(array.actions[i])
            self.rewards.append(array.rewards[i])
            self.values.append(array.values[i].item())
            self.policies.append(array.policies[i])
            self.priorities.append(abs(array.values[i].item() - self.compute_nstep_value(len(self.priorities), array.values[i]).item()))

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
        if (len(self.observations) > self.length):
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.policies.pop(0)
            self.priorities.pop(0)

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

    # TODO investigate vpmapping
    def sample(self, n, key):

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
        indices = random.choice(subkey, available_indices, shape=(1, n), p=priorities).squeeze()
        observation_result = []
        action_result = []
        reward_result = []
        value_result = []
        policy_result = []
        index_result = []
        priority_result = []

        for count, i in enumerate(indices):
            k_step_actions = []
            k_step_rewards = []
            k_step_values = []
            k_step_policies = []
            for k_step in range(self.rollout_size + 1):
              k_step_actions.append(np.array(self.actions[i - 32 + k_step : i + k_step]))
              k_step_rewards.append(self.rewards[i + k_step])

              k_step_values.append(self.compute_nstep_value(i + k_step, self.rewards[i + k_step]) - self.values[i + k_step])
              k_step_policies.append(self.policies[i + k_step])
            observation_result.append(np.array(self.observations[i - 32 : i]))
            action_result.append(np.stack(k_step_actions))
            reward_result.append(np.array(k_step_rewards))
            value_result.append(np.array(k_step_values))
            policy_result.append(np.array(k_step_policies))
            index_result.append(indices)
            priority_result.append(priorities[count])

        return key, {
            "observations": np.stack(observation_result),
            "actions": np.stack(action_result),
            "rewards": np.stack(reward_result),
            "values": np.stack(value_result),
            "policies": np.stack(policy_result),
            "indices": np.array(indices),
            "priority": np.stack(priority_result),
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
