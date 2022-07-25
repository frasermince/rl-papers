from os import environ
import jax.numpy as jnp
import numpy as np
from model import one_hot_encode, support_to_scalar, MuZeroNet, scatter
import jax.lax as lax
import jax
import flax
import flax.linen as nn
import envpool
from operator import itemgetter
# from jax.config import config

# config.update('jax_disable_jit', True)

# TODO Follow reference implementation for game history
# TODO Move to storing children of tree in tensor following https://github.com/Hwhitetooth/jax_muzero/blob/948ecc268c88521c0f7a8a5d12df68e3bb370b3a/algorithms/agents.py

discount_factor = jnp.array(0.99)
c1 = jnp.array(1.25)
c2 = jnp.array(19652)
zeros = jnp.zeros(18).astype(jnp.int32)

def min_max_update(value, min_max):
  minimum, maximum = min_max
  return (jnp.min(minimum, value), jnp.max(maximum, value))

def normalize(value, min_max):
  minimum, maximum = min_max
  return jnp.where(maximum > minimum, (value - minimum) / (maximum - minimum), value)

def visit_policy(environment):
  result = jnp.zeros(18)
  result = environment["policy"][1:19] / environment["visit_count"][1:19]
  # for i in range(18):
  #   result[i] = environment["policy"][0][i] / environment["visit_count"][i]
  return nn.softmax(result, axis=0)


def initial_expand(environment, state, policy):
  environment["state"] = environment["state"].at[0, :, :, :].set(state.squeeze())
  #policy = {a: math.exp(policy[a]) for a in range(18)}
  #policy_sum = sum(policy)
  policy = jnp.exp(policy)
  policy_sum = policy.sum()
  for action in range(18):
    environment["current_index"] += 1
    environment["policy"] = environment["policy"].at[environment["current_index"]].set(policy[action] / policy_sum)
    environment["children"] = environment["children"].at[0, action].set(environment["current_index"])
    #self.children.append(Node(p / policy_sum, self.prediction_network, self.dynamics_network, self.device, self.normalizer))
  return environment


def ucb_count_all_actions(environment, parent_index, first_child_index):
  indices = jnp.array(range(0, 18))
  indices = (indices + first_child_index).astype(jnp.int32)
  upper_confidence_bound = jnp.sqrt(environment["visit_count"][parent_index])
  visit_count = jnp.take(environment["visit_count"], indices, axis=0)
  upper_confidence_bound /= 1 + visit_count
  upper_confidence_bound *= c1 + jnp.log((environment["visit_count"][parent_index] + c2 + 1) / c2)
  upper_confidence_bound *= environment["policy"].take(indices, axis=0)
  upper_confidence_bound = jnp.where(environment["visit_count"].take(indices, axis=0) > 0, upper_confidence_bound + environment["q_val"].take(indices, axis=0), upper_confidence_bound)
  return normalize(upper_confidence_bound, environment["min_max"])


def choose_child(environment, index):
  results = ucb_count_all_actions(environment, index, environment["children"][index, 0])
  action = results.argmax()
  #return self.children[index, action], action
  return environment["children"].take(index, axis=0).take(action, axis=0).astype(jnp.int32), action

def expand(environment, action, state, index):
  _, dynamics_params, prediction_params = environment["params"]
  network = MuZeroNet()
  action = jnp.expand_dims(action, axis=0)
  state = jnp.expand_dims(state, axis=0)
  action = one_hot_encode(action, action.shape, state.shape[1:3])

  (new_state, reward), _ = network.dynamics_net(dynamics_params, state, action)
  (value, policy), _ = network.prediction_net(prediction_params, state)
  policy = policy.squeeze()
  environment["state"] = environment["state"].at[index].set(new_state.squeeze())
  policy = jnp.exp(policy)
  policy_sum = policy.sum()
  action_indices = jnp.array(range(0,18))
  children_indices = action_indices + environment["current_index"] + 1
  
  environment["current_index"] += 18
  environment["policy"] = scatter(environment["policy"], -1, children_indices,  policy / policy_sum)
  environment["children"] = environment["children"].at[index].set(scatter(environment["children"][index], -1, action_indices, children_indices))
  scalar_reward = support_to_scalar(reward).squeeze().squeeze()
  environment["reward"] = environment["reward"].at[index].set(scalar_reward)
  return environment, value

def branch(params):
  environment, index = params
  child_index, action = choose_child(environment, index)
  environment["history"] = environment["history"].at[environment["history_length"]].set(action)
  environment["search_path"] = environment["search_path"].at[environment["history_length"]].set(index)
  environment["history_length"] += 1
  return environment, child_index
  # previous_qval, environment = monte_carlo_simulation(environment, parent_index=index, index=child_index, parent_action=action)
  # cumulative_discounted_reward = environment["reward"][index] + discount_factor * previous_qval
  # environment["min_max"] = min_max_update(cumulative_discounted_reward, environment["min_max"])
  # environment["q_val"].at[index].update((environment["visit_count"][index] * environment["q_val"][index] + cumulative_discounted_reward) / (environment["visit_count"][index] + 1))
  # environment["visit_count"][index] += 1
  # environment["actions"].append(action)
  # environment["history"].append(index)
  # return cumulative_discounted_reward, environment

def scan_fn(accum):
  environment, value, i = accum
  index = environment["search_path"][i]
  environment["visit_count"].at[index].set(environment["visit_count"][index] + 1)
  environment["q_val"] = environment["q_val"].at[index].set(value)
  value = environment["reward"][index] + discount_factor * value
  return environment, value, i - 1

def condition_fn(accum):
  _, _, i = accum
  return i >= 0


def backpropagate(environment, value):
  environment, value, _ = lax.while_loop(condition_fn, scan_fn, (environment, support_to_scalar(value).squeeze().squeeze(), environment["history_length"]))
  return environment

def not_all_zeros(params):
  environment, index = params
  children = environment["children"][index]
  return jnp.all(children != zeros)

def monte_carlo_simulation(i, data):
  environment, index = data
  # TODO: the following:
  # 1. Add temperature formula see Appendix D of paper
  # 2. Confirm that value formulas match paper
  environment, index = lax.while_loop(not_all_zeros, branch, (environment, index))
  # while not_all_zeros((environment, index)):
  #   environment, index = branch((environment, index))
  parent_index = environment["search_path"][environment["history_length"] - 1]
  index = environment["search_path"][environment["history_length"]]
  environment, value = expand(environment, environment["history"][index], environment["state"][parent_index], index)
  return backpropagate(environment, value), index

  #print("INDEX", index)
  # if jnp.all(children != self.zeros):
  #     child_index, action = self.choose_child(index)
  #     previous_qval = self.monte_carlo_tree_search(parent_index=index, index=child_index, parent_action=action)
  #     cumulative_discounted_reward = self.reward[index] + self.discount_factor * previous_qval
  #     self.normalizer.update(cumulative_discounted_reward)
  #     self.q_val = self.q_val.at[index].update((self.visit_count[index] * self.q_val[index] + cumulative_discounted_reward) / (self.visit_count[index] + 1))
  #     self.visit_count[index] += 1
  #     return cumulative_discounted_reward
  # else:
  #     reward, value = self.expand(parent_action, self.state[parent_index], index)
  #     self.q_val = self.q_val.at[index].update(support_to_scalar(value.unsqueeze(0)))
  #     self.reward = self.reward.at[index].update(support_to_scalar(reward.unsqueeze(0).unsqueeze(0)))
  #     #print("REWARD", index, self.reward)
  #     #print(self.reward[index], self.q_val[index])
  #     #TODO check if this is right
  #     self.normalizer.update(self.reward[index])
  #     self.visit_count[index] += 1
  #     return self.q_val[index]#, action
            
def perform_simulations(params, hidden_state, policy):
  environment = {
    "policy": jnp.zeros([2000]),
    "visit_count": jnp.ones([2000]),
    "q_val": jnp.zeros([2000]),
    "reward": jnp.zeros([2000]),
    "children": jnp.zeros([2000, 18]).astype(jnp.int32),
    "state": jnp.zeros([2000, 6, 6, 256]),
    "min_max": (jnp.array(float("infinity")), jnp.array(float("-infinity"))),
    "params": params,
    "history": jnp.array(range(2000)),
    "search_path": jnp.array(range(2000)),
    "current_index": jnp.array(0),
    "history_length": 0
  }
  environment = initial_expand(environment, hidden_state, policy)
        # self.discount_factor = jnp.array(0.99)
        # self.c1 = jnp.array(1.25)
        # self.c2 = jnp.array(19652)
        # self.indices = jnp.array(range(0, 18))
        # self.normalizer = normalizer
        # self.current_index = 0
        # self.simulations = simulations
        # self.prediction_network = prediction_network
        # self.dynamics_network = dynamics_network
  # (environment, _) = lax.fori_loop(0, 25, monte_carlo_simulation, (environment, jnp.array(0)))
  (environment, _) = lax.fori_loop(0, 50, monte_carlo_simulation, (environment, jnp.array(0)))
  # for i in range(25):
  #   (environment, _) = monte_carlo_simulation(i, (environment, jnp.array(0)))
  # TODO confirm that we don't need to divide by the visit count
  return visit_policy(environment), environment["q_val"][0], environment

@jax.jit
def monte_carlo_tree_search(params, observation, action):
    network = MuZeroNet()
    (representation_params, _, prediction_params) = params
    # print("REP SHAPE", representation_params)
    # TODO find better way than copying 8 times

    # import code; code.interact(local=dict(globals(), **locals()))
    # representation_fn = jax.pmap(lambda *x: network.representation_net(*x))
    hidden_state, _ = network.representation_net(representation_params, observation, action)
    (value, policy), _ = network.prediction_net(prediction_params, hidden_state)
    #node = Node(policy, network.prediction_net, network.dynamics_net, self.device, normalizer)
    policy, value, environment = perform_simulations(params, hidden_state, policy.squeeze())

    return policy, value, environment


# @jax.jit
def add_item(i, data):
  current_games, time_steps, env_id, observation, action, policy, value, reward = data
  id = env_id[i]
  step = time_steps[id]
  current_games.observations = current_games.observations.at[id, step].set(observation[i].transpose(1, 2, 0))
  current_games.actions = current_games.actions.at[id, step].set(action[i])
  current_games.policies = current_games.policies.at[id, step].set(policy[i])
  current_games.values = current_games.values.at[id, step].set(value[i])
  current_games.rewards = current_games.rewards.at[id, step].set(reward[i])
  time_steps = time_steps.at[id].set(time_steps[id] + 1)
  return current_games, time_steps, env_id, observation, action, policy, value, reward


# TODO explore if steps are recorded correctly due to async gym
monte_carlo_fn = jax.pmap(lambda *x: jax.vmap(lambda *y: monte_carlo_tree_search(*y), (None, 0, 0))(*x), in_axes=((None, 0, 0)), devices=jax.devices()[0: 4])
get_actions = jax.vmap(lambda current_games, index, steps: lax.dynamic_slice_in_dim(current_games.actions[index], steps[index] - 32, 32, axis=0).squeeze(), (None, 0, None))
get_observations = jax.vmap(lambda current_games, index, steps: lax.dynamic_slice_in_dim(current_games.observations[index], steps[index] - 32, 32, axis=0), (None, 0, None))
def play_step(i, p): #params, current_game_buffer, env_handle, recv, send):
    (key, params, env, current_games, steps, rewards) = p
    # if self.steps == 1:
    #   self.network.set_device(self.device)
    #   self.target_network.set_device(self.device)
    second_observation, reward, is_done, info = env.recv()
    print(info['env_id'][is_done])
    print(reward)
    rewards = rewards.at[info['env_id'][is_done]].set(0)
    rewards = rewards.at[info['env_id']].set(rewards[info['env_id']] + reward)
    # get_actions = jax.vmap(lambda current_games, index: lax.dynamic_slice_in_dim(current_games.actions[index], steps[index] - 32, 32, axis=0).squeeze(), (None, 0))
    # get_observations = jax.vmap(lambda current_games, index: lax.dynamic_slice_in_dim(current_games.observations[index], steps[index] - 32, 32, axis=0), (None, 0))
    past_actions = jnp.expand_dims(get_actions(current_games, info['env_id'], steps), axis=1)
    past_observations = jnp.expand_dims(get_observations(current_games, info['env_id'], steps), axis=1)
    past_observations = np.reshape(past_observations, (4, int(past_observations.shape[0] / 4), 1, 32, 96, 96, 3))
    past_actions = jnp.reshape(past_actions, (4, int(past_actions.shape[0] / 4), 1, 32))

    # TODO use 0 for past_observations upon changing to multigame memory
    # import code; code.interact(local=dict(globals(), **locals()))
    # print("BEFORE SEARCH")
    policy, value, _ = monte_carlo_fn(params, past_observations, past_actions)
    # import code; code.interact(local=dict(globals(), **locals()))
    # policy, value, environment = monte_carlo_tree_search(params, past_observations[0, 0], past_actions[0, 0])
    # print("ENVIRONMENT", environment["history_length"], environment["history"], environment["search_path"])
    # print("AFTER SEARCH")
    key, subkey = jax.random.split(key)
    policy = policy.reshape(policy.shape[0] * policy.shape[1], policy.shape[2])
    value = value.reshape(value.shape[0] * value.shape[1])
    action = jax.random.categorical(subkey, policy, axis=1)

    past_observations = None
    past_actions = None

    # second_observation, reward, is_done, _ = env.step(action)
    env.send(np.array(action), info['env_id'])
    # import code; code.interact(local=dict(globals(), **locals()))

    # reward += reward
    # if reward > 0:
    #     game_reward += reward
    # if is_done:
    #     self.last_ten.append(self.game_reward)
    #     self.logger.experiment.add_scalar("game_score", self.game_reward, self.games)
    #     second_observation = self.env.reset()
    #     game_reward = 0
    #     self.games += 1

    # cpus = jax.devices("cpu")
    # current_game_buffer.append((jnp.array(second_observation) / 255, action, policy, value, reward))
    for j in range(info['env_id'].shape[0]):
      current_games, steps, _, _, _, _, _, _ = add_item(j, (current_games, steps, info['env_id'], second_observation, action, policy, value, reward))
    # current_games, steps, _, _, _, _, _, _ = lax.fori_loop(0, info['env_id'].shape[0], add_item, (current_games, steps, info['env_id'], second_observation, action, policy, value, reward))

    print(i)
    if i % 50 == 0:
      print("MAX STEP", jnp.max(steps))
      print("rewards", jnp.mean(rewards))
    previous_steps = steps
    # for i in range(0, info['env_id'].shape[0]):
    #   current_games, steps, _, _, _, _, _, _ = add_item(i, (current_games, steps, info['env_id'], second_observation, action, policy, value, reward))
    # import code; code.interact(local=dict(globals(), **locals()))
    return (key, params, env, current_games, steps, rewards)#, game_reward

# @jax.jit
def play_game(key, params, self_play_memories, env, steps, rewards, halting_steps):
    # TODO set 200 per environment
    # (key, params, new_handle, recv, send, self_play_memories) = lax.fori_loop(0, 200, play_step, (key, params, env_handle, recv, send, self_play_memories))
    # return key, current_game_buffer, env_handle
    # jax.default_device = jax.devices("cpu")[0]
    # jax.default_device = None
      # while(jnp.any(steps < 40)):
      for i in range(100):
      #     # TODO backfill from previous memories
        (key, params, env, self_play_memories, steps, rewards) = play_step(i, (key, params, env, self_play_memories, steps, rewards))

      finished_indices = jnp.argwhere(steps >= halting_steps).squeeze()
      return key, self_play_memories, steps, finished_indices, rewards
      # return key, current_game_buffer
