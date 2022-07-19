from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from typing import Generator, Mapping, Text, Tuple, Optional, List
import time
from ml_collections import config_dict
from jaxline import utils as jl_utils
import utils
from jaxline import experiment
import optax
from experience_replay import Memory
import experience_replay
from model import support_to_scalar, scalar_to_support, MuZeroNet
import functools
import gym
import gym.wrappers as wrappers
from jax import random
from experience_replay import MuZeroMemory, SelfPlayMemory
from self_play import play_game  
from chex import assert_axis_dimension, assert_shape
import envpool
from jax.tree_util import register_pytree_node


class MuzeroExperiment(experiment.AbstractExperiment):
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_state': 'state',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config, learning_rate=1e-4, normalize_advantages=False, batch_size=128, rollout_size=5):
      super(MuzeroExperiment, self).__init__(mode=mode, init_rng=init_rng)
      self.mode = mode
      self.init_rng = init_rng
      self.config = config

      self._params = None
      self._state = None
      self._opt_state = None

      self.normalize_advantages = normalize_advantages
      self.game_reward = 0
      self.eps = np.finfo(np.float32).eps.item()
      self.steps_to_train = 4
      self.rollout_size = 5
      self.discount_factor = 0.99
      self.reward = 0
      self.games = 0
      self.epoch_length = 20000
      self.batch_size = batch_size
      self.lr_init = 0.05
      self.lr_decay_rate = 0.1
      self.lr_decay_steps = 350e3
      self.n_step = 10
      self.last_ten = Memory(10)
      self.last_hundred = Memory(100)
      self.rollout_size = 5
      self.key = random.PRNGKey(0)
      self.network = MuZeroNet()
      self.memory = MuZeroMemory(125000, rollout_size=rollout_size)
      self.starting_memories = SelfPlayMemory(64)
      register_pytree_node(
          MuZeroMemory,
          experience_replay.memory_flatten,
          experience_replay.memory_unflatten
      )
      register_pytree_node(
          SelfPlayMemory,
          experience_replay.self_play_flatten,
          experience_replay.self_play_unflatten
      )
      self._train_input = None
      
      self.env = envpool.make("Pong-v5", env_type="gym", num_envs=64, batch_size=16, img_height=96, img_width=96, gray_scale=False, stack_num=1)
      self.env.async_reset()
      self.env_handle, self.recv, self.send, _ = self.env.xla()

      self.initial_observation = self.env.reset()
      self.starting_policy = jnp.ones(18) / 18
      self._update_func = jax.pmap(self._update_func, axis_name='i',
                                 donate_argnums=(0, 1), in_axes=(None, None, (0, 0, 0, 0, 0, 0, 0), None, None), out_axes=(None, None, 0, 0))


      self.initial_observation = jnp.transpose(jnp.array(self.initial_observation), (0, 2, 3, 1))
      starting_policy = jnp.ones(18) / 18
      for i in range(32):
        self.starting_memories.observations = self.starting_memories.observations.at[i].set(self.initial_observation[0])
        self.starting_memories.policies = self.starting_memories.policies.at[:, i].set(starting_policy)
        # self.starting_memories.append((self.initial_observation[0], 0, starting_policy, jnp.broadcast_to(jnp.array(1.0), (1)), 0))

      self.entropy_scaling = 0.01
      self.automatic_optimization = False
      self.target_update_rate = 1000
      self._params = None
      self._target_params = None
      self.training_device_count = int(jax.device_count() / 2)

  def self_play(self, key, params, starting_memory, handle, recv, send):
      key, game_buffer, self.env_handle  = play_game(key, params, starting_memory, handle, recv, send)
      # key, game_buffer, self.env_handle = play_game(key, params, starting_memory, handle, recv, send)
      self.memory.append_multiple(game_buffer)
      return key, game_buffer

  
  def play_games(self):
    while(True):
      key, game_buffer = self.self_play(self.key, self._target_params, self.starting_memories, self.env_handle, jax.tree_util.Partial(self.recv), jax.tree_util.Partial(self.send))
      yield key, game_buffer


  def train_loop(
      self,
      config: config_dict.ConfigDict,
      state,
      periodic_actions: List[jl_utils.PeriodicAction],
      writer: Optional[jl_utils.Writer] = None,
  ) -> None:
    """Default training loop implementation.
    Can be overridden for advanced use cases that need a different training loop
    logic, e.g. on device training loop with jax.lax.while_loop or to add custom
    periodic actions.
    Args:
      config: The config of the experiment that is being run.
      state: Checkpointed state of the experiment.
      periodic_actions: List of actions that should be called after every
        training step, for checkpointing and logging.
      writer: An optional writer to pass to the experiment step function.
    """

    @functools.partial(jax.pmap, axis_name="i")
    def next_device_state(
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        host_id: Optional[jnp.ndarray],
    ):
      """Updates device global step and rng in one pmap fn to reduce overhead."""
      global_step += 1
      step_rng, state_rng = tuple(jax.random.split(rng))
      step_rng = jl_utils.specialize_rng_host_device(
          step_rng, host_id, axis_name="i", mode=config.random_mode_train)
      return global_step, (step_rng, state_rng)

    global_step_devices = np.broadcast_to(state.global_step,
                                          [jax.local_device_count()])
    host_id_devices = jl_utils.host_id_devices_for_rng(config.random_mode_train)
    step_key = state.train_step_rng

    self._optimizer = utils.make_optimizer(
        self.config.optimizer,
        self._lr_schedule)

    #TODO check how keys in jaxline are supposed to work
    self.key, representation_params, dynamics_params, prediction_params = self.network.initialize_networks_individual(self.key)
    self._params = (representation_params, dynamics_params, prediction_params)
    # init_opt = jax.pmap(self._optimizer.init)
    self._opt_state = self._optimizer.init(self._params)

    self._target_params = self._params

    self.key, game_buffer = next(self.play_games()) #play_game(self.key, self._target_params, self.memory, self.env, self.game_memory)
    self.memory.append_multiple(game_buffer)

    with jl_utils.log_activity("training loop"):
      while self.should_run_step(state.global_step, config):
        with jax.profiler.StepTraceAnnotation(
            "train", step_num=state.global_step):
          scalar_outputs = self.step(
              global_step=state.global_step, rng=step_key, writer=writer)

          t = time.time()
          # Update state's (scalar) global step (for checkpointing).
          # global_step_devices will be back in sync with this after the call
          # to next_device_state below.
          state.global_step += 1
          global_step_devices, (step_key, state.train_step_rng) = (
              next_device_state(global_step_devices,
                                state.train_step_rng,
                                host_id_devices))

        for action in periodic_actions:
          action(t, state.global_step, scalar_outputs)
  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step: int, rng: jnp.ndarray,
           *unused_args, **unused_kwargs):
    """See base class."""

    if self._train_input is None:
      self._initialize_train()

    inputs = next(self._train_input)

    (observations, actions, policies, values, rewards, indices, priorities) = inputs
    self._params, self._opt_state, scalars, value_difference = (
        self._update_func(
            self._params, self._opt_state, inputs, rng, global_step
            ))

    #TODO make sure this is right
    assert_shape(value_difference, (self.training_device_count, self.batch_size / self.training_device_count, None)) 
    assert_shape(indices, (self.training_device_count, self.batch_size / self.training_device_count)) 
    for i in range(value_difference.shape[0]):
      self.memory.update_priorities(value_difference[i, :, -1], indices[i])
    if global_step % self.target_update_rate == 0:
          self._target_params = self._params

    scalars = jl_utils.get_first(scalars)
    return scalars

  def _initialize_train(self):
    self._train_input = jl_utils.py_prefetch(self._build_train_input)

    total_batch_size = self.config.training.batch_size
    # steps_per_epoch = (
    #     self.config.training.images_per_epoch / self.config.training.batch_size)
    # Scale by the (negative) learning rate.
    self._optimizer = utils.make_optimizer(
        self.config.optimizer,
        self._lr_schedule)

    # Check we haven't already restored params
    if self._params is None:
      logging.info('Initializing parameters.')


      # init_net = jax.pmap(lambda *a: self.network.initialize_networks(*a))
      init_opt = jax.pmap(self._optimizer.init)

      # Init uses the same RNG key on all hosts+devices to ensure everyone
      # computes the same initial state.
      # init_rng = jl_utils.bcast_local_devices(self.init_rng)

      # self._params = init_net(init_rng)
      # self.target_params = self._params
      self._opt_state = init_opt(self._params)

  # def _load_data(self, split, is_training, batch_dims):
  #   """Wrapper for dataset loading."""

  #   return dataset.load(
  #       split=split,
  #       is_training=is_training,
  #       batch_dims=batch_dims,
  #       im_dim=self.config.data.im_dim,
  #       augmentation_settings=self.config.data.augmentation,
  #       )

  def _build_train_input(self):
    """See base class."""
    # num_devices = jax.device_count()
    # global_batch_size = self.config.training.batch_size
    while True:
      self.key, memories = self.memory.sample(self.batch_size, self.key)
      observations = np.reshape(memories["observations"], (self.training_device_count, int(self.batch_size / self.training_device_count), 32, 96, 96, 3))
      actions = np.reshape(memories["actions"], (self.training_device_count, int(self.batch_size / self.training_device_count), 6, 32))
      policies = np.reshape(memories["policies"], (self.training_device_count, int(self.batch_size / self.training_device_count), 6, 18))
      values = np.reshape(memories["values"], (self.training_device_count, int(self.batch_size / self.training_device_count), 6))
      rewards = np.reshape(memories["rewards"], (self.training_device_count, int(self.batch_size / self.training_device_count), 6))
      indices = np.reshape(memories["indices"], (self.training_device_count, int(self.batch_size / self.training_device_count)))
      priorities = np.reshape(memories["priority"], (self.training_device_count, int(self.batch_size / self.training_device_count)))
      print(observations.shape, actions.shape, policies.shape, values.shape, rewards.shape, indices.shape, priorities.shape)
      yield (observations, actions, policies, values, rewards, indices, priorities)

    # per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

    # if ragged:
    #   raise ValueError(
    #       f'Global batch size {global_batch_size} must be divisible by '
    #       f'num devices {num_devices}')


    # return self._load_data(
    #     split=split,
    #     is_training=True,
    #     batch_dims=[jax.local_device_count(), per_device_batch_size])

  def _one_hot(self, value):
    """One-hot encoding potentially over a sequence of labels."""
    y = jax.nn.one_hot(value, self.config.data.num_classes)
    return y


  def _muzero_loss_fn(
      self,
      params,
      inputs,
      rng,
      global_step,
  ):
        observations, actions, search_policy, search_value, simulator_reward, indices, priorities = inputs
        support_value = scalar_to_support(search_value)
        support_reward = scalar_to_support(simulator_reward)

        # cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # self.log("lr", cur_lr, prog_bar=True, on_step=True)
        # self.log("game_reward", self.game_reward, prog_bar=True, on_step=True)
        # self.log("last_ten_reward", self.last_ten.average(),
        #           prog_bar=True, on_step=True)
        # self.log("last_hundred_reward", self.last_hundred.average(),
        #           prog_bar=True, on_step=True)
        values = []
        policies = []
        rewards = []
        # TODO add target params
        # forward_observation_fn = jax.pmap(lambda *a: self.network.forward_observation(*a))
        # forward_hidden_state_fn = jax.pmap(lambda *a: self.network.forward_hidden_state(*a))
        value, policy, reward, hidden_state, self.key = self.network.forward_observation(self.key, params, actions[:, 0, :].astype(float), observations)

        values.append(value)
        policies.append(policy)
        rewards.append(reward)
        policy_loss = 0
        value_loss = 0
        reward_loss = 0
        for i in range(self.rollout_size):

          value, policy, reward, hidden_state, self.key = self.network.forward_hidden_state(self.key, params, actions[:, i, :].astype(float), hidden_state)
          # hidden_state.register_hook(lambda grad: grad * 0.5)
          values.append(value)
          policies.append(policy)
          rewards.append(reward)
        for i in range(len(values)):
          current_policy_loss = optax.softmax_cross_entropy(policies[i], search_policy.transpose(1, 0, 2)[i, :, :])
          current_value_loss = optax.softmax_cross_entropy(values[i].squeeze(), support_value[:, i, :])
          current_reward_loss = optax.softmax_cross_entropy(rewards[i], support_reward[:, i, :])
          # current_value_loss.register_hook(
          #   lambda grad: grad / self.rollout_size
          # )
          # current_reward_loss.register_hook(
          #   lambda grad: grad / self.rollout_size
          # )
          # current_policy_loss.register_hook(
          #   lambda grad: grad / self.rollout_size
          # )

          value_loss += current_value_loss
          reward_loss += current_reward_loss
          policy_loss += current_policy_loss

        loss = policy_loss + value_loss + reward_loss
        loss /= (self.memory.item_count() * priorities)
        loss = loss.mean()
        scaled_loss = loss / self.training_device_count
        # self.log("loss", loss, prog_bar=True, on_step=True)
        # self.log("play_step", self.steps, prog_bar=True, on_step=True)
        # self.manual_backward(loss)
        # with torch.no_grad():
        scalar_values = support_to_scalar(jnp.stack(values).transpose(1, 0, 2))
        value_difference = jnp.abs(search_value - scalar_values)
 
        loss_scalars = dict(
          loss=loss,
          value_difference=value_difference,
        )

        return scaled_loss, loss_scalars
        # return loss, state


  def _lr_schedule(self, step):
    return self.lr_init * self.lr_decay_rate ** (
      step / self.lr_decay_steps
    )


  def _loss_fn(
      self,
      params,
      state,
      inputs,
      rng,
  ):
    logits, state = self.forward.apply(
        params, state, rng, inputs, is_training=True)

    label = self._one_hot(inputs['labels'])
    # Handle cutmix/mixup label mixing:
    if 'mix_labels' in inputs:
      logging.info('Using mixup or cutmix!')
      mix_label = self._one_hot(inputs['mix_labels'])
      mix_ratio = inputs['ratio'][:, None]
      label = mix_ratio * label + (1. - mix_ratio) * mix_label

    # Apply label-smoothing to one-hot labels.
    label_smoothing = self.config.training.label_smoothing
    if not (label_smoothing >= 0. and label_smoothing < 1.):
      raise ValueError(
          f"'label_smoothing is {label_smoothing} and should be in [0, 1)")
    if label_smoothing > 0:
      smooth_positives = 1. - label_smoothing
      smooth_negatives = label_smoothing / self.config.data.num_classes
      label = smooth_positives * label + smooth_negatives

    loss_w_batch = utils.softmax_cross_entropy(logits, label)
    loss = jnp.mean(loss_w_batch, dtype=loss_w_batch.dtype)
    scaled_loss = loss / jax.device_count()

    metrics = utils.topk_correct(logits, inputs['labels'], prefix='')
    metrics = jax.tree_map(jnp.mean, metrics)

    top_1_acc = metrics['top_1_acc']
    top_5_acc = metrics['top_5_acc']

    loss_scalars = dict(
        loss=loss,
        top_1_acc=top_1_acc,
        top_5_acc=top_5_acc,
    )

    return scaled_loss, (loss_scalars, state)

  def _update_func(
      self,
      params,
      opt_state,
      inputs,
      rng: jnp.ndarray,
      global_step: int,
  ):
    """Applies an update to parameters and returns new state."""
    # This function computes the gradient of the first output of loss_fn and
    # passes through the other arguments unchanged.
    grad_loss_fn = jax.grad(self._muzero_loss_fn, has_aux=True)
    scaled_grads, loss_scalars = grad_loss_fn(
        params, inputs, rng, global_step)

    grads = jax.lax.psum(scaled_grads, axis_name='i')
    value_difference = loss_scalars["value_difference"]
    del loss_scalars["value_difference"]

    # Grab the learning rate to log before performing the step.
    learning_rate = self._lr_schedule(global_step)

    # Compute and apply updates via our optimizer.
    updates, opt_state = self._optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # n_params = 0
    # for k in params.keys():
    #   for l in params[k]:
    #     n_params = n_params + np.prod(params[k][l].shape)

    # Scalars to log (note: we log the mean across all hosts/devices).
    scalars = {'learning_rate': learning_rate,
              #  'n_params (M)': float(n_params/1e6),
               'global_gradient_norm': optax.global_norm(grads),
               }
    loss_scalars = {f'train_{k}': v for k, v in loss_scalars.items()}
    scalars.update(loss_scalars)
    scalars = jax.lax.pmean(scalars, axis_name='i')

    return params, opt_state, scalars, value_difference

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_args):
    """See base class."""
    global_step = np.array(jl_utils.get_first(global_step))
    scalars = jax.device_get(self._eval_epoch(jl_utils.get_first(rng)))

    logging.info('[Step %d] Eval scalars: %s', global_step, scalars)
    return scalars

  def _eval_batch(
      self,
      params,
      state,
      inputs,
      rng: jnp.ndarray,
  ):
    """Evaluates a batch."""
    logits, _ = self.forward.apply(
        params, state, rng, inputs, is_training=False)

    labels = self._one_hot(inputs['labels'])
    loss = utils.softmax_cross_entropy(logits, labels)

    metrics = utils.topk_correct(logits, inputs['labels'], prefix='')
    metrics = jax.tree_map(jnp.mean, metrics)
    top_1_acc = metrics['top_1_acc']
    top_5_acc = metrics['top_5_acc']

    bs = logits.shape[0]

    top_1_acc = jnp.expand_dims(top_1_acc, axis=0) * bs
    top_5_acc = jnp.expand_dims(top_5_acc, axis=0) * bs

    # NOTE: Returned values will be summed and finally divided by num_samples.
    return {
        'eval_loss': loss,
        'eval_top_1_acc': top_1_acc, 'eval_top_5_acc': top_5_acc}

  def _build_eval_input(self):
    split = dataset.Split.from_string(self.config.evaluation.subset)

    return self._load_data(
        split=split,
        is_training=False,
        batch_dims=[self.config.evaluation.batch_size])

  def _eval_epoch(self, rng):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_scalars = None

    params = jl_utils.get_first(self._params)
    state = jl_utils.get_first(self._state)

    for inputs in self._build_eval_input():
      num_samples += inputs['labels'].shape[0]
      scalars = self._eval_batch(params, state, inputs, rng)

      # Accumulate the sum of scalars for each step.
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_multimap(jnp.add, summed_scalars, scalars)

    mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
    return mean_scalars
