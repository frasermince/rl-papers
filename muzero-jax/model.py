import jax
import jax.numpy as jnp
import flax.linen as nn
import random
from chex import assert_axis_dimension, assert_shape
from jax import random
from jaxline import utils as jl_utils

epsilon = 0.001

def scatter(input, dim, index, src, reduce=None, should_print=False):
  idx = jnp.meshgrid(*(jnp.arange(n) for n in input.shape), sparse=True, indexing='ij')
  # if should_print:
    # print("ARANGE", *(jnp.arange(n) for n in input.shape))
    # print(input.shape, dim, index.shape, src)
    # for n in input.shape:
    #   print("axes", jnp.arange(n))
    # print("first idx", idx[0].shape)
    # print("first idx", idx[1].shape)
    # print("first idx", idx[2].shape)

    # print("index", index)
    # print("dim", dim)
  idx[dim] = index
  # if should_print:
  #   print("idx", idx[0].shape)
  #   print("idx", idx[1].shape)
  #   print("idx", idx[2].shape)
  #   print("src", src)
  # if should_print:
    # print("indexed", idx[dim])
  return getattr(input.at[tuple(idx)], reduce or "set")(src)

def support_to_scalar(supports):
  indices = jnp.expand_dims(jnp.array(range(-300, 301), dtype=jnp.float32), axis=0)
  indices = jnp.expand_dims(indices, axis=0)
  probabilities = nn.softmax(supports, axis=-1)
  scalar = (probabilities * indices).sum(axis=-1)
  sign = jnp.sign(scalar)
  scalar = ((jnp.sqrt(1 + 4 * epsilon * (jnp.abs(scalar) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1
  scalar *= sign

  return scalar
  
  # x = torch.sign(x) * (
  #       ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
  #       ** 2
  #       - 1
    # )
def scalar_to_support(scalar):
  scalar = jnp.sign(scalar) * ((jnp.sqrt(jnp.abs(scalar) + 1)) - 1 + epsilon * scalar)
  supports = jnp.expand_dims(jnp.zeros(scalar.shape), axis=-1)
  supports = jnp.broadcast_to(supports, (supports.shape[0], supports.shape[1], 601))
  floor = jnp.expand_dims((jnp.floor(scalar) + 300).astype(int), axis=-1)
  remainder = jnp.expand_dims(scalar - jnp.floor(scalar), axis=-1)
  supports = scatter(supports, len(supports.shape) - 1, floor, 1.0, should_print=True)
  supports = scatter(supports, len(supports.shape) - 1, floor + 1, remainder)
  return supports

def one_hot_encode(action, input_shape, action_output_shape):
  result = jnp.zeros((input_shape[0], action_output_shape[0] * action_output_shape[1]))
  result = scatter(result, 1, jnp.expand_dims(action, axis=1), 1)
  assert_shape(result, (None, action_output_shape[0] * action_output_shape[1]))
  result = jnp.reshape(result, (input_shape[0], action_output_shape[0], action_output_shape[1]))
  assert_shape(result, (input_shape[0], action_output_shape[0], action_output_shape[1]))
  return result 


class ResBlock(nn.Module):
    inplanes: int
    planes: int
    stride: int

    @nn.compact
    def __call__(self, x):
        residual = x
        assert_axis_dimension(x, 3, self.inplanes)
        x = nn.Sequential([
          nn.Conv(self.planes, kernel_size=(3, 3), strides=self.stride, padding=1),
          nn.BatchNorm(use_running_average=False),
          nn.relu,
          nn.Conv(self.planes, kernel_size=(3, 3), strides=self.stride, padding=1),
          nn.BatchNorm(use_running_average=False),
        ])(x)
        assert_axis_dimension(x, 3, self.planes)
        if residual.shape != x.shape:
            residual = nn.Conv(self.planes, (1, 1),
                           self.stride, name='conv_proj')(residual)
            # residual = self.norm(name='norm_proj')(residual)
        return nn.relu(x + residual)
    
class PredictionNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        super(PredictionNet, self).__init__()
        assert_shape(x, (None, 6, 6, 256))
        x = nn.relu(nn.Conv(128, kernel_size=(3, 3))(x))
        x = nn.relu(nn.Conv(64, kernel_size=(3, 3))(x))
        # print("X SHAPE BEFORE", x.shape)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = jnp.reshape(x, (x.shape[0], -1))
        # print("X SHAPE AFTER", x.shape)
        policy = nn.relu(nn.Dense(18)(x))
        value = nn.relu(nn.Dense(512)(x))
        value = nn.Dense(601)(x)
        return value, policy

class RepresentationNet(nn.Module):
    @nn.compact
    def __call__(self, inputs, actions):
        assert_shape(inputs, (None, 32, 96, 96, 3))
        inputs = jnp.transpose(inputs, (0, 4, 1, 2, 3))
        inputs = jnp.reshape(inputs, (inputs.shape[0], inputs.shape[3], inputs.shape[4], inputs.shape[1] * inputs.shape[2]))
        actions = jnp.expand_dims(actions, axis=1)
        actions = jnp.expand_dims(actions, axis=1) / 18
        actions = jnp.tile(actions, (1, 96, 96, 1))
        assert_shape(inputs, (None, 96, 96, 96))
        assert_shape(actions, (None, 96, 96, 32))
        # TODO CONVERT TO FLOAT
        inputs = jnp.concatenate((inputs, actions), axis=3)
        assert_shape(inputs, (None, 96, 96, 128))
        # inputs = #nn.Sequential([
        inputs = nn.Conv(128, kernel_size=(3, 3), strides=2, padding=1)(inputs)
        inputs = nn.relu(inputs)
        inputs = ResBlock(inplanes=128, planes=128, stride=1)(inputs)
        inputs = ResBlock(inplanes=128, planes=128, stride=1)(inputs)
        inputs = nn.Conv(256, kernel_size=(3, 3), strides=2, padding=1)(inputs)
        inputs = nn.relu(inputs)
        inputs = ResBlock(inplanes=256, planes=256, stride=1)(inputs)
        inputs = ResBlock(inplanes=256, planes=256, stride=1)(inputs)
        inputs = ResBlock(inplanes=256, planes=256, stride=1)(inputs)
        # ])(inputs)
        inputs = nn.avg_pool(inputs, (2, 2), strides=(2, 2), padding="SAME")
        inputs = nn.Sequential([
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
        ])(inputs)
        inputs = nn.avg_pool(inputs, (2, 2), strides=(2, 2), padding="SAME")
        assert_shape(inputs, (None, 6, 6, 256))
        return inputs
    
class DynamicsNet(nn.Module):
    @nn.compact
    def __call__(self, hidden_state, action):
        action = jnp.expand_dims(action, axis=3)
        assert_shape(action, (None, 6, 6, 1))
        assert_shape(hidden_state, (None, 6, 6, 256))
        inputs = jnp.concatenate((action, hidden_state), axis=3)
        assert_shape(inputs, (None, 6, 6, 257))
        new_hidden = nn.Sequential([
            ResBlock(inplanes=257, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
            ResBlock(inplanes=256, planes=256, stride=1),
        ])(inputs)

        assert_shape(new_hidden, (None, 6, 6, 256))
        reward = nn.Sequential([
            nn.Conv(128, kernel_size=(3, 3), padding=1, strides=1),
            nn.relu,
            nn.Conv(64, kernel_size=(3, 3), padding=1, strides=1),
            nn.relu,
        ])(new_hidden)
        reward = jnp.transpose(reward, (0, 3, 1, 2))
        reward = jnp.reshape(reward, (reward.shape[0], -1))
        reward = nn.Sequential([
            nn.Dense(512),
            nn.relu,
            nn.Dense(601)
        ])(reward)
        return (new_hidden, reward)

class MuZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.device = device
        self.representation = RepresentationNet()
        self.dynamics = DynamicsNet()
        self.prediction = PredictionNet()

    def set_device(self, device):
      if (device != self.device):
        self.device = device

    def representation_net(self, params, inputs, actions):
        inputs = inputs#.to(self.device)
        actions = actions#.to(self.device)
        return self.representation.apply(params, inputs, actions, mutable=['batch_stats'])

    def dynamics_net(self, params, hidden_state, action):
        hidden_state = hidden_state#.to(self.device)
        action = action#.to(self.device)
        return self.dynamics.apply(params, hidden_state, action, mutable=['batch_stats'])

    def prediction_net(self, params, hidden_state):
        hidden_state = hidden_state#.to(self.device)
        return self.prediction.apply(params, hidden_state, mutable=['batch_stats'])

    def initialize_networks(self, key):
      key, key1, key2, key3 = random.split(key, num=4)
      x = random.normal(key1, (8, 1, 32, 96, 96, 3))
      y = random.normal(key2, (8, 1, 32))
      representation_init = jax.pmap(lambda *x: self.representation.init(*x))

      key3 = jl_utils.bcast_local_devices(key3)
      representation_params = representation_init(key3, x, y)
      key, key1, key2, key3 = random.split(key, num=4)
      x = random.normal(key2, (8, 1, 6, 6, 256)) 
      y = random.normal(key1, (8, 1, 6, 6)) 

      dynamics_init = jax.pmap(lambda *x: self.dynamics.init(*x))
      key3 = jl_utils.bcast_local_devices(key3)
      dynamics_params = dynamics_init(key3, x, y)

      key, key1, key2 = random.split(key, num=3)
      x = random.normal(key1, (8, 1, 6, 6, 256)) 
      prediction_init = jax.pmap(lambda *x: self.prediction.init(*x))
      key2 = jl_utils.bcast_local_devices(key2)
      prediction_params = prediction_init(key2, x)
      return (key, representation_params, dynamics_params, prediction_params)

    def initialize_networks_individual(self, key):
      key, key1, key2, key3 = random.split(key, num=4)
      x = random.normal(key1, (1, 32, 96, 96, 3))
      y = random.normal(key2, (1, 32))

      representation_params = self.representation.init(key3, x, y)
      key, key1, key2, key3 = random.split(key, num=4)
      x = random.normal(key2, (1, 6, 6, 256)) 
      y = random.normal(key1, (1, 6, 6)) 

      dynamics_params = self.dynamics.init(key3, x, y)

      key, key1, key2 = random.split(key, num=3)
      x = random.normal(key1, (1, 6, 6, 256)) 
      prediction_params = self.prediction.init(key2, x)
      return (key, representation_params, dynamics_params, prediction_params)


    def forward_observation(self, key, params, actions, observations):
        representation_params, _, _ = params 
        observations = observations#.to(self.device)
        actions = actions#.to(self.device)
        hidden_state, representation_params = self.representation.apply(representation_params, observations, actions, mutable=['batch_stats'])
        return self.forward_hidden_state(key, params, actions, hidden_state)

    def forward_hidden_state(self, key, params, actions, hidden_state):
        representation_params, dynamics_params, prediction_params = params 
        (value, policy), prediction_params = self.prediction.apply(prediction_params, hidden_state, mutable=['batch_stats'])
        key, subkey = random.split(key)
        new_actions = jax.random.categorical(subkey, policy)
        assert_shape(new_actions, [actions.shape[0]])
        new_actions = one_hot_encode(new_actions, new_actions.shape, hidden_state.shape[1:3])
        (new_state, reward), dynamics_params = self.dynamics.apply(dynamics_params, hidden_state, new_actions, mutable=['batch_stats'])
        params = (representation_params, dynamics_params, prediction_params)
        return (value, policy, reward, new_state, key)
