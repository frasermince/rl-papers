import jax
import envpool

env = envpool.make("Pong-v5", env_type="gym", num_envs=64, batch_size=16)
# env.async_reset()
handle, recv, send, _ = env.xla()


handle2, data = recv(handle)


