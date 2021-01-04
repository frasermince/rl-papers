import gym
env = gym.make("Pong-v0")
observation = env.reset()
for _ in range(10000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()

class TrainingNet(LightningModule): 
    def __init__(self, lr):
        self.env = gym.make("Pong-v0")
        self.lr = 0.01
        observation = env.reset()
        self.history_frames = [observation]
        self.dicount_factor = 0.9
        super().__init__()

    def forward(self, observation):
        return self.env.action_space.sample()

    def training_step(self, data, batch_idx):
        self.env.render()
        action = self(self.history_frames)
        observation, reward, done, info = self.env.step(action)
        find_new_history(observation)
        yj = reward if done else reward + (self.discount_factor)
        loss = 

        return loss

    def find_new_history(self, observation):
        self.history_frames.append(observation)
        if self.history_frames.length > 4:
            self.history_frames.pop(0)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
