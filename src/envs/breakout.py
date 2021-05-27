import gym
import torch
import numpy as np


# Define a valid gym environment. This modifies Breakout by adding a terminal signal at every death and resetting
# when stuck in a loop. The functions reset and step as well as the observation_space and action_space are required.
class ModifiedBreakout(gym.Env):
    def __init__(self):
        self.env = gym.make('Breakout-v4')
        self.env.env.frameskip = (3, 5)
        self.time_since_last_reward = 0
        self.counter = 0
        self.observation_space = gym.spaces.Box(low=np.float32(0), high=np.float32(255), shape=(92,76))
        self.action_space = self.env.action_space
        self.env._max_episode_steps = 10000
        self._max_episode_steps = self.env._max_episode_steps
        self.max_without_reward = 400

    def _preprocess(self, raw):
        return np.mean(raw[26::2, 4:-4:2], axis=-1).astype(np.uint8)

    def reset(self):
        if self.env.unwrapped.ale.game_over() or self.counter == 0:
            self.env.reset()
        self.counter = 0
        self.time_since_last_reward = 0
        valid = False
        while not valid:
            self.env.unwrapped.ale.act(1)
            obs = self._preprocess(self.env.unwrapped._get_obs())
            valid = obs[34:-10, 2:-2].sum() > 0

        return obs

    def step(self, action):
        num_lives = self.env.unwrapped.ale.lives()
        raw, r, _, inf = self.env.step(action+1)
        while self.time_since_last_reward >= self.max_without_reward and self.env.ale.lives() == num_lives:
            self.env.unwrapped.ale.act(np.random.randint(1,4))
        t = self.env.ale.lives() < num_lives
        self.time_since_last_reward = (self.time_since_last_reward + 1) if r == 0 else 0
        self.counter += 1
        return self._preprocess(raw), r, t, self.counter

    def render(self, mode='human'):
        return self.env.render(mode)


# The new gym env has to be registered.
gym.register('ModifiedBreakout-v0', entry_point=ModifiedBreakout)


# Another wrapper has to be defined to use the environment in this code. It contains a gym vector environment to run
# multiple environments in parallel. It also contains data processing interfaces.
class BreakoutWrapper:
    input_shape = (1, 92, 76)
    n_out = 3
    start_action = 0
    max_steps = 10000

    def __init__(self, num_envs):
        self.env = gym.vector.make('ModifiedBreakout-v0', num_envs=num_envs, asynchronous=True)
        self.num_envs = num_envs

    # This method returns a storage for the input samples with dimensions of a single sample plus additional dimensions
    # given as input. It has to support index operations like '__get__'.
    @staticmethod
    def make_obs_storage(dims, device):
        return torch.zeros(tuple(dims) + BreakoutWrapper.input_shape, dtype=torch.uint8, device=device)

    # Samples are processed using the function before being handed to the model (after being in storage)
    @staticmethod
    def postprocess(inputs):
        return inputs.float()/255

    # Processing before storing samples in replay buffer
    def _preprocess(self, raw):
        return torch.from_numpy(raw).unsqueeze(1)

    def reset(self):
        return self._preprocess(self.env.reset())

    def step(self, actions):
        raw, r, t, inf = self.env.step(actions)
        return self._preprocess(raw), r, t, None

