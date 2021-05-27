import numpy as np
import torch
import gym
import torchvision.transforms as T
from PIL import Image
from threading import BoundedSemaphore

#TODO: andere envs anpassen

ui_protect = BoundedSemaphore()


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class VisionCartPole(gym.Env):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.observation_space = gym.spaces.Box(low=np.float32(0), high=np.float32(1), shape=(3, 40, 90))
        self.action_space = gym.spaces.Discrete(2)


    def reset(self):
        self.counter = 0
        self.env.reset()
        return self.get_screen()

    def step(self, action):
        self.counter += 1
        _, r, t, inf = self.env.step(action)
        return self.get_screen(), r, t, self.counter

    def get_cart_location(self, screen_width):
        world_width = self.env.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        with ui_protect:
            screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).numpy()#.unsqueeze(0)#.to(device)


gym.register('CartPoleVision-v0', entry_point=VisionCartPole)


class CartPoleMulti:
    def __init__(self, frame_stack, num_envs):
        dummy_env = gym.make('CartPole-v0')
        self.env = None#gym.vector.make('ModifiedBreakout-v0', num_envs=num_envs, asynchronous=True)
        self.input_shape = [3, 40, 90]
        self.n_out = 2
        self.frame_stack = frame_stack
        self.max_steps = dummy_env._max_episode_steps
        self.num_envs = num_envs
        self.start_action = 0

        #self.storage_type = torch.float

    def make_obs_storage(self, dims, device):
        return torch.zeros(dims+self.input_shape, dtype=torch.float, device=device)

    def postprocess(self, inputs):
        return inputs


    #def preprocess(self, raw):
    #    return (torch.from_numpy(raw)[26::2, 4:-4:2].float().mean(dim=-1))/255 #make 4 steps in first dimension

    def _preprocess(self, raw):
        return torch.from_numpy(raw)

    def reset(self):
        self.env = gym.vector.make('CartPoleVision-v0', num_envs=self.num_envs, asynchronous=False)
        return self._preprocess(self.env.reset())

    def step(self, actions):
        raw, r, t, inf = self.env.step(actions.numpy())
        return self._preprocess(raw), r, t, None


class CartPoleNoVisionMulti:
    def __init__(self, frame_stack, num_envs):
        dummy_env = gym.make('CartPole-v0')
        self.env = None#gym.vector.make('ModifiedBreakout-v0', num_envs=num_envs, asynchronous=True)
        self.input_shape = [4]
        self.n_out = 2
        self.frame_stack = frame_stack
        self.max_steps = dummy_env._max_episode_steps
        self.num_envs = num_envs
        self.start_action = 0

        self.storage_type = torch.float

    def make_obs_storage(self, dims, device):
        return torch.zeros(dims+self.input_shape, dtype=torch.float, device=device)

    def postprocess(self, inputs):
        return inputs


    #def preprocess(self, raw):
    #    return (torch.from_numpy(raw)[26::2, 4:-4:2].float().mean(dim=-1))/255 #make 4 steps in first dimension

    def _preprocess(self, raw):
        return torch.from_numpy(raw)

    def reset(self):
        self.env = gym.vector.make('CartPole-v0', num_envs=self.num_envs, asynchronous=False) #TODO: async here?
        return self._preprocess(self.env.reset())

    def step(self, actions):
        raw, r, t, inf = self.env.step(actions.numpy())
        return self._preprocess(raw), r, t, None


