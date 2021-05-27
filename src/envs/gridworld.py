import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import gym
import torch
from skimage.color import rgb2gray
from PIL import Image

def imresize(arr, size, interp=None):
    return np.array(Image.fromarray(arr).resize(size, resample=Image.NEAREST))


class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class gameEnv():
    def __init__(self,partial,size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        a = self.reset()
        plt.imshow(a,interpolation="nearest")
        
        
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        bug = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug)
        hole = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole)
        bug2 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug2)
        hole2 = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole2)
        bug3 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug3)
        bug4 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug4)
        state = self.renderEnv()
        self.state = state
        return state

    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1     
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize
    
    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(),1,1,1,1,'goal'))
                else: 
                    self.objects.append(gameOb(self.newPosition(),1,1,0,-1,'fire'))
                return other.reward,False
        if ended == False:
            return 0.0,False

    def renderEnv(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY+2,self.sizeX+2,3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        b = imresize(a[:,:,0],[84,84],interp='nearest')#[..., np.newaxis]
        c = imresize(a[:,:,1],[84,84],interp='nearest')#[..., np.newaxis]
        d = imresize(a[:,:,2],[84,84],interp='nearest')#[..., np.newaxis]
        a = np.stack([b,c,d],axis=2)
        return a

    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done
        else:
            return state,(reward+penalty),done




class ModifiedGridworld(gym.Env):
    def __init__(self):
        self.env = gameEnv(partial=True, size=9)
        self.time_since_last_reward = 0
        #self.observation_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([92,76]))
        self.observation_space = gym.spaces.Box(low=np.float32(0), high=np.float32(1), shape=(84,84))
        self.action_space = gym.spaces.Discrete(4)
        self.counter = 0
        self._max_episode_steps = 50
        #os.nice(15)
        #self._max_episode_steps = self.env._max_episode_steps

    def _preprocess(self, raw):
        return (rgb2gray(raw)*255).astype(np.uint8)
        #return np.transpose(raw, (2, 0, 1)).astype(np.uint8)

    def reset(self):
        self.counter = 0
        return self._preprocess(self.env.reset())

    def step(self, action):
        self.counter += 1
        raw, r, t = self.env.step(action)
        terminal = t or self.counter >= self._max_episode_steps
        return self._preprocess(raw), r, terminal, None



gym.register('ModifiedGridworld-v0', entry_point=ModifiedGridworld)

class GridWorldMultiWrapper:
    def __init__(self, frame_stack, num_envs):
        dummy_env = gym.make('ModifiedGridworld-v0')
        self.env = None#gym.vector.make('ModifiedBreakout-v0', num_envs=num_envs, asynchronous=True)
        self.input_shape = [1, 84, 84]
        self.n_out = 4
        self.frame_stack = frame_stack
        self.max_steps = dummy_env._max_episode_steps
        self.num_envs = num_envs
        self.start_action = 0
        #self.storage_type = torch.uint8

    def make_obs_storage(self, dims, device):
        return torch.zeros(dims+self.input_shape, dtype=torch.uint8, device=device)

    def postprocess(self, inputs):
        return inputs.float()/255


    #def preprocess(self, raw):
    #    return (torch.from_numpy(raw)[26::2, 4:-4:2].float().mean(dim=-1))/255 #make 4 steps in first dimension

    def _preprocess(self, raw):
        return torch.from_numpy(raw).unsqueeze(1)

    def reset(self):
        self.env = gym.vector.make('ModifiedGridworld-v0', num_envs=self.num_envs, asynchronous=True)
        return self._preprocess(self.env.reset())

    def step(self, actions):
        raw, r, t, inf = self.env.step(actions)
        return self._preprocess(raw), r, t, None