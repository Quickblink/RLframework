import torch
from time import sleep
from .utils import StateContainer


class DataManager:
    def __init__(self, policy_manager, env, data_net, comms, CONFIG):
        data_device = torch.device('cpu')  # hard coded because gpu is not really useful/possible
        self.buffer = GeneratorBufferMulti(env, data_device, data_net.get_initial_state(1), CONFIG)
        self.policy_manager = policy_manager
        self.worker = DataWorker(env, data_net, policy_manager, data_device, CONFIG)
        self.data_net = data_net

        self.train_to_data = comms['train_to_data']
        self.data_to_train = comms['data_to_train']
        self.condition = comms['condition']

    def run(self):
        while True:
            finished_episodes = self.worker.one_step()
            if self.buffer.above_min() and not self.train_to_data.empty():
                self.sync()
            for ep in finished_episodes:
                self.buffer.enter_episode(*ep)
                while self.buffer.full() and self.train_to_data.empty():
                    sleep(0.1)
                if self.buffer.above_min() and not self.train_to_data.empty():
                    self.sync()

    def sync(self):
        with self.condition:
            self.data_net.load_state_dict(self.train_to_data.get())
            do_print = self.train_to_data.get()
            self.data_to_train.put(self.buffer.get_contents())
            self.condition.notify_all()
            self.condition.wait()
            self.buffer.reset(self.train_to_data.get())
            if do_print:
                self.policy_manager.report()
                self.policy_manager.save_results()


class DataWorker:
    def __init__(self, env, model, policy_manager, store_device, CONFIG):
        self.model = model
        self.env = env
        self.num_envs = env.num_envs
        self.policy_manager = policy_manager
        self.store_device = store_device
        self.net_device = CONFIG.DATA_DEVICE
        self.state_storage_type = CONFIG.state_storage_type
        self.frame_stack = CONFIG.FRAME_STACK

        self.position = 0
        self.starts = [0] * self.num_envs
        self.policies = [None] * self.num_envs
        self.actors = [None] * self.num_envs
        for i in range(self.num_envs):
            self.policies[i] = self.policy_manager.get_policy()
            self.actors[i] = self.policies[i].get_actor()

        self.capacity = self.env.max_steps + 2
        self.inputs = env.make_obs_storage([self.capacity, self.num_envs], store_device)
        self.actions = torch.empty([self.capacity, self.num_envs], device=store_device, dtype=torch.long)
        self.rewards = torch.empty([self.capacity, self.num_envs], device=store_device)
        self.model_state = StateContainer(model.get_initial_state(1), (self.capacity, self.num_envs), store_device, CONFIG.state_storage_type)
        self.cur_model_state = StateContainer(self.model.get_initial_state(self.num_envs))
        for container, entry in self.model_state.transfer(self.cur_model_state.state):
            container[0] = entry.detach()
        self.inputs[0] = self.env.reset()


    def one_step(self):
        with torch.no_grad():
            inputs = self.inputs[
                torch.arange(self.position - self.frame_stack + 1, self.position + 1, device=self.store_device)]
            inputs = self.env.postprocess(inputs.to(self.net_device))
            y, self.cur_model_state.state = self.model(inputs, self.cur_model_state.state)
            out = y.to(self.store_device).squeeze(0)
            actions = torch.empty([self.num_envs], dtype=torch.long)

            for i in range(self.num_envs):
                actions[i] = self.actors[i](out[i]) if (self.position - self.starts[i]) % self.capacity >= self.frame_stack - 1 else self.env.start_action
                #assert (self.position - self.starts[i])%self.capacity >= self.frame_stack - 1, (self.position, self.starts[i], self.position - self.starts[i], (self.position - self.starts[i])%self.capacity)

            obs, r, t, count = self.env.step(actions)
            self.actions[self.position] = actions
            self.rewards[self.position] = torch.from_numpy(r)
            self.position = (self.position + 1) % self.capacity
            finished_episodes = []
            for i in range(self.num_envs):
                if t[i]:
                    finished_episodes.append(self._package_episode(i))
                    self.policies[i] = self.policy_manager.get_policy()
                    self.actors[i] = self.policies[i].get_actor()
                    self.starts[i] = self.position
                if (self.position - self.starts[i]) % self.capacity == self.frame_stack - 1:
                    for container, entry in self.cur_model_state.transfer(self.model.get_initial_state(1)):
                        container[i] = entry[0].detach()

            for container, entry in self.model_state.transfer(self.cur_model_state.state):
                container[self.position] = entry.to(self.state_storage_type)
            self.inputs[self.position] = obs
        return finished_episodes

    def _package_episode(self, i):
        if self.position < self.starts[i]:
            idx = torch.arange(self.starts[i]-self.capacity, self.position, device=self.store_device)
        else:
            idx = slice(self.starts[i], self.position)
        rewards = self.rewards[idx, i]
        self.policies[i].enter_result(rewards.sum().item(), len(rewards))
        return self.inputs[idx, i], self.model_state.get(lambda x: x[idx, i]), self.actions[idx, i], rewards


class GeneratorBufferMulti:
    def __init__(self, env, device, model_state, CONFIG):
        self.frame_stack = CONFIG.FRAME_STACK
        self.capacity = CONFIG.CREATION_BUFFER
        self.inputs = env.make_obs_storage([self.capacity], device)
        self.actions = torch.empty([self.capacity], device=device, dtype=torch.long)
        self.rewards = torch.empty([self.capacity], device=device)
        self.terminals = torch.empty([self.capacity], device=device, dtype=torch.bool)
        self.prios = torch.empty([self.capacity+CONFIG.N_SEQ+self.frame_stack])
        self.position = 0
        self.n_seq = CONFIG.N_SEQ
        self.model_state = StateContainer(model_state, (self.capacity,), device, CONFIG.state_storage_type)
        self.min_per_update = CONFIG.DATA_MIN * CONFIG.TRANSFER_RATE
        self.full_after = CONFIG.DATA_MAX * CONFIG.TRANSFER_RATE


    def get_contents(self):
        self.prios[self.position:self.position+self.n_seq+self.frame_stack] = 0
        return self.inputs, self.model_state.state, self.actions, self.rewards, self.terminals, self.prios, self.position

    def reset(self, buffer_return):
        self.inputs, self.model_state.state, self.actions, self.rewards, self.terminals, self.prios, _ = buffer_return
        self.position = 0

    def above_min(self):
        return self.position > self.min_per_update

    def full(self):
        return self.position > self.full_after

    def build_simple_sampling(self, input_size):
        prio = torch.ones([input_size])
        offset = self.frame_stack + self.n_seq - 2
        prio[:offset] = 0
        return prio


    def enter_episode(self, input, state, action, reward):
        end = self.position + len(input)
        if end > self.capacity:
            raise Exception('DataBuffer overflowing!')
        self.inputs[self.position:end] = input
        self.rewards[self.position:end] = reward
        self.actions[self.position:end] = action
        self.terminals[self.position:end-1] = False
        self.terminals[end-1] = True
        self.prios[self.position:end] = self.build_simple_sampling(len(input))
        for container, entry in self.model_state.transfer(state):
            container[self.position:end] = entry
        self.position = end
