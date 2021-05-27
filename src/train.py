import numpy as np
import torch
from .utils import StateContainer
from time import time


class SumTreeTorch:

    def __init__(self, capacity, device):
        self._num_levels = int(np.ceil(np.log2(capacity)))
        self._num_nodes = int(2 ** self._num_levels) #-1
        self.tree = torch.zeros([self._num_nodes + capacity], device=device)
        self.capacity = capacity
        self.device = device

    @property
    def total_sum(self):
        return self.tree[1]

    def inspect_tree(self):
        return self.tree[self._num_nodes:]


    def sample(self, size):
        r = (torch.rand(size, device=self.device) + torch.arange(size, device=self.device)) * (self.total_sum / size) #
        idx = torch.ones(size, device=self.device, dtype=torch.long)
        for _ in range(self._num_levels):
            idx *= 2
            b = ((r > self.tree[idx]) & (self.tree[idx + 1] != 0))  # this can happen due to imprecisions
            r -= self.tree[idx] * b
            idx += b

        return idx - self._num_nodes  # , self.tree[idx] / self.total_sum + 1e-16


    def update_range(self, vs, start):
        a, b = self._num_nodes + start, self._num_nodes + start + len(vs)
        self.tree[a:b] = vs
        while b >= 3:
            a, b = a//2, (b+1)//2
            c, d = a * 2, b * 2
            self.tree[a:b] = self.tree[c:d].reshape((-1, 2)).sum(dim=1)



class SumTree:

    def __init__(self, capacity, device):
        self._num_levels = int(np.ceil(np.log2(capacity)))
        self._num_nodes = int(2 ** self._num_levels) #-1
        self.tree = np.zeros(self._num_nodes + capacity, dtype=np.float64)
        self.capacity = capacity
        self.max_value = 1e-8

    @property
    def total_sum(self):
        return self.tree[1]

    def inspect_tree(self):
        return self.tree[self._num_nodes:]

    def update(self, i, v):
        assert not (np.unique(i, return_counts=True)[1] > 1).any()
        #i, i_idx = np.unique(i, return_index=True)
        #v = v[i_idx]
        i = i + self._num_nodes
        d = v - self.tree[i]
        self.tree[i] = v

        self.max_value = max(self.max_value, np.max(v))

        for _ in range(self._num_levels):
            i = (i) // 2 # - 1
            np.add.at(self.tree, i, d)

    def sample(self, size):
        r = (np.random.ranf(size) + np.arange(size)) * (self.total_sum / size)
        idx = np.ones(size, dtype=np.long)
        for _ in range(self._num_levels):
            idx *= 2
            #idx += 1
            b = np.logical_and(r > self.tree[idx], self.tree[idx + 1] != 0)  # this can happen due to imprecisions
            r -= self.tree[idx] * b
            idx += b

        return torch.from_numpy(idx - self._num_nodes)  # , self.tree[idx] / self.total_sum + 1e-16

    def recompute_tree(self):
        a, b, c = self._num_nodes // 2, self._num_nodes, len(self.tree)
        self.tree[a:b] = np.pad(np.sum(self.tree[b:c].reshape(((c - b) // 2, 2)), axis=1), (0, (b - a - (c - b) // 2)), 'constant')
        a, b, c = a // 2, a, b
        while b >= 1:
            self.tree[a:b] = np.sum(self.tree[b:c].reshape((a + 1, 2)), axis=1)
            a, b, c = a//2, a, b

    def update_range(self, vs, start):
        #assert start + len(vs) <= self.capacity, (start, len(vs))
        a, b = self._num_nodes + start, self._num_nodes + start + len(vs)
        self.tree[a:b] = vs
        while b >= 3:
            a, b = a//2, (b+1)//2
            c, d = a * 2, b * 2
            self.tree[a:b] = np.sum(self.tree[c:d].reshape((-1, 2)), axis=1)







class SamplingBuffer:
    def __init__(self, EnvClass, model_state, CONFIG):
        self.frame_stack = CONFIG.FRAME_STACK
        self.n_seq = CONFIG.N_SEQ
        self.device = CONFIG.SAMPLING_DEVICE
        self.state_storage_type = CONFIG.state_storage_type

        self.capacity = CONFIG.BUFFER_CAPACITY
        self.inputs = EnvClass.make_obs_storage([self.capacity + 1], self.device)
        self.actions = torch.empty([self.capacity], device=self.device, dtype=torch.long)
        self.rewards = torch.empty([self.capacity], device=self.device)
        self.terminals = torch.empty([self.capacity], device=self.device, dtype=torch.bool)
        self.model_state = StateContainer(model_state, (self.capacity + 1,), self.device, CONFIG.state_storage_type)
        self.priority_queue = SumTree(self.capacity, torch.device('cpu'))
        self.position = 0
        self.size = 0


    def enter_block(self, input, state, action, reward, terminals, prio, length):
        end = self.position + length
        if end > self.capacity:
            self.position = 0
            end = length
        self.inputs[self.position:end] = input[:length].to(self.device)
        self.rewards[self.position:end] = reward[:length].to(self.device)
        self.actions[self.position:end] = action[:length].to(self.device)
        self.terminals[self.position:end] = terminals[:length].to(self.device)
        for container, entry in self.model_state.transfer(state):
            container[self.position:end] = entry[:length]
        prio_end = min(end + self.frame_stack + self.n_seq, self.capacity)
        prio_src_end = prio_end - self.position

        self.priority_queue.update_range(prio[:prio_src_end], self.position) #.to(self.device)

        self.position = end
        self.size = max(self.position, self.size)




    def make_batch(self, batch_size, device, state_type):
        idx = self.priority_queue.sample(batch_size)#.to(device) #torch.ones([batch_size], device=torch.device('cpu'), dtype=torch.long)
        self.inp_idx = idx.unsqueeze(0) + torch.arange(-(self.n_seq + self.frame_stack)+2, 2, device=idx.device).unsqueeze(1)
        rest_idx = self.inp_idx[self.frame_stack-1:-1]
        #assert (self.priority_queue.inspect_tree()[idx] > 0).all()
        #assert ((inp_idx > -self.capacity) & (inp_idx < self.capacity)).all(), inp_idx
        inputs = self.inputs[self.inp_idx].to(device, non_blocking=True)
        actions = self.actions[rest_idx].to(device, non_blocking=True)
        rewards = self.rewards[rest_idx].to(device, non_blocking=True)
        terminals = self.terminals[idx].to(device, non_blocking=True)
        #assert not self.terminals[rest_idx[:-1]].any()
        #assert not self.terminals[self.inp_idx[:-2]].any()
        state_idx = idx - self.n_seq + 1
        target_state_idx = idx - self.n_seq + 2

        state = self.model_state.get(lambda x: x[state_idx].to(device, state_type, non_blocking=True))
        target_state = self.model_state.get(lambda x: x[target_state_idx].to(device, state_type, non_blocking=True))

        return inputs, state, actions, rewards, terminals, target_state

    def enter_hidden_state(self, state):
        for container, entry in self.model_state.transfer(state):
            container[self.inp_idx[self.frame_stack+1:]] = entry[:-1].detach().to(self.device, self.state_storage_type)



class Trainer:
    def __init__(self, train_net, target_net, EnvClass, comms, gamma, sampling_buffer, opt, criterion):
        self.train_net = train_net
        self.target_net = target_net
        self.train_to_data = comms['train_to_data']
        self.data_to_train = comms['data_to_train']
        self.condition = comms['condition']
        self.sampling_buffer = sampling_buffer
        self.second_cuda_stream = torch.cuda.Stream()
        self.criterion = criterion
        self.opt = opt
        self.EnvClass = EnvClass
        self.gamma = gamma

    def sync(self, do_print):
        with self.condition:
            self.train_to_data.put(self.train_net.state_dict())
            self.train_to_data.put(do_print)
            self.condition.wait()
            data = self.data_to_train.get()
            self.sampling_buffer.enter_block(*data)
            self.train_to_data.put(data)
            self.condition.notify_all()

    def train_one(self, batch, print_grads=False):
        for p in self.train_net.parameters():
            p.grad = None
        inputs, state, actions, rewards, terminals, target_state = batch
        inputs = self.EnvClass.postprocess(inputs)

        self.second_cuda_stream.wait_stream(torch.cuda.default_stream())
        with torch.no_grad(), torch.cuda.stream(self.second_cuda_stream):
            t_out, h = self.target_net(inputs[1:], target_state)
            pred_targets, _ = t_out[-1].max(dim=-1)
            #proc_t_out = t_out[-1] # TODO: remove risk averse update rule
            #_, max_ind = proc_t_out.max(dim=-1)
            #pred_targets = (torch.tensor([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]], device=inputs.device)[max_ind] * proc_t_out).sum(dim=-1)


            targets = self.gamma * pred_targets * (1 - terminals.float()) + rewards[-1].clamp(
                max=1)  # TODO: check this again, move this to env
            state_con = StateContainer(h)
            #print(state_con.get(lambda t:t.shape))
            for container, entry in state_con.transfer(
                    self.target_net.get_initial_state(int(terminals.float().sum().item()))):
                #print(container.shape, entry.shape, terminals.shape)
                container[-2][terminals] = entry

        out, _ = self.train_net(inputs[:-1], state)

        q_values = out[-1].gather(-1, actions[-1].unsqueeze(-1)).squeeze(-1)
        test_var = out.var().item()
        test_mean = out.mean().item()
        max_loss = (q_values - targets).abs().max()

        torch.cuda.default_stream().wait_stream(self.second_cuda_stream)

        self.sampling_buffer.enter_hidden_state(state_con.state)

        loss = self.criterion(q_values, targets.detach())
        loss.backward()
        if print_grads:
            for n, p in self.train_net.named_parameters():
                if p.grad is not None:
                    print(f'{n:<50} {p.grad.norm().item():.1e} | {p.grad.var().sqrt().item():.1e}')
        self.opt.step()
        return loss.item(), test_var, test_mean, max_loss.item()

    def run(self, run_id, CONFIG):
        outstart = time()

        # test = q_to_train.get()
        # print('???')
        while self.sampling_buffer.size < CONFIG.START_TRAINING_AFTER:
            self.sync(False)

        print(time() - outstart)

        i = 1
        round_count = 0
        loss_sum = 0
        max_loss_outer = 0
        round_start = time()
        print_results = False
        while True:
            current_batch = self.sampling_buffer.make_batch(CONFIG.BATCH_SIZE, CONFIG.TRAIN_DEVICE, CONFIG.state_type)
            loss, test_var, test_mean, max_loss = self.train_one(current_batch, print_grads=False)  # (i%100==0)
            max_loss_outer = max(max_loss, max_loss_outer)
            loss_sum += loss
            if i % CONFIG.UPDATE_RATE == 0:
                self.target_net.load_state_dict(self.train_net.state_dict())
                print(f'Round: {round_count:>3} Time: {time() - round_start:>3.1f}'
                      f' Loss: {loss_sum / CONFIG.UPDATE_RATE:.2e} Var: {test_var:.2e} Mean: {test_mean:.2e}'
                      f' Max Dev: {max_loss_outer:.2e}')
                loss_sum = 0
                max_loss_outer = 0
                round_start = time()
                round_count += 1
                torch.save(self.train_net, f'saves/models/model_{run_id}')
                print_results = True
            if i % CONFIG.TRANSFER_RATE == 0:
                self.sync(print_results)
                print_results = False
            i += 1