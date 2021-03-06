import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

class OptWrapper:
    def __init__(self, params, CONFIG):
        self.opt = Adam(params, lr=CONFIG.START_LR)
        self.scheduler = StepLR(self.opt, CONFIG.UPDATE_RATE, CONFIG.LR_DECAY)
        self.params = list(params)
        self.final_lr = CONFIG.FINAL_LR

    def get_grad_norm(self):
        norm = 0
        for p in self.params:
            norm += (p.grad.norm().item()) ** 2
        return norm ** (1 / 2)

    def step(self):
        self.opt.step()
        if self.scheduler.get_last_lr()[0] > self.final_lr:
            self.scheduler.step()

    def zero_grad(self):
        self.opt.zero_grad()



class StateContainer:
    def __init__(self, init_state, add_dims=None, device=None, state_storage_type=None):
        self.state = init_state if not (add_dims or device or state_storage_type) else self._make(init_state, add_dims, device, state_storage_type)

    def _make(self, state, add_dims, device, state_storage_type):
        if isinstance(state, dict):
            new_log = {k: self._make(v, add_dims, device, state_storage_type) for k, v in state.items()}
        elif isinstance(state, (list, tuple)):
            new_log = [self._make(v, add_dims, device, state_storage_type) for v in state]
        elif isinstance(state, torch.Tensor):
            new_log = torch.empty(add_dims + state.shape[1:], device=device, dtype=state_storage_type) #state.to(device).reshape(add_dims + state.shape)
        else:
            raise Exception('Unknown type in model state!')
        return new_log

    def transfer(self, other):
        return self._transfer(self.state, other)

    def _transfer(self, state, other):
        if isinstance(state, dict):
            for k in state:
                for state1, other1 in self._transfer(state[k], other[k]):
                    yield state1, other1
        elif isinstance(state, (list, tuple)):
            for k in range(len(state)):
                for state1, other1 in self._transfer(state[k], other[k]):
                    yield state1, other1
        elif isinstance(state, torch.Tensor):
            #print('yields')
            yield state, other
        else:
            raise Exception('Unknown type in model state!')


    def get(self, fn):
        return self._get(self.state, fn)

    def _get(self, state, fn):
        if isinstance(state, dict):
            new_log = {k: self._get(v, fn) for k, v in state.items()}
        elif isinstance(state, (list, tuple)):
            new_log = [self._get(v, fn) for v in state]
        elif isinstance(state, torch.Tensor):
            new_log = fn(state)
        else:
            raise Exception('Unknown type in model state!')
        return new_log

