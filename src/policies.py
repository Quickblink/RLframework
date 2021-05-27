import torch
from torch.distributions.categorical import Categorical
import json


class PolicyManagerNew:
    def __init__(self, policies, priority_changelist, save_file):
        self.save_file = save_file
        self.priority_changelist = [*priority_changelist]
        self.priorities = torch.tensor(self.priority_changelist[0][1], dtype=torch.float)
        del self.priority_changelist[0]
        self.cat_dist = Categorical(self.priorities)
        self.policies = policies

    def report(self):
        for p in self.policies:
            p.report()

    def save_results(self):
        dump = {}
        for p in self.policies:
            name, rewards, lengths = p.save_stats()
            dump[name] = {'rewards': rewards[:-1], 'lengths': lengths[:-1]}
        with open(self.save_file, 'w') as config_file:
            json.dump(dump, config_file, indent=2)

    def update_priorities(self):
        total_length = 0
        for p in self.policies:
            total_length += p.get_total_length()
        if self.priority_changelist and total_length >= self.priority_changelist[0][0]:
            self.priorities = torch.tensor(self.priority_changelist[0][1], dtype=torch.float)
            del self.priority_changelist[0]
            self.cat_dist = Categorical(self.priorities)

    def get_policy(self):
        self.update_priorities()
        idx = self.cat_dist.sample().item()
        return self.policies[idx]


class PolicyBase:
    def __init__(self):
        self.name = type(self).__name__
        self.rewards = [[]]
        self.lengths = [[]]
        self.total_length = 0

    def get_actor(self):
        return self

    def enter_result(self, reward, length):
        self.rewards[-1].append(reward)
        self.lengths[-1].append(length)
        self.total_length += length

    def save_stats(self):
        self.rewards.append([])
        self.lengths.append([])
        return self.name, self.rewards, self.lengths

    def get_lengths(self):
        return self.lengths

    def get_total_length(self):
        return self.total_length

    def report(self):
        round_rewards = torch.tensor(self.rewards[-1], dtype=torch.float).mean().item()
        round_lengths = torch.tensor(self.lengths[-1], dtype=torch.float).mean().item()
        round_rewards10 = torch.tensor(sum(self.rewards[-10:], []), dtype=torch.float).mean().item()
        print(f'{self.name} : Avg Length over Round: {round_lengths:>3.1f} Reward: {round_rewards:>3.2f}'
              f' Reward (10 rounds): {round_rewards10:>3.2f} Episodes: {sum(map(len, self.rewards)):>5}')


class DeterministicPolicy(PolicyBase):
    def __call__(self, model_out):
        return model_out.argmax(dim=-1)


class RandomPolicy(PolicyBase):
    def __call__(self, model_out):
        return torch.randint(model_out.shape[-1], [1])


class GreedyPolicy(PolicyBase):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, model_out):
        proposal = model_out.argmax(dim=-1)
        action = torch.where((torch.rand_like(proposal, dtype=torch.float) > self.epsilon), proposal,
                             torch.randint_like(proposal, high=model_out.shape[-1]))
        return action


class AdvSwitchPolicy(PolicyBase):
    class Actor:
        def __init__(self, switch_point, actor1, actor2):
            self.switch_point = switch_point
            self.actor1 = actor1
            self.actor2 = actor2
            self.steps = 0

        def __call__(self, model_out):
            self.steps += 1
            return self.actor1(model_out) if self.steps < self.switch_point else self.actor2(model_out)

    def __init__(self, reference_policy, actor1, actor2):
        super().__init__()
        self.reference_policy = reference_policy
        self.actor1 = actor1
        self.actor2 = actor2

    def get_actor(self):
        ref_lengths = sum(self.reference_policy.get_lengths()[-3:], [])
        if len(ref_lengths) < 2:
            ref_lengths = [42, 42]
        idx = torch.randint(-len(ref_lengths), -1, [1])
        switch_point = ref_lengths[idx] - 10
        return self.Actor(switch_point, self.actor1, self.actor2)


class SwitchingPolicy(PolicyBase):
    def __init__(self, pol1, pol2, prob):
        super().__init__()
        self.pol1 = pol1
        self.pol2 = pol2
        self.prob = prob
        self.cur_one = True

    def __call__(self, model_out):
        if torch.rand([1]) < self.prob:
            self.cur_one = not self.cur_one
        return self.pol1(model_out) if self.cur_one else self.pol2(model_out)


class CertaintyPolicy(PolicyBase):
    def __init__(self, margin=1, base=0.02):
        super().__init__()
        self.margin = margin
        self.base = base

    def __call__(self, model_out):
        max, _ = model_out.max(dim=-1)
        scores = torch.relu(model_out - max + self.margin) + self.base
        return Categorical(scores).sample()


class GreedyDynamicPolicy(PolicyBase):
    def __call__(self, model_out):
        epsilon = max(0.1, 1-0.9*(self.total_length/1000000))
        proposal = model_out.argmax(dim=-1)
        action = torch.where((torch.rand_like(proposal, dtype=torch.float) > epsilon), proposal,
                             torch.randint_like(proposal, high=model_out.shape[-1]))
        return action
