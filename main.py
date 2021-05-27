import torch
from src.data import DataManager
from src.policies import PolicyManagerNew, DeterministicPolicy, GreedyPolicy, AdvSwitchPolicy, RandomPolicy
from src.train import SamplingBuffer, Trainer
from src.envs.breakout import BreakoutWrapper
from src.utils import OptWrapper
import sys
import torch.multiprocessing as mp
import rnnbuilder as rb


run_id = sys.argv[1]


class CONFIG:
    FRAME_STACK = 1  # amounts of input steps stacked into one input, put to 1 to disable frame stacking
    NUM_ENVS = 16  # number of environments to be run in parallel for data generation, they share one network for parallel evaluation
    BATCH_SIZE = 64  # batch size for training only
    N_SEQ = 8  # number of elements in a sequence used for training recurrent networks, error signals are only generated for the last element and propagated for this many steps backward
    TRAIN_DEVICE = torch.device('cuda')  # device used for performing the main training loop
    SAMPLING_DEVICE = torch.device('cuda')  # where to store the replay buffer, may require significant gpu memory
    DATA_DEVICE = torch.device('cuda')  # where to run the network for data generation
    GAMMA = 0.99  # reward decay parameter from classic q learning
    UPDATE_RATE = 3000  # after how many training iterations the target network is updated
    BUFFER_CAPACITY = 500000  # capacity of the replay buffer
    START_LR = 1e-4  # base learning rate
    FINAL_LR = 1e-5  # decay stops here
    LR_DECAY = 0.998  # learning rate is multiplied by this factor every UPDATE_RATE iterations
    START_TRAINING_AFTER = 50000  # training starts when the replay buffer is filled to this point
    RANDOM_UNTIL = 100000  # a random policy is used for the first iterations, see policies below
    DATA_MIN = 5  # minimum of data generated per training iteration
    DATA_MAX = 10  # maximum of data generated per training iteration
    TRANSFER_RATE = 1000  # after how many iterations training and data generation are synced, one process may wait until the above minimum and maximum are satisfied
    CREATION_BUFFER = 15005  # buffer size for data generation, should be able to hold an episode of maximum length, needs to fit in shared memory
    state_storage_type = torch.float16  # State is converted for storage for saving space
    state_type = torch.float32  # This is the recovery type


# Testing parameters for weaker machines to test code

class CONFIG_HOME:
    FRAME_STACK = 3
    NUM_ENVS = 4
    BATCH_SIZE = 32
    N_SEQ = 5
    TRAIN_DEVICE = torch.device('cuda')
    SAMPLING_DEVICE = torch.device('cuda')
    DATA_DEVICE = torch.device('cuda')
    GAMMA = 0.99
    UPDATE_RATE = 500
    BUFFER_CAPACITY = 20000#200000
    START_LR = 1e-4  # base learning rate
    FINAL_LR = 1e-5  # decay stops here
    LR_DECAY = 0.998
    START_TRAINING_AFTER = 5000
    RANDOM_UNTIL = 30000
    DATA_MIN = 5
    DATA_MAX = 10
    TRANSFER_RATE = 500
    CREATION_BUFFER = 15005
    state_storage_type = torch.float16
    state_type = torch.float32

if run_id == 'home':
    CONFIG = CONFIG_HOME

EnvClass = BreakoutWrapper# multi-environment class, see /envs/breakout.py


det_pol = DeterministicPolicy()
policies = [  # The set of policies used for data generation
    det_pol,
    AdvSwitchPolicy(det_pol, DeterministicPolicy(), GreedyPolicy(0.1)),  # Switches to a greedy policy later in an episode
    GreedyPolicy(0.1),
    RandomPolicy()
]
change_list = [  # priority of the above policies over time (played steps)
    (0, [0, 0, 0, 1]),  # in the beginning only random policy
    (CONFIG.RANDOM_UNTIL, [1, 1, 1, 0])  # after that the other 3 policies are equally played
]


CONV_NEURON = rb.nn.ReLU()

conv_stack = rb.Sequential(
    rb.rnn.TempConvPlus2d(out_channels=32, kernel_size=8, stride=4, time_kernel_size=CONFIG.FRAME_STACK), CONV_NEURON,
    rb.nn.Conv2d(out_channels=64, kernel_size=4, stride=2), CONV_NEURON,
    rb.nn.Conv2d(out_channels=64, kernel_size=3, stride=1), CONV_NEURON)


ll_lstm = rb.rnn.LSTM(512)

ll_ffann = rb.Sequential(rb.nn.Linear(512), rb.nn.ReLU())


LAST_LAYER = ll_lstm

factory = rb.Sequential(conv_stack, LAST_LAYER, rb.nn.Linear(EnvClass.n_out))


def make_model():
    return factory.make_model(EnvClass.input_shape)




def data_process(comms):
    env = EnvClass(CONFIG.NUM_ENVS)
    data_net = make_model().to(CONFIG.DATA_DEVICE)

    policy_manager = PolicyManagerNew(policies, change_list, f'saves/results/results_{run_id}.json')

    dataM = DataManager(policy_manager, env, data_net, comms, CONFIG)

    dataM.run()




if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    data_to_train = ctx.Queue()
    train_to_data = ctx.Queue()
    condition = ctx.Condition()
    comms = {
        'data_to_train': data_to_train,
        'train_to_data': train_to_data,
        'condition': condition
    }
    dp = ctx.Process(target=data_process, args=(comms,))
    dp.start()

    train_net = make_model().to(CONFIG.TRAIN_DEVICE)

    target_net = make_model().to(CONFIG.TRAIN_DEVICE)
    target_net.load_state_dict(train_net.state_dict())
    target_net.configure(full_state=True)

    criterion = torch.nn.MSELoss()
    opt = OptWrapper(train_net.parameters(), CONFIG)

    sampling_buffer = SamplingBuffer(EnvClass, train_net.get_initial_state(1), CONFIG)

    trainer = Trainer(train_net, target_net, EnvClass, comms, CONFIG.GAMMA, sampling_buffer, opt, criterion)

    trainer.run(run_id, CONFIG)



