# import d4rl

import gym
import numpy as np
import torch
import tensorflow as tf
from algorithm_offline.model.td3bc import TD3 as TD3BC
from algorithm_offline.utils.memory import ReplayBuffer
from algorithm_offline.utils.params import get_args
from algorithm_offline.utils.utils import evaluation
from algorithm_offline.agent.cql import CQL
from algorithm_offline.utils.utils import get_output_folder, TensorBoardLogger

task_name = "{}-{}-v0"
env_names = ['hopper']  # ['halfcheetah', 'hopper', 'walker2d']
levels = ['random', 'medium', 'expert']

def save_data(task_name, env_name, level):
    path = './dataset_mujoco/{}_{}_data.npy'.format(env_name, level)
    env = gym.make(task_name.format(env_name, level))
    dataset = env.get_dataset()

    states = dataset["observations"][:]
    actions = dataset["actions"][:]
    next_states = np.concatenate(
        [dataset["observations"][1:], np.zeros_like(states[0])[np.newaxis, :]], axis=0)
    rewards = dataset["rewards"][:, np.newaxis]
    terminals = dataset["terminals"][:, np.newaxis] + 0.

    state_dict = {'state': states,
                  'action': actions,
                  'next_state': next_states,
                  'reward': rewards,
                  'terminal': terminals}

    np.save(path, state_dict)


def load_data(path):
    data = np.load(path, allow_pickle=True).item()
    states = data['state']
    actions = data['action']
    next_states = data['next_state']
    rewards = data['reward']
    terminals = data['terminal']

    dataset = {'state': states,
               'action': actions,
               'next_state': next_states,
               'reward': rewards,
               'terminal': terminals}

    return dataset


args = get_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.state_dim = 11
args.action_dim = 3
save_interval = 1000
eval_interval = 100

for env_name in env_names:
    for level in levels:
        dataset = load_data('./dataset_mujoco/{}_{}_data.npy'.format(env_name, level))
        states, actions, next_states, rewards, terminals = dataset['state'], dataset['action'], dataset['next_state'], dataset['reward'], dataset['terminal']
        outputdir = get_output_folder('../out', env_name + '_' + level)
        board_logger = TensorBoardLogger(outputdir)
        replay_buffer = ReplayBuffer(args)
        replay_buffer.set_buffer(states, actions, next_states, rewards, terminals)
        policy = CQL(args)

        for i in range(20000):
            print(i)
            q_loss, cql_loss, total_loss, tp_loss = policy.train(replay_buffer)
            if (i > 0) and (i % eval_interval == 0):
                board_logger.scalar_summary("Performance", i, evaluation(policy, "Hopper-v3"))
        policy.save_model(outputdir+"/model.pt".format(i))
