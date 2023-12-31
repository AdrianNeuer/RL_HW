import argparse
import os
import random
import torch
from torch.optim import Adam
from torch.nn import MSELoss
import torch.nn.functional as F
from tester import Tester
from buffer import RolloutStorage
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from config import Config
from core.util import get_class_attr_val
from model import CnnDQN
from trainer import Trainer
import numpy as np


class CnnDDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = RolloutStorage(config)
        self.model = CnnDQN(self.config.state_shape,
                            self.config.action_dim)
        self.target_model = CnnDQN(
            self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate,
                                               eps=1e-5, weight_decay=0.95, momentum=0, centered=True)
        self.loss_fn = MSELoss()
        if self.config.use_cuda:
            self.cuda()
        self.double = self.config.double

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float)/255.0
            if self.config.use_cuda:
                state = state.to(self.config.device)
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):
        s0, s1, a, r, done = self.buffer.sample(self.config.batch_size)
        s0 = s0.float()/255.0
        s1 = s1.float()/255.0
        if self.config.use_cuda:
            s0 = s0.to(self.config.device)
            s1 = s1.to(self.config.device)
            a = a.to(self.config.device)
            r = r.to(self.config.device)
            done = done.to(self.config.device)

        # How to calculate Q(s,a) for all actions
        # q_values is a vector with size (batch_size, action_shape, 1)
        # each dimension i represents Q(s0,a_i)
        q_values = self.model(s0).cuda()

        q_eval = q_values.gather(1, a)

        with torch.no_grad():
            q_next = self.target_model(s1).cuda()
            if self.double:
                actions = self.model(s1).cuda().max(1)[1].unsqueeze(1)
                q_next = q_next.gather(1, actions)
                q_target = r + self.config.gamma * q_next * (1. - done)
            else:
                q_target = r + self.config.gamma * q_next.max(1)[0].unsqueeze(1) * (1. - done)

        # Tips: function torch.gather may be helpful
        # You need to design how to calculate the loss
        loss = self.loss_fn(q_eval, q_target)

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.config.update_tar_interval == 0:
            # params1 = list(self.model.parameters())
            # params2 = list(self.target_model.parameters())

            # # 比较参数
            # for p1, p2 in zip(params1, params2):
            #     if torch.allclose(p1.data, p2.data):
            #         print("Parameters are equal.")
            #     else:
            #         print("Parameters are not equal.")
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def cuda(self):
        self.model.to(self.config.device)
        self.target_model.to(self.config.device)

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)

    def save_model(self, output, name=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar' % (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return fr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train',
                        action='store_true', help='train model')
    parser.add_argument('--env', default='PongNoFrameskip-v4',
                        type=str, help='gym environment')
    parser.add_argument('--test', dest='test',
                        action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain',
                        action='store_true', help='retrain model')
    parser.add_argument('--model_path', type=str,
                        help='if test or retrain, import the model')
    parser.add_argument('--double', type=bool, default=False,
                        help='whether Double DQN')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument('--cuda_id', type=str, default='0',
                        help='if test or retrain, import the model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.05
    config.eps_decay = 30000
    config.frames = 2000000
    config.use_cuda = args.cuda
    config.learning_rate = 1e-6
    config.init_buff = 10000
    config.max_buff = 100000
    config.learning_interval = 4
    config.update_tar_interval = 1000
    config.batch_size = 32
    config.gif_interval = 20000
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 500000
    config.win_reward = 18
    config.win_break = True
    config.double = args.double
    config.device = torch.device("cuda:"+args.cuda_id if args.cuda else "cpu")
    # handle the atari env
    env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    print(config.action_dim, config.state_shape)
    agent = CnnDDQNAgent(config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test()

    elif args.retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)

        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        trainer.train(fr)
