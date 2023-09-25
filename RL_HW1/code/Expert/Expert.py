import gym
import cv2
from abc import ABC
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


# Expert Model, load weights from ./params.pth

def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class PolicyModel(nn.Module, ABC):

    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                               stride=1)  # Nature paper -> kernel_size = 3, OpenAI repo -> kernel_size = 4

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=448)

        self.extra_value_fc = nn.Linear(in_features=448, out_features=448)
        self.extra_policy_fc = nn.Linear(in_features=448, out_features=448)

        self.policy = nn.Linear(in_features=448, out_features=self.n_actions)
        self.int_value = nn.Linear(in_features=448, out_features=1)
        self.ext_value = nn.Linear(in_features=448, out_features=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.fc1.bias.data.zero_()
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        self.fc2.bias.data.zero_()

        nn.init.orthogonal_(self.extra_policy_fc.weight, gain=np.sqrt(0.1))
        self.extra_policy_fc.bias.data.zero_()
        nn.init.orthogonal_(self.extra_value_fc.weight, gain=np.sqrt(0.1))
        self.extra_value_fc.bias.data.zero_()

        nn.init.orthogonal_(self.policy.weight, gain=np.sqrt(0.01))
        self.policy.bias.data.zero_()
        nn.init.orthogonal_(self.int_value.weight, gain=np.sqrt(0.01))
        self.int_value.bias.data.zero_()
        nn.init.orthogonal_(self.ext_value.weight, gain=np.sqrt(0.01))
        self.ext_value.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_v = x + F.relu(self.extra_value_fc(x))
        x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)
        policy = self.policy(x_pi)
        probs = F.softmax(policy, dim=1)
        dist = Categorical(probs)

        return dist, int_value, ext_value, probs


def mean_of_list(func):
    def function_wrapper(*args, **kwargs):
        lists = func(*args, **kwargs)
        return [sum(list) / len(list) for list in lists[:-4]] + [explained_variance(lists[-4], lists[-3])] + \
               [explained_variance(lists[-2], lists[-1])]

    return function_wrapper


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    return img


def stack_states(stacked_frames, state, is_new_episode):
    frame = preprocessing(state)

    if is_new_episode:
        stacked_frames = np.stack([frame for _ in range(4)], axis=0)
    else:
        stacked_frames = stacked_frames[1:, ...]
        stacked_frames = np.concatenate(
            [stacked_frames, np.expand_dims(frame, axis=0)], axis=0)
    return stacked_frames


# Calculates if value function is a good predictor of the returns (ev > 1)
# or if it's just worse than predicting nothing (ev =< 0)
def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def make_atari(env_id, max_episode_steps, sticky_action=True, max_and_skip=True):
    env = gym.make(env_id, render_mode='human')
    # print(env._max_episode_steps)
    env._max_episode_steps = max_episode_steps * 4

    assert 'NoFrameskip' in env.spec.id
    if sticky_action:
        env = StickyActionEnv(env)
    if max_and_skip:
        env = RepeatActionEnv(env)
    # env = MontezumaVisitedRoomEnv(env, 3)
    # env = AddRandomStateToInfoEnv(env)

    return env


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()


class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = np.zeros(
            (2,) + self.env.observation_space.shape, dtype=np.uint8)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, _, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info
