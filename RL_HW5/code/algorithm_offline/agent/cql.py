import copy
import torch
import torch.nn.functional as F
import numpy as np

from algorithm_offline.network.actor import Stochastic_Actor
from algorithm_offline.network.critic import Twin_Qnetwork


class CQL:

    def __init__(self, args):
        self.args = args
        self.seed = args.seed

        self.state_dim = self.args.state_dim
        self.action_dim = self.args.action_dim
        self.hidden_dim = self.args.hidden_dim
        self.action_clip = self.args.action_clip
        self.grad_norm_clip = self.args.grad_norm_clip

        self.gamma = args.gamma
        self.tau = args.tau
        self.lr = args.lr
        self.update_interval = args.update_interval
        self.target_entropy = - self.action_dim
        self.batch_size = args.batch_size_mf
        self.device = args.device

        if self.device == 'gpu':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.log_alpha_prime = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha_prime = self.log_alpha_prime.exp()

        self.actor_eval = Stochastic_Actor(
            self.state_dim, self.action_dim, self.hidden_dim, self.action_clip).to(self.device)
        self.critic_eval = Twin_Qnetwork(
            self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        # self.actor_target = copy.deepcopy(self.actor_eval).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_eval).to(self.device)

        self.actor_optim = torch.optim.Adam(
            self.actor_eval.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(
            self.critic_eval.parameters(), lr=self.lr)
        self.temp_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.temp_prime_optim = torch.optim.Adam(
            [self.log_alpha_prime], lr=self.lr)

        self.total_it = 0
        self.cql_samples = 10
        self.action_gap = 10

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def inference(self, state, deterministic=False):
        input_state = torch.tensor(
            state, dtype=torch.float32)shape.unsqueeze(0).to(self.device)
        output_action, _, output_mean = self.actor_eval(
            input_state, get_mean=deterministic)
        action = output_action.data.cpu().numpy()[0]
        if deterministic:
            mean = output_mean.data.cpu().numpy()[0]
            return mean
        return action

    def _get_tensor_values(self, state, actions):
        num_repeat = 10
        batch_size = state.shape[0]
        state_temp = state.unsqueeze(1).repeat(1, num_repeat, 1).view(
            batch_size*num_repeat, state.shape[-1])
        action_temp = actions.view(batch_size*num_repeat, actions.shape[-1])
        values = self.critic_eval(state_temp, action_temp)
        values = (value.view(state.shape[0], num_repeat) for value in values)
        return values

    def _get_policy_actions(self, state, repeat):
        state_temp = state.unsqueeze(1).repeat(1, repeat, 1).view(
            state.shape[0] * repeat, state.shape[1])
        new_state_actions, new_state_log_pi, _ = self.actor_eval(
            state_temp, get_logprob=True, get_mean=True
        )
        new_state_actions = new_state_actions.view(state.shape[0], repeat, -1)
        new_state_log_pi = new_state_log_pi.view(state.shape[0], repeat, -1)
        return new_state_actions, new_state_log_pi

    def soft_update(self):
        for param, target_param in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        # for param, target_param in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
        #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_critic(self, state, action, next_state, reward, done):
        with torch.no_grad():
            next_action, next_logprobs, _ = self.actor_eval(
                next_state, get_logprob=True)
            q_t1, q_t2 = self.critic_target(next_state, next_action)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward + \
                (1.0 - done) * self.gamma * \
                (q_target - self.alpha * next_logprobs)
        q_1, q_2 = self.critic_eval(state, action)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)

        # Add CQL, reference https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py
        random_actions = action.new_empty(
            (action.shape[0], self.cql_samples, self.action_dim), requires_grad=False).uniform_(-1, 1)

        policy_actions, log_probs = self._get_policy_actions(
            state, repeat=self.cql_samples)

        next_policy_actions, next_log_probs = self._get_policy_actions(
            next_state, repeat=self.cql_samples)

        q1_random, q2_random = self._get_tensor_values(state, random_actions)
        q1_policy, q2_policy = self._get_tensor_values(state, policy_actions)
        q1_next_policy, q2_next_policy = self._get_tensor_values(
            state, next_policy_actions)

        random_density = np.log(0.5 ** self.action_dim)
        policy_density = log_probs.detach().squeeze(-1)
        next_policy_density = next_log_probs.detach().squeeze(-1)

        concat_q1 = torch.cat([q1_random-random_density, q1_next_policy-next_policy_density,
                              q1_policy - policy_density], dim=1)
        concat_q2 = torch.cat([q2_random-random_density, q2_next_policy-next_policy_density,
                              q2_policy - policy_density ], dim=1)

        logsumexp_q1 = torch.logsumexp(concat_q1, dim=1).unsqueeze(1).mean()
        logsumexp_q2 = torch.logsumexp(concat_q2, dim=1).unsqueeze(1).mean()

        qf1_loss = logsumexp_q1 - q_1.mean()
        qf2_loss = logsumexp_q2 - q_2.mean()

        qf1_loss = self.alpha_prime * (qf1_loss - self.action_gap)
        qf2_loss = self.alpha_prime * (qf2_loss - self.action_gap)

        tp_loss = (-qf1_loss - qf2_loss) * 0.5
        self.temp_prime_optim.zero_grad()
        tp_loss.backward(retain_graph=True)
        self.temp_prime_optim.step()
        self.alpha_prime = self.log_alpha_prime.exp()

        q_loss = loss_1 + loss_2
        qf_loss = qf1_loss + qf2_loss
        
        q_loss_step = q_loss + qf_loss
        self.critic_optim.zero_grad()
        q_loss_step.backward()
        self.critic_optim.step()

        qf_loss = qf1_loss + qf2_loss

        return q_loss.item(), qf_loss.item(), q_loss_step.item(), tp_loss.item()

    def update_actor(self, state):
        action, logprobs, _ = self.actor_eval(state, get_logprob=True)
        q_b1, q_b2 = self.critic_eval(state, action)
        qval_batch = torch.min(q_b1, q_b2)
        actor_loss = (self.alpha * logprobs - qval_batch).mean()
        temp_loss = -self.log_alpha * \
            (logprobs.detach() + self.target_entropy).mean()

        for p in self.critic_eval.parameters():
            p.requires_grad = False
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        for p in self.critic_eval.parameters():
            p.requires_grad = True

        # self.actor_optim.zero_grad()
        # actor_loss.backward()
        # #torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), self.grad_norm_clip)
        # self.actor_optim.step()
        #
        # self.temp_optim.zero_grad()
        # temp_loss.backward()
        # self.temp_optim.step()

        self.alpha = self.log_alpha.exp()

        return actor_loss.item(), temp_loss.item()

    def train(self, memory):
        self.total_it += 1
        state, action, next_state, reward, done = memory.sample(
            self.batch_size)

        q_loss, qf_loss, total_loss, tp_loss = None, None, None, None
        q_loss, qf_loss, total_loss, tp_loss = self.update_critic(
            state, action, next_state, reward, done)
        aloss, tloss = self.update_actor(state)
        if self.total_it % self.update_interval == 0:
            self.soft_update()

        return q_loss, qf_loss, total_loss, tp_loss

    def save_model(self, path):
        state_dict = {'actor_eval': self.actor_eval.state_dict(),
                      'logalpha': self.log_alpha,
                      'logalphaprime': self.log_alpha_prime,
                      # 'actor_target': self.actor_target.state_dict(),
                      'critic_eval': self.critic_eval.state_dict(),
                      'critic_target': self.critic_target.state_dict(),
                      'actor_optim': self.actor_optim.state_dict(),
                      'critic_optim': self.critic_optim.state_dict(), }

        torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.actor_eval.load_state_dict(state_dict['actor_eval'])
        self.log_alpha = state_dict['logalpha']
        self.log_alpha_prime = state_dict['logalphaprime']
        # self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic_eval.load_state_dict(state_dict['critic_eval'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optim.load_state_dict(state_dict['actor_optim'])
        self.critic_optim.load_state_dict(state_dict['critic_optim'])
