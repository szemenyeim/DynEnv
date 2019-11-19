from os.path import abspath
from time import gmtime, strftime

import torch
import torch.nn as nn

from logger import TemporalLogger
from utils import AgentCheckpointer


class Runner(object):

    def __init__(self, net, env, params, is_cuda=True, seed=42, log_dir=abspath("/data/patrik")):
        super().__init__()

        # constants
        self.timestamp = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        self.seed = seed
        self.is_cuda = torch.cuda.is_available() and is_cuda

        # parameters
        self.params = params

        """Logger"""
        self.logger = TemporalLogger(self.params.env_name, self.timestamp, log_dir, *["rewards", "features"])
        self.checkpointer = AgentCheckpointer(self.params.env_name, self.params.num_updates, self.timestamp)

        """Environment"""
        self.env = env

        """Network"""
        self.net = net

        if self.is_cuda:
            self.net = self.net.cuda()

    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        features = None
        # self.storage.states[0].copy_(self.storage.obs2tensor(obs))

        for num_update in range(self.params.num_updates):
            self.net.optimizer.zero_grad()
            """A2C cycle"""
            # get action
            actions, log_probs, entropies, values, features = self.net.a2c.get_action(obs)

            # interact
            state, new_obs, rewards, finished = self.env.step(torch.stack(actions, dim=1).detach().cpu())
            # self.net.a2c.reset_recurrent_buffers()

            a2c_loss, rewards = self.a2c_loss(values, entropies, rewards, log_probs)

            """ICM prediction """
            # tensors for the curiosity-based loss
            # feature, feature_pred: fwd_loss
            # a_t_pred: inv_loss
            icm_loss = self.net.icm(obs, new_obs, actions)

            """Assemble loss"""

            loss = a2c_loss + icm_loss

            loss.backward(retain_graph=False)

            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.params.max_grad_norm)

            """Log rewards & features"""
            # if len(self.storage.episode_rewards) > 1:
            #     self.logger.log(
            #         **{"rewards": np.array(self.storage.episode_rewards),
            #            "features": self.storage.features[-1].detach().cpu().numpy()})

            self.net.optimizer.step()

            self.net.a2c.reset_recurrent_buffers()
            obs = new_obs

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            # self.storage.after_update()

            # if len(self.storage.episode_rewards) > 1:
            #     self.checkpointer.checkpoint(loss, self.storage.episode_rewards, self.net)

            # if num_update % 1000 == 0:
            #     print("current loss: ", loss.item(), " at update #", num_update)
            #     self.storage.print_reward_stats()
            # torch.save(self.net.state_dict(), "a2c_time_log_no_norm")

        self.env.close()

        # self.logger.save(*["rewards", "features"])
        # self.params.save(self.logger.data_dir, self.timestamp)

    def a2c_loss(self, values, entropies, rewards, log_probs):
        # due to the fact that batches can be shorter (e.g. if an env is finished already)
        # MEAN is used instead of SUM
        # calculate advantage
        # i.e. how good was the estimate of the value of the current state
        # todo: use final value
        # rewards = self._discount_rewards(final_values)
        rewards = values.__class__(rewards).reshape(values.shape)
        advantage = rewards - values

        # weight the deviation of the predicted value (of the state) from the
        # actual reward (=advantage) with the negative log probability of the action taken
        policy_loss = torch.stack([(-log_prob * advantage.detach()).mean() for log_prob in log_probs]).mean()

        # the value loss weights the squared difference between the actual
        # and predicted rewards
        value_loss = advantage.pow(2).mean()

        # construct loss
        loss = policy_loss + self.params.value_coeff * value_loss - self.params.entropy_coeff * torch.stack(
            entropies).mean()

        return loss, rewards.detach().cpu().numpy()
