from os.path import abspath
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions(precision=1)

from .logger import TemporalLogger
from .utils import AgentCheckpointer
from .storage import RolloutStorage
import time


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
        self.logger = TemporalLogger(self.params.env_name, self.timestamp, log_dir, *["ep_rewards"])
        self.checkpointer = AgentCheckpointer(self.params.env_name, self.params.num_updates, self.timestamp)

        """Environment"""
        self.env = env

        """Network"""
        self.net = net

        if self.is_cuda:
            self.net = self.net.cuda()

        """Storage"""
        self.storage = RolloutStorage(self.params.rollout_size, self.params.num_envs, self.net.num_players,
                                      len(self.net.action_descriptor), self.params.n_stack, self.net.feat_size,
                                      is_cuda=self.is_cuda, use_full_entropy=self.params.use_full_entropy)

    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states.append(obs)

        epLen = self.env.get_attr('stepNum')[0]
        updatesPerEpisode = epLen/self.params.rollout_size

        r_loss = 0
        p_loss = 0
        v_loss = 0
        be_loss = 0
        te_loss = 0
        f_loss = 0
        i_loss = 0

        t_prev = time.clock()

        for num_update in range(self.params.num_updates):
            self.net.optimizer.zero_grad()

            """A2C cycle"""
            final_value, action_probs = self.episode_rollout()

            """ICM prediction """
            # tensors for the curiosity-based loss
            # feature, feature_pred: fwd_loss
            # a_t_pred: inv_loss
            icm_losses = self.net.icm(self.storage.features, self.storage.actions, self.storage.agentFinished)
            icm_loss = sum(icm_losses)

            """Assemble loss"""
            a2c_losses, rewards = self.storage.a2c_loss(final_value, action_probs, self.params.value_coeff,
                                                        self.params.entropy_coeff)

            a2c_loss = sum(a2c_losses)

            loss = a2c_loss + icm_loss

            loss.backward(retain_graph=False)

            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.params.max_grad_norm)

            self.net.optimizer.step()

            r_loss += loss.item()
            p_loss += a2c_losses[0].item()
            v_loss += a2c_losses[1].item()
            be_loss += a2c_losses[2].item()
            te_loss += a2c_losses[3].item()
            f_loss += icm_losses[0].item()
            i_loss += icm_losses[1].item()

            dones = self.storage.dones[-1].bool()

            if dones.any():
                self.net.a2c.reset_recurrent_buffers(reset_indices=dones)

                last_r = np.array(self.storage.episode_rewards[-self.params.num_envs:]).mean()
                last_avg_r = np.array(self.storage.episode_rewards).mean()

                '''r_loss/=updatesPerEpisode
                p_loss/=updatesPerEpisode
                v_loss/=updatesPerEpisode
                be_loss/=updatesPerEpisode
                te_loss/=updatesPerEpisode
                f_loss/=updatesPerEpisode
                i_loss/=updatesPerEpisode'''

                self.logger.log(
                             **{"ep_rewards": np.array(self.storage.episode_rewards),})

                t_next = time.clock()
                print("Ep %d: %d sec (%d/%d) Loss: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f) "
                      % (int(num_update/updatesPerEpisode), (t_next - t_prev), num_update + 1, self.params.num_updates, r_loss, p_loss,
                         v_loss, be_loss, te_loss, f_loss, i_loss),
                      "Rewards: [",
                      "{0:.2f}".format(last_r), ",", "{0:.2f}".format(last_avg_r), "]")  # r_r.astype('int32'),

                t_prev = t_next
                r_loss = 0
                p_loss = 0
                v_loss = 0
                be_loss = 0
                te_loss = 0
                f_loss = 0
                i_loss = 0

                if len(self.storage.episode_rewards) >= self.storage.episode_rewards.maxlen:
                     self.checkpointer.checkpoint(loss, self.storage.episode_rewards, self.net, updatesPerEpisode)

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

            #if (num_update + 1) % 3000 == 0:
            #    torch.save(self.net.state_dict(), ("saved/net%d.pth" % (num_update + 1)))
            #    torch.save(torch.tensor(losses), "saved/losses.pth")
            #    torch.save(torch.tensor(rews), "saved/rewards.pth")
            #    torch.save(torch.tensor(rewps), "saved/rewards_pos.pth")
            #     print("current loss: ", loss.item(), " at update #", num_update)
            #     self.storage.print_reward_stats()
            # torch.save(self.net.state_dict(), "a2c_time_log_no_norm")

        self.env.close()

        self.logger.save(*["ep_rewards"])
        self.params.save(self.logger.data_dir, self.timestamp)

    def episode_rollout(self):
        episode_action_probs = []
        for step in range(self.params.rollout_size):
            """Interact with the environments """
            # call A2C
            actions, log_probs, action_probs, values, features = self.net.a2c.get_action(self.storage.get_state(step))
            # accumulate episode entropy
            episode_action_probs.append(action_probs)

            # interact
            actionsForEnv = (torch.stack(actions, dim=1)).view(
                (self.net.num_envs, self.net.num_players * 2, -1)).detach().cpu().numpy()
            new_obs, rewards, dones, state = self.env.step(actionsForEnv)

            agentFinished = torch.tensor([[agent[-1] for agent in s['Full State'][0]] for s in state]).bool()
            # save episode reward
            self.storage.log_episode_rewards(state)

            self.storage.insert(step, rewards, agentFinished, new_obs, actions, log_probs, values, dones, features)

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            _, _, _, final_value, final_features = self.net.a2c.get_action(self.storage.get_state(step + 1))

        self.storage.features[step + 1].copy_(final_features)

        return final_value, episode_action_probs
