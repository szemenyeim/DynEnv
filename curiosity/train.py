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
        self.logger = TemporalLogger(self.params.env_name, self.timestamp, log_dir, *["ep_rewards", "rewards"])
        self.checkpointer = AgentCheckpointer(self.params.env_name, self.params.num_updates, self.timestamp)

        """Environment"""
        self.env = env

        """Network"""
        self.net = net

        if self.is_cuda:
            self.net = self.net.cuda()

        """Storage"""
        self.storage = RolloutStorage(self.params.rollout_size, self.params.num_envs, self.net.num_players,
                                      len(self.net.action_descriptor), self.params.n_stack, 2 * self.net.feat_size,
                                      is_cuda=self.is_cuda)

    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states.append(obs)

        entropyLossScale = 1.0/(100*self.params.entropy_coeff)

        epLen = self.env.get_attr('stepNum')[0]
        updatesPerEpisode = epLen/self.params.rollout_size

        r_loss = 0
        p_loss = 0
        v_loss = 0
        e_loss = 0
        f_loss = 0
        i_loss = 0

        r_r = 0
        rews = []
        mean_rews = []

        t_prev = time.clock()

        for num_update in range(self.params.num_updates):
            self.net.optimizer.zero_grad()

            """A2C cycle"""
            final_value, entropy = self.episode_rollout()

            """ICM prediction """
            # tensors for the curiosity-based loss
            # feature, feature_pred: fwd_loss
            # a_t_pred: inv_loss
            icm_losses = self.net.icm(self.storage.features, self.storage.actions, self.storage.carFinished)
            icm_loss = sum(icm_losses)

            """Assemble loss"""
            a2c_losses, rewards = self.storage.a2c_loss(final_value, entropy, self.params.value_coeff,
                                                        self.params.entropy_coeff)

            a2c_loss = sum(a2c_losses)

            loss = a2c_loss + icm_loss

            loss.backward(retain_graph=False)

            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.params.max_grad_norm)

            self.net.optimizer.step()

            rewards = rewards.view((-1, 1))

            r_loss += loss.item()
            p_loss += a2c_losses[0].item()
            v_loss += a2c_losses[1].item()
            e_loss += a2c_losses[2].item()
            f_loss += icm_losses[0].item()
            i_loss += icm_losses[1].item()

            l = [rew.item() for rew in rewards]
            r_r += np.array(l)

            dones = self.storage.dones[-1].bool()

            # logger test
            self.logger.log(**{"rewards": r_r})

            if dones.any():
                self.net.a2c.reset_recurrent_buffers(reset_indices=dones)

                rews.append(r_r)
                mean_rews.append(r_r.mean() * self.params.rollout_size*50)

                last_r = np.array(self.storage.episode_rewards[-self.params.num_envs:]).mean()
                last_avg_r = np.array(self.storage.episode_rewards).mean()

                self.logger.log(
                             **{"ep_rewards": np.array(self.storage.episode_rewards),})

                avg_r = sum(mean_rews) / len(mean_rews) if len(mean_rews) < 10 else sum(mean_rews[-10:]) / 10.0

                t_next = time.clock()
                print("Ep %d: %d sec (%d/%d) Loss: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f) "
                      % (len(rews), (t_next - t_prev), num_update + 1, self.params.num_updates, r_loss, p_loss,
                         v_loss, e_loss*entropyLossScale, f_loss, i_loss),
                      "Rewards: [", int(mean_rews[-1]), ",", int(avg_r), "] [",
                      "{0:.2f}".format(last_r), ",", "{0:.2f}".format(last_avg_r), "]")  # r_r.astype('int32'),
                t_prev = t_next
                r_loss = 0
                p_loss = 0
                v_loss = 0
                e_loss = 0
                f_loss = 0
                i_loss = 0
                r_r = 0

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

        self.logger.save(*["ep_rewards", "rewards"])
        self.params.save(self.logger.data_dir, self.timestamp)

    def episode_rollout(self):
        episode_entropy = 0
        for step in range(self.params.rollout_size):
            """Interact with the environments """
            # call A2C
            actions, log_probs, entropies, values, features = self.net.a2c.get_action(self.storage.get_state(step))
            # accumulate episode entropy
            episode_entropy += torch.stack(entropies, dim=-1)

            # interact
            actionsForEnv = (torch.stack(actions, dim=1)).view(
                (self.net.num_envs, self.net.num_players * 2, -1)).detach().cpu().numpy()
            new_obs, rewards, dones, state = self.env.step(actionsForEnv)

            carFinished = torch.tensor([[car[-1] for car in s['Full State'][0]] for s in state]).bool()
            # save episode reward
            self.storage.log_episode_rewards(state)

            self.storage.insert(step, rewards, carFinished, new_obs, actions, log_probs, values, dones, features)

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            _, _, _, final_value, final_features = self.net.a2c.get_action(self.storage.get_state(step + 1))

        self.storage.features[step + 1].copy_(final_features)

        return final_value, episode_entropy
