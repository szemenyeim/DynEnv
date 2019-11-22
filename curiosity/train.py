from os.path import abspath
from time import gmtime, strftime

import torch
import torch.nn as nn
import numpy as np
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
        self.logger = TemporalLogger(self.params.env_name, self.timestamp, log_dir, *["rewards", "features"])
        self.checkpointer = AgentCheckpointer(self.params.env_name, self.params.num_updates, self.timestamp)

        """Environment"""
        self.env = env


        """Network"""
        self.net = net

        if self.is_cuda:
            self.net = self.net.cuda()

        """Storage"""
        self.storage = RolloutStorage(self.params.rollout_size, self.params.num_envs, self.net.num_players,
                                      len(self.net.action_descriptor), self.params.n_stack, 2*self.net.feat_size, is_cuda=self.is_cuda)


    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states.append(obs)
        features = None

        r_loss = 0
        r_r = 0
        losses = []
        rews = []

        t_prev = time.clock()

        # self.storage.states[0].copy_(self.storage.obs2tensor(obs))

        for num_update in range(self.params.num_updates):
            self.net.optimizer.zero_grad()

            """A2C cycle"""
            final_value, entropy = self.episode_rollout()

            """ICM prediction """
            # tensors for the curiosity-based loss
            # feature, feature_pred: fwd_loss
            # a_t_pred: inv_loss
            # if self.net.icm.prev_features is not None:
            icm_loss = self.net.icm(self.storage.features, self.storage.actions)
            # self.net.icm.prev_features = features


            """Assemble loss"""
            a2c_loss, rewards = self.storage.a2c_loss(final_value, entropy, self.params.value_coeff,
                                                      self.params.entropy_coeff)

            loss = a2c_loss + icm_loss

            loss.backward(retain_graph=True)

            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.params.max_grad_norm)

            """Log rewards & features"""
            # if len(self.storage.episode_rewards) > 1:
            #     self.logger.log(
            #         **{"rewards": np.array(self.storage.episode_rewards),
            #            "features": self.storage.features[-1].detach().cpu().numpy()})

            self.net.optimizer.step()

            r_loss += loss.item()
            #l = [max(rew[0],0.0) for rew in rewards]
            l = [rew[0] for rew in rewards]
            r_r += np.array(l)

            self.net.a2c.reset_recurrent_buffers()

            #todo: ezt át kellene írni
            if finished.any():
                r_loss /= (600*self.net.num_envs*self.net.num_players*2)
                losses.append(r_loss)
                rews.append(r_r)
                t_next = time.clock()
                print("Episode %d finished: Time: %d Iters: %d/%d Loss: %.2f " % (len(losses)-1, (t_next-t_prev), num_update+1, self.params.num_updates, r_loss))
                print("Rewards: ",  int(r_r.mean())) #r_r.astype('int32'),
                t_prev = t_next
                r_loss = 0
                r_r = 0
                self.net.icm.prev_features = None
                obs = self.env.reset()
            else:
                obs = new_obs

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            # self.storage.after_update()

            # if len(self.storage.episode_rewards) > 1:
            #     self.checkpointer.checkpoint(loss, self.storage.episode_rewards, self.net)

            if num_update % 25000 == 0 and num_update:
                torch.save(self.net.state_dict(), ("/saved/net%d.pth" % num_update))
            #     print("current loss: ", loss.item(), " at update #", num_update)
            #     self.storage.print_reward_stats()
            # torch.save(self.net.state_dict(), "a2c_time_log_no_norm")

        self.env.close()

        # self.logger.save(*["rewards", "features"])
        # self.params.save(self.logger.data_dir, self.timestamp)


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

            # save episode reward
            # self.storage.log_episode_rewards(infos)

            self.storage.insert(step, rewards, new_obs, actions, log_probs, values, dones, features)
            self.net.a2c.reset_recurrent_buffers(reset_indices=dones)

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            _, _, _, final_value, final_features = self.net.a2c.get_action(self.storage.get_state(step + 1))

        self.storage.features[step + 1].copy_(final_features)

        return final_value, episode_entropy

