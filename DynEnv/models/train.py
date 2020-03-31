from os.path import abspath
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions(precision=1)

from ..utils.logger import TemporalLogger
from ..utils.utils import AgentCheckpointer
from .storage import RolloutStorage
from .models import ReconLosses, A2CLosses, ICMLosses
from gym.spaces import Box


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
        self.logger = TemporalLogger(self.params.env_name, self.timestamp, log_dir,
                                     *["ep_rewards", "ep_pos_rewards", "ep_goals"])
        self.checkpointer = AgentCheckpointer(self.params.env_name, self.params.num_updates, self.timestamp, log_dir)

        """Environment"""
        self.env = env

        """Network"""
        self.net = net

        if self.is_cuda:
            self.net = self.net.cuda()

        # Get number of actions
        self.numActions = sum(
            [sum(act.shape) if type(act) == Box else sum(act.nvec.shape) for act in self.net.action_descriptor])

        """Storage"""
        self.storage = RolloutStorage(self.params.rollout_size, self.params.num_envs, self.net.num_players,
                                      self.numActions, self.params.n_stack, self.net.feat_size,
                                      is_cuda=self.is_cuda, use_full_entropy=self.params.use_full_entropy)

    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states.append(obs)

        """Variables for proper logging"""
        epLen = self.env.get_attr('stepNum')[0]
        updatesPerEpisode = epLen / self.params.rollout_size

        """Losses"""

        r_loss = 0

        a2c_loss_accumulated = A2CLosses()

        icm_loss_accumulated = ICMLosses()

        """Recon Losses"""

        recon_loss_accumulated = ReconLosses()

        num_rollout = 0

        for num_update in range(self.params.num_updates):
            self.net.optimizer.zero_grad()

            """A2C cycle"""
            final_value, action_probs = self.episode_rollout()

            """ICM prediction """
            # tensors for the curiosity-based loss
            # feature, feature_pred: fwd_loss
            # a_t_pred: inv_loss
            icm_losses, recon_loss = self.net.icm(self.storage.features, self.storage.actions,
                                                  self.storage.agentFinished,
                                                  self.storage.full_state_targets)

            """Assemble loss"""
            a2c_losses, rewards = self.storage.a2c_loss(final_value, action_probs, self.params.value_coeff,
                                                        self.params.entropy_coeff)

            loss = a2c_losses.loss + icm_losses.loss + recon_loss.loss

            loss.backward(retain_graph=False)

            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.params.max_grad_norm)

            self.net.optimizer.step()

            """Running losses"""
            r_loss += loss.item()

            a2c_losses.detach_loss()
            a2c_loss_accumulated += a2c_losses

            icm_losses.detach_loss()
            icm_loss_accumulated += icm_losses

            """Running recon losses"""
            recon_loss.detach_loss()
            recon_loss_accumulated += recon_loss
            num_rollout += 1

            """Print to console at the end of each episode"""
            dones = self.storage.dones[-1].bool()
            if dones.any():
                self.net.a2c.reset_recurrent_buffers(reset_indices=dones)

                """Get average rewards and positive rewards (for this episode and the last 10)"""
                last_r = np.array(self.storage.episode_rewards[-self.params.num_envs:]).mean()
                last_avg_r = np.array(self.storage.episode_rewards).mean()

                last_p_r = np.array(self.storage.episode_pos_rewards[-self.params.num_envs:]).mean()
                last_avg_p_r = np.array(self.storage.episode_pos_rewards).mean()

                """Get goals"""
                goals = np.array(self.storage.goals).T * self.params.num_envs

                self.logger.log(
                    **{"ep_rewards": np.array(self.storage.episode_rewards[-self.params.num_envs:]),
                       "ep_pos_rewards": np.array(self.storage.episode_pos_rewards[-self.params.num_envs:]),
                       "ep_goals": goals})

                print("Ep %d: (%d/%d) L: (%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f)"
                      % (int(num_update / updatesPerEpisode), num_update + 1, self.params.num_updates, r_loss,
                         a2c_loss_accumulated.policy, a2c_loss_accumulated.value, a2c_loss_accumulated.entropy,
                         a2c_loss_accumulated.temp_entropy,
                         icm_loss_accumulated.forward, icm_loss_accumulated.inverse),
                      "R: [",
                      "{0:.2f}".format(last_r), "/", "{0:.2f}".format(last_avg_r), ",",
                      "{0:.2f}".format(last_p_r), "/", "{0:.2f}".format(last_avg_p_r), "]",
                      "[", int(goals.mean(axis=1)[0]), ":", int(goals.mean(axis=1)[1]), "]")

                recon_loss_accumulated.precision /= num_rollout
                recon_loss_accumulated.recall /= num_rollout

                print(recon_loss_accumulated)

                r_loss = 0

                a2c_loss_accumulated = A2CLosses()  # reset
                icm_loss_accumulated = ICMLosses()

                recon_loss_accumulated = ReconLosses()  # reset
                num_rollout = 0

                """Best model is saved according to reward in the driving env, but positive rewards are used for robocup"""
                rewards_that_count = self.storage.episode_rewards if self.params.env_name == 'Driving' else self.storage.episode_pos_rewards
                if len(rewards_that_count) >= rewards_that_count.maxlen:
                    self.checkpointer.checkpoint(loss, rewards_that_count, self.net, updatesPerEpisode)

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

        self.env.close()

        self.logger.save(*["ep_rewards", "ep_pos_rewards", "ep_goals"])
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
                (self.net.num_envs, self.net.num_players, -1)).detach().cpu().numpy()
            new_obs, rewards, dones, state = self.env.step(actionsForEnv)

            # Finished states (ICM loss ignores predictions from crashed/finished cars or penalized robots)
            fullStates = [s['Recon States'] for s in state]
            agentFinished = torch.tensor([[agent[-1] for agent in s['Full State'][0]] for s in state]).bool()

            # save episode reward
            self.storage.log_episode_rewards(state)

            self.storage.insert(step, rewards, agentFinished, new_obs, actions, log_probs, values, dones, features,
                                fullStates)

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            _, _, _, final_value, final_features = self.net.a2c.get_action(self.storage.get_state(step + 1))

        self.storage.features[step + 1].copy_(final_features)

        return final_value, episode_action_probs
