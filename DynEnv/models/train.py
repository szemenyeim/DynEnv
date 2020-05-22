import os.path as osp
import pickle
from time import gmtime, strftime, sleep

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions(precision=1)

from ..utils.logger import TemporalLogger
from ..utils.utils import AgentCheckpointer, transformActions, flatten
from .storage import RolloutStorage
from .loss_descriptors import A2CLosses, ICMLosses, LocalizationLosses, ReconLosses
from gym.spaces import Box

import progressbar


class Runner(object):

    def __init__(self, net, env, net_params, hparams):
        super().__init__()

        # constants
        self.timestamp = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        self.seed = hparams.seed
        self.is_cuda = torch.cuda.is_available() and hparams.cuda

        self.mse = torch.nn.MSELoss()

        # parameters
        self.params = net_params
        self.recon = hparams.use_reconstruction
        self.recon_factor = hparams.recon_factor
        self.recon_pretrained = hparams.recon_pretrained

        """Logger"""
        self.logger = TemporalLogger(self.params.env_name, self.timestamp, hparams.log_dir,
                                     *["ep_rewards", "ep_pos_rewards", "ep_obs_rewards", "ep_goals"])
        self.checkpointer = AgentCheckpointer(self.params.env_name, self.params.num_updates, self.timestamp,
                                              hparams.log_dir)

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
                                      self.numActions, self.params.n_stack, self.net.feat_size * 2,
                                      is_cuda=self.is_cuda, use_full_entropy=self.params.use_full_entropy)

    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states.append(obs[..., :-1])
        self.storage.seens.append(obs[..., -1])

        '''initLoc = torch.tensor(flatten(self.env.env_method('get_agent_locs'))).squeeze()
        if self.is_cuda:
            initLoc = initLoc.cuda()
        self.net.a2c.embedder_base.initialize(initLoc)'''

        """Variables for proper logging"""
        epLen = self.env.get_attr('stepNum')[0]
        updatesPerEpisode = int(epLen // self.params.rollout_size)
        bar = progressbar.ProgressBar(0, updatesPerEpisode, redirect_stdout=False)

        """Losses"""
        r_loss = 0
        a2c_loss_accumulated = A2CLosses()
        icm_loss_accumulated = ICMLosses()
        loc_losses_accumulated = LocalizationLosses()
        recon_loss_accumulated = ReconLosses(num_thresh=3, num_classes=2)

        if self.is_cuda:
            loc_losses_accumulated.cuda()
            recon_loss_accumulated.cuda()

        self.corrs = []
        self.avgPrecs = []

        num_rollout = 0

        for num_update in range(self.params.num_updates):
            bar.update(num_rollout)
            self.net.optimizer.zero_grad()

            """A2C cycle"""
            final_value, action_probs = self.episode_rollout()

            """ICM prediction """
            # tensors for the curiosity-based loss
            # feature, feature_pred: fwd_loss
            # a_t_pred: inv_loss
            icm_losses = self.net.icm(self.storage.features, self.storage.actions, self.storage.agentFinished)

            loc_losses = self.net.a2c.embedder_base.compute_loc_loss(self.storage.positions, self.storage.pos_target) \
                if self.recon else LocalizationLosses()

            recon_loss = self.compute_recon_losses(self.storage.features[:-2, :, self.net.feat_size:],
                                                   self.storage.full_state_targets, self.storage.seens) \
                if self.recon else ReconLosses(num_thresh=3, num_classes=2)

            if not self.recon:
                if self.is_cuda:
                    loc_losses.cuda()
                    recon_loss.cuda()
                loc_losses.prepare_losses()
                recon_loss.prepare_losses()

            """Assemble loss"""
            a2c_losses, rewards = self.storage.a2c_loss(final_value, action_probs, self.params.value_coeff,
                                                        self.params.entropy_coeff)

            loss = a2c_losses.loss + icm_losses.loss + self.recon_factor * (recon_loss.loss + loc_losses.loss)

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
            loc_losses.detach_loss()
            loc_losses_accumulated += loc_losses
            num_rollout += 1
            loc_losses.finalize_corr()
            recon_loss.compute_APs()

            """Print to console at the end of each episode"""
            dones = self.storage.dones[-1].bool()
            if dones.any():
                self.net.a2c.reset_recurrent_buffers(reset_indices=dones)

                bar.finish()

                """Get average rewards and positive rewards (for this episode and the last 10)"""
                last_r = np.array(self.storage.episode_rewards[-self.params.num_envs:]).mean()
                last_avg_r = np.array(self.storage.episode_rewards).mean()

                last_p_r = np.array(self.storage.episode_pos_rewards[-self.params.num_envs:]).mean()
                last_avg_p_r = np.array(self.storage.episode_pos_rewards).mean()

                last_o_r = np.array(self.storage.episode_obs_rewards[-self.params.num_envs:]).mean()
                last_avg_o_r = np.array(self.storage.episode_obs_rewards).mean()

                """Get goals"""
                goals = np.array(self.storage.goals).T * self.params.num_envs

                recon_loss_accumulated.div(num_rollout)
                loc_losses_accumulated.div(num_rollout)
                loc_losses_accumulated.finalize_corr()
                recon_loss_accumulated.compute_APs()

                self.logger.log(
                    **{"ep_rewards": np.array(self.storage.episode_rewards[-self.params.num_envs:]),
                       "ep_pos_rewards": np.array(self.storage.episode_pos_rewards[-self.params.num_envs:]),
                       "ep_obs_rewards": np.array(self.storage.episode_obs_rewards[-self.params.num_envs:])})

                self.avgPrecs.append(
                    (recon_loss_accumulated.recall.mean(dim=0) + recon_loss_accumulated.precision.mean(dim=0)) / 2)
                self.corrs.append(loc_losses_accumulated.corr)

                print("Ep %d: (%d/%d) L: (Loss: %.2f, P: %.2f, V: %.2f, E: %.2f, TE: %.2f, F: %.2f, I: %.2f)"
                      % (int(num_update / updatesPerEpisode), num_update + 1, self.params.num_updates, r_loss,
                         a2c_loss_accumulated.policy, a2c_loss_accumulated.value, a2c_loss_accumulated.entropy,
                         a2c_loss_accumulated.temp_entropy,
                         icm_loss_accumulated.forward, icm_loss_accumulated.inverse),
                      "R: [",
                      "{0:.2f}".format(last_r), "/", "{0:.2f}".format(last_avg_r), ",",
                      "{0:.2f}".format(last_p_r), "/", "{0:.2f}".format(last_avg_p_r), ",",
                      "{0:.2f}".format(last_o_r), "/", "{0:.2f}".format(last_avg_o_r), "]",
                      "[", int(goals.mean(axis=1)[0]), ":", int(goals.mean(axis=1)[1]), "]", flush=True)

                if self.recon:
                    print(recon_loss_accumulated, flush=True)
                    print(loc_losses_accumulated, flush=True)

                r_loss = 0

                a2c_loss_accumulated = A2CLosses()  # reset
                icm_loss_accumulated = ICMLosses()

                recon_loss_accumulated = ReconLosses(num_thresh=3, num_classes=2)  # reset
                loc_losses_accumulated = LocalizationLosses()
                if self.is_cuda:
                    loc_losses_accumulated.cuda()
                    recon_loss_accumulated.cuda()

                num_rollout = 0
                sleep(0.5)
                bar = progressbar.ProgressBar(0, updatesPerEpisode, redirect_stdout=False)

                '''initLoc = torch.tensor(flatten(self.env.env_method('get_agent_locs'))).squeeze()
                if self.is_cuda:
                    initLoc = initLoc.cuda()
                self.net.a2c.embedder_base.initialize(initLoc)'''

                """Best model is saved according to reward in the driving env, but positive rewards are used for robocup"""
                rewards_that_count = self.storage.episode_rewards if self.params.env_name == 'Driving' else self.storage.episode_pos_rewards
                if len(rewards_that_count) >= rewards_that_count.maxlen:
                    self.checkpointer.checkpoint(loss, rewards_that_count, self.net, updatesPerEpisode)

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

        baseName = osp.join(self.logger.data_dir, "Recon" if self.recon else "NoRecon")
        file = open(baseName + "Pret.pickle" if self.recon else ".pickle", "wb")
        pickle.dump([self.corrs, self.avgPrecs], file)

        self.env.close()

        self.logger.save(*["ep_rewards", "ep_pos_rewards", "ep_obs_rewards", "ep_goals"])
        self.params.save(self.logger.data_dir, self.timestamp)

    def episode_rollout(self):
        episode_action_probs = []

        if self.recon:
            initLoc = torch.tensor(flatten(self.env.env_method('get_agent_locs'))).squeeze()
            initLoc += (torch.randn(initLoc.shape)) / (20)
            if self.is_cuda:
                initLoc = initLoc.cuda()
            self.net.a2c.embedder_base.initialize(initLoc)

        for step in range(self.params.rollout_size):
            """Interact with the environments """
            # call A2C
            actions, log_probs, action_probs, values, features, pos = self.net.a2c.get_action(
                (self.storage.get_state(step), transformActions(self.storage.actions[step - 1].t(), True)))
            # accumulate episode entropy
            episode_action_probs.append(action_probs)

            truePos = torch.tensor(flatten(self.env.env_method('get_agent_locs'))).squeeze()
            if self.is_cuda:
                truePos = truePos.cuda()

            # interact
            actionsForEnv = (torch.stack(actions, dim=1)).view(
                (self.net.num_envs, self.net.num_players, -1)).detach().cpu().numpy()
            new_obs, rewards, dones, state = self.env.step(actionsForEnv)
            # self.net.a2c.embedder_base.initialize(truePos)

            # Finished states (ICM loss ignores predictions from crashed/finished cars or penalized robots)
            fullStates = [s['Recon States'] for s in state]
            fullStates = [[s[0::2] for s in state] for state in fullStates]
            agentFinished = torch.tensor([[agent[-1] for agent in s['Full State'][0]] for s in state]).bool()

            # save episode reward
            self.storage.log_episode_rewards(state)

            self.storage.insert(step, rewards, agentFinished, new_obs[..., :-1], actions, log_probs, values, dones,
                                features,
                                fullStates, pos, truePos, new_obs[..., -1])

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            states = self.net.a2c.embedder_base.get_states()
            _, _, _, final_value, final_features, _ = self.net.a2c.get_action(
                (self.storage.get_state(step + 1), transformActions(self.storage.actions[step].t())))
            self.net.a2c.embedder_base.set_states(states)

        self.storage.features[step + 1].copy_(final_features.detach())
        # self.storage.positions.append(final_pos)

        return final_value, episode_action_probs

    def compute_recon_losses(self, features, targets, seens):

        objSeens = np.array(seens)
        sizes = objSeens.shape
        objSeens = objSeens.transpose(0, 1, 3, 2).reshape(sizes[0], -1, sizes[2])
        robSeens = torch.tensor(
            [[[[s for s in step[1]] for step in rob] for rob in time] for time in objSeens]).bool().any(
            dim=2)
        ballSeens = torch.tensor(
            [[[step[2] for step in rob] for rob in time] for time in objSeens]).bool().any(
            dim=2)
        robSeenBefore = [robSeens[:i + 1].any(dim=0) for i in range(robSeens.shape[0])]
        ballSeenBefore = [ballSeens[:i + 1].any(dim=0) for i in range(ballSeens.shape[0])]

        recLosses = ReconLosses(num_classes=2, num_thresh=3)
        recLosses.cuda()

        for j, (obj_features) in enumerate(features):
            recLosses += self.net.a2c.embedder_base.reconstructor(obj_features, targets[j],
                                                                  (ballSeenBefore[j], robSeenBefore[j]))

        recLosses.div(features.shape[0])

        return recLosses
