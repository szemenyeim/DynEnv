import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import deque

import cv2
import numpy as np
import pygame
import pymunk.pygame_util
from gym import Space

from .cutils import ObservationType, NoiseType


class EnvironmentBase(object, metaclass=ABCMeta):
    def __init__(self, width, height, caption, n_players, max_players, n_time_steps, observation_type: ObservationType,
                 noise_type: NoiseType, render: bool, obs_space_cast: bool, noise_magnitude, max_time, step_iter_cnt):

        self.maxTime = max_time
        self.elapsed = 0
        self.stepIterCnt = step_iter_cnt
        self.timeStep = 100.0
        self.nTimeSteps = n_time_steps

        self.obs_space_cast = obs_space_cast
        self.observationType = observation_type

        self.maxPlayers = max_players
        self.nPlayers = min(n_players, self.maxPlayers)

        self.W = width
        self.H = height
        self.caption = caption
        self.renderVar = render

        self.noiseType = noise_type
        self.noiseMagnitude = noise_magnitude

        self.__setup_simulator()
        self.__setup_visualization()

        self.__setup_render_options()

    @abstractmethod
    def drawStaticObjects(self):
        pass

    @abstractmethod
    def create_observation_space(self):
        pass

    @abstractmethod
    def _init_rewards(self):
        pass

    @abstractmethod
    def _setup_normalization(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def _setup_action_space(self):
        pass

    def __setup_simulator(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

    def __setup_visualization(self):
        self.agentVisID = None
        self.renderMode = 'human'
        self.screenShots = deque(maxlen=self.stepIterCnt)
        self.obsVis = deque(maxlen=self.stepIterCnt // 10)

    def __setup_render_options(self):
        # Render options
        if self.renderVar:
            pygame.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption(self.caption)
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def _render_internal(self):

        self.drawStaticObjects()
        pygame.display.flip()
        self.clock.tick(self.timeStep)

        if self.renderMode == 'memory':
            img = pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])
            self.screenShots.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _setup_vision(self, width_scale_x=0.4, width_scale_y=0.8):

        if self.noiseMagnitude < 0 or self.noiseMagnitude > 5:
            print("Error: The noise magnitude must be between 0 and 5!")
            exit(0)
        if self.observationType == ObservationType.FULL and self.noiseMagnitude > 0:
            print(
                "Warning: Full observation type does not support noisy observations, "
                "but your noise magnitude is set to a non-zero value! "
                "(The noise setting has no effect in this case)")
        self.randBase = 0.01 * self.noiseMagnitude
        self.noiseMagnitude = self.noiseMagnitude
        self.maxVisDist = [(self.W * width_scale_x) ** 2, (self.W * width_scale_y) ** 2]

    def _setup_observation_space(self):
        self.observation_space = None
        self.create_observation_space()
        self._bypass_subprocvecenv()

    def _bypass_subprocvecenv(self):
        if self.obs_space_cast:
            # IMPORTANT: the following step is needed to fool
            # the SubprocVecEnv of stable-baselines
            # only for vectorized environments
            self.observation_space.__class__ = Space

    def set_random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def close(self):
        pass

    def render(self):
        if self.renderMode == 'human':
            warnings.warn("env.render(): This function does nothing in human render mode.\n"
                          "If you want to render into memory, set the renderMode variable to 'memory'!")
            return
        return self.screenShots, self.obsVis
