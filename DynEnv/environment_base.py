import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
import pygame
import pymunk.pygame_util
from gym import Space

from .cutils import CollisionType
from .cutils import ObservationType, NoiseType


@dataclass
class StateSpaceDescriptor:
    numItemsPerGridCell: int
    space: Space


@dataclass
class PredictionDescriptor:
    numContinuous: int = None  # position not included
    numBinary: int = 0  # confidence not included
    contIdx: List[int] = None
    binaryIdx: List[int] = None
    posIdx: List[int] = (0, 1)  # currently this is used everywhere, but it can be configured
    categoricIdx: int = None  # currently only one categorial variable is allowed


@dataclass
class RecoDescriptor:
    featureGridSize: Tuple[int, int]
    fullStateSpace: List[StateSpaceDescriptor]
    targetDefs: List[PredictionDescriptor]


class EnvironmentBase(object, metaclass=ABCMeta):
    def __init__(self, width, height, caption, n_players, max_players, n_time_steps, observation_type: ObservationType,
                 noise_type: NoiseType, render: bool, obs_space_cast: bool, noise_magnitude, max_time, step_iter_cnt):

        # todo: flag whether to include reconstruction

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
    def _create_observation_space(self):
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

    @abstractmethod
    def _setup_reconstruction_info(self):
        pass

    @abstractmethod
    def _handle_collisions(self):
        pass

    @abstractmethod
    def getAgentVision(self, agent):
        pass

    @abstractmethod
    def getFullState(self, agent):
        pass

    @abstractmethod
    def get_full_obs(self):
        pass

    @abstractmethod
    def get_class_specific_args(self):
        # include all arguments of the subclass constructor,
        # which are specific to that subclass
        pass

    @abstractmethod
    def _create_agents(self):
        self.agents = None

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

    def __bypass_subprocvecenv(self):
        if self.obs_space_cast:
            # IMPORTANT: the following step is needed to fool
            # the SubprocVecEnv of stable-baselines
            # only for vectorized environments
            self.observation_space.__class__ = Space

    def _render_internal(self):
        self.drawStaticObjects()
        pygame.display.flip()
        self.clock.tick(self.timeStep)

        if self.renderMode == 'memory':
            img = pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])
            self.screenShots.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _setup_vision(self, width_scale_x, width_scale_y):
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
        self._create_observation_space()
        self.__bypass_subprocvecenv()

    def _add_collision_handler(self, coll1: CollisionType, coll2: CollisionType, begin=None, post_solve=None,
                               separate=None):
        h = self.space.add_collision_handler(coll1, coll2)

        if begin is not None:
            h.begin = begin
        if post_solve is not None:
            h.post_solve = post_solve
        if separate is not None:
            h.separate = separate

    def set_random_seed(self, seed):
        # todo: set cuda backend
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

    def reset(self):
        # Agent ID and render mode must survive init
        agentID = self.agentVisID
        renderMode = self.renderMode

        self.__init__(self.nPlayers, self.renderVar, self.observationType, self.noiseType, self.noiseMagnitude,
                      self.obs_space_cast, *self.get_class_specific_args())

        self.agentVisID = agentID
        self.renderMode = renderMode

        # First observations
        observations = []
        for _ in range(self.nTimeSteps):
            if self.observationType == ObservationType.FULL:
                observations.append(self.get_full_obs())
            else:
                observations.append([self.getAgentVision(agent) for agent in self.agents])

        return observations
