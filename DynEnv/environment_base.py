from abc import ABCMeta, abstractmethod

import cv2
import pygame
import pymunk.pygame_util
from gym import Space
from gym.spaces import Tuple


from .cutils import ObservationType, NoiseType


class EnvironmentBase(object, metaclass=ABCMeta):
    def __init__(self, width, height, caption, n_players, max_players, n_time_steps, observation_type:ObservationType,
                 noise_type:NoiseType, render:bool, obs_space_cast:bool, noise_magnitude):


        self.obs_space_cast = obs_space_cast
        self.renderVar = render
        self.observationType = observation_type
        self.maxPlayers = max_players
        self.nPlayers = min(n_players, self.maxPlayers)
        self.noiseType = noise_type
        self.nTimeSteps = n_time_steps
        self.display_caption = caption
        self.W = width
        self.H = height
        self.renderMode = 'human'
        self.timeStep = 100.0
        self.noiseMagnitude = noise_magnitude

    @abstractmethod
    def drawStaticObjects(self):
        raise NotImplementedError

    @abstractmethod
    def _create_observation_space(self):
        raise NotImplementedError

    def render_internal(self):

        self.drawStaticObjects()
        pygame.display.flip()
        self.clock.tick(self.timeStep)

        if self.renderMode == 'memory':
            img = pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])
            self.screenShots.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _setup_render_options(self):
        # Render options
        if self.renderVar:
            pygame.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption(self.display_caption)
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

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
        self._create_observation_space()
        self._bypass_subprocvecenv()

    def _bypass_subprocvecenv(self):
        if self.obs_space_cast:
            # IMPORTANT: the following step is needed to fool
            # the SubprocVecEnv of stable-baselines
            # only for vectorized environments
            self.observation_space.__class__ = Space

    @abstractmethod
    def _setup_action_space(self):
        raise NotImplementedError




