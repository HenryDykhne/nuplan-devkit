from abc import ABCMeta, abstractmethod
from collections import deque
from typing import Tuple

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObjects

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation


class AbstractOcclusionManager(metaclass=ABCMeta):
    def __init__(
        self,
        scenario: AbstractScenario
    ):
        self._masks = {}
        self.scenario = scenario

    def reset(self) -> None:
        self._masks = {}

    def occlude_input(self, input_buffer: SimulationHistoryBuffer) -> SimulationHistoryBuffer:
        ego_state_buffer = input_buffer.ego_state_buffer
        observations_buffer = input_buffer.observation_buffer
        sample_interval = input_buffer.sample_interval

        for ego_state, observations in zip(ego_state_buffer, observations_buffer):
            if ego_state.time_us not in self._masks:
                self._masks[ego_state.time_us] = self._compute_mask(ego_state, observations)
                
        output_buffer = SimulationHistoryBuffer(ego_state_buffer, \
                            deque([self._mask_input(ego_state.time_us, observations) for ego_state, observations in zip(ego_state_buffer, observations_buffer)]), \
                                sample_interval)

        return output_buffer
    
    @abstractmethod
    def _compute_mask(self, ego_state: EgoState, observations: DetectionsTracks) -> set:
        pass

    def _mask_input(self, time_us: int, observations: DetectionsTracks) -> DetectionsTracks:
        assert time_us in self._masks, "Attempted to mask non-cached timestep!"
        assert isinstance(observations, DetectionsTracks), "Occlusions only support DetectionsTracks."

        mask = self._masks[time_us]
        tracks = observations.tracked_objects.tracked_objects

        visible_tracks = [track for track in tracks if track.metadata.track_token in mask]

        return DetectionsTracks(tracked_objects=TrackedObjects(visible_tracks))


