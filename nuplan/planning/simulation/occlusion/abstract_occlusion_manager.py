from abc import ABCMeta, abstractmethod
from collections import deque
import time
from typing import Deque
import copy

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObjects

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation


import itertools ## change
class AbstractOcclusionManager(metaclass=ABCMeta):
    """
    Interface for a generic occlusion manager.
    """

    def __init__(
        self,
        scenario: AbstractScenario,
        # these must be the first and second optional arguments to the constructor
        uncloak_reaction_time: float = 1.5, #seconds. I probably want this set to 1.5 for other agents and 0.0 - 0.5 for ego 
        notice_threshold: float = 1.0 # seconds 
    ):
        assert uncloak_reaction_time >= 0.0, f"Uncloak reaction time ({uncloak_reaction_time}) must be non-negative."
        assert notice_threshold >= 0.0, f"Notice threshold ({notice_threshold}) must be non-negative."
        assert notice_threshold <= uncloak_reaction_time, f"Notice threshold ({notice_threshold}) must be less than or equal to uncloak reaction time ({uncloak_reaction_time})."
        self._visible_agent_cache = {}
        self._noticed_agent_cache = {} 
        self._historical_noticed_agent_cache = {} # this exists because the display needs the noticed values at the time they were initially computed and they can change with subsequent occlusions
        self.scenario = scenario
        self.uncloak_reaction_time = uncloak_reaction_time 
        self.notice_threshold = notice_threshold 

    def reset(self) -> None:
        """
        Resets occlusion manager cache.
        """
        self._visible_agent_cache = {}
        self._noticed_agent_cache = {} 
        self._historical_noticed_agent_cache = {} # this exists because the display needs the noticed values at the time they were initially computed and they can change with subsequent occlusions

    def occlude_input(self, input_buffer: SimulationHistoryBuffer) -> SimulationHistoryBuffer:
        """
        Occludes SimulationHistoryBuffer input. Loops through each timestep defined by time_us,
        checks to see if timestep is already contained in _visible_agent_cache and computes
        occlusions if not, and occludes timestep using cached results. Repacks output in 
        SimulationHistoryBuffer.
        """

        ego_state_buffer = input_buffer.ego_state_buffer
        observations_buffer = input_buffer.observation_buffer
        sample_interval = input_buffer.sample_interval

        for ego_state, observations in zip(ego_state_buffer, observations_buffer):
            if ego_state.time_us not in self._visible_agent_cache:
                self._visible_agent_cache[ego_state.time_us] = self._compute_visible_agents(ego_state, observations)

        current_time_seconds = ego_state_buffer[-1].time_seconds
        assert len(ego_state_buffer) * input_buffer.sample_interval >= self.uncloak_reaction_time, "SimulationHistoryBuffer must be at least as long as uncloak reaction time."
        for i, (ego_state, observations) in enumerate(zip(ego_state_buffer, observations_buffer)):#we loop through to find the first timestep inside the uncloak_reaction_time
            if ego_state.time_us not in self._noticed_agent_cache: #we only enter here at the begining of the simulation to determine the noticed cache of the history
                for j, ego_state_c in enumerate(ego_state_buffer): #this for loop only exists to find the right index
                    if ego_state.time_seconds - ego_state_c.time_seconds <= self.uncloak_reaction_time: #this will eventually be true
                        self._compute_noticed_agents(input_buffer.sample_interval, 
                                                    deque(itertools.islice(ego_state_buffer, j, i + 1)), 
                                                    deque(itertools.islice(observations_buffer, j, i + 1))) #this is only run once per state not in the noticed_agents_cache
                        break   
                self._historical_noticed_agent_cache[ego_state.time_us] = copy.deepcopy(self._noticed_agent_cache[ego_state.time_us])
            elif current_time_seconds - ego_state.time_seconds <= self.uncloak_reaction_time: #this for loop only exists to find the right index
                self._compute_noticed_agents(input_buffer.sample_interval, 
                                            deque(itertools.islice(ego_state_buffer, i, None)), 
                                            deque(itertools.islice(observations_buffer, i, None))) #this only gets run once since it breaks out of the loop immedietly afterwards
                break

        output_buffer = SimulationHistoryBuffer(ego_state_buffer, \
                            deque([self._mask_input(ego_state.time_us, observations) for ego_state, observations in zip(ego_state_buffer, observations_buffer)]), \
                                sample_interval)
        
        current_time_us = ego_state_buffer[-1].time_us
        self._historical_noticed_agent_cache[current_time_us] = copy.deepcopy(self._noticed_agent_cache[current_time_us])
        return output_buffer
    
    @abstractmethod
    def _compute_visible_agents(self, ego_state: EgoState, observations: DetectionsTracks) -> set:
        """
        Returns set of track tokens that represents the observations visible to the ego at this time step.
        """
        pass

    ######################################################################################################################### changes
    def _compute_noticed_agents(self, sample_interval: float, ego_state_buffer: Deque[EgoState], observations_buffer: Deque[Observation]) -> None:
        """
        Fills out the latest uncloak window for the _noticed_agent_cache. agents that are noticed at a timestep will always be noticed at that timestep
        """
        self._noticed_agent_cache[ego_state_buffer[-1].time_us] = set()
        notice_threshold_over_sample_interval = int(self.notice_threshold / sample_interval)
        time_us_at_begining_of_window = ego_state_buffer[0].time_us
        
        initial_tokens = self._visible_agent_cache[time_us_at_begining_of_window].union(self._noticed_agent_cache[time_us_at_begining_of_window])

        for agent in observations_buffer[0].tracked_objects.tracked_objects:
            token = agent.metadata.track_token
            if token in initial_tokens:
                vis_count = sum(token in self._visible_agent_cache[ego_state.time_us] for ego_state in ego_state_buffer)
                if vis_count >= notice_threshold_over_sample_interval:
                    for ego_state in ego_state_buffer:
                        if ego_state.time_us not in self._noticed_agent_cache:
                            self._noticed_agent_cache[ego_state.time_us] = set()
                        self._noticed_agent_cache[ego_state.time_us].add(token)
    ####################################################################################################################################   

    def _mask_input(self, time_us: int, observations: DetectionsTracks) -> DetectionsTracks:
        """
        Occludes observations at timestep time_us based on cached occlusions.
        """

        assert time_us in self._visible_agent_cache, "Attempted to mask non-cached timestep!"
        assert isinstance(observations, DetectionsTracks), "Occlusions only support DetectionsTracks."

        visible_mask = self._visible_agent_cache[time_us]
        noticed_mask = self._noticed_agent_cache[time_us]## changes
        tracks = observations.tracked_objects.tracked_objects

        visible_tracks = [track for track in tracks if ((track.metadata.track_token in visible_mask) and (track.metadata.track_token in noticed_mask))] ## changes basically, to be a visible track, you need to both be visible and noticed

        return DetectionsTracks(tracked_objects=TrackedObjects(visible_tracks))