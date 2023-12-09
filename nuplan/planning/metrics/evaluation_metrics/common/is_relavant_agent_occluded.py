from typing import List, Optional

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from shapely.geometry import Point, LineString, Polygon
from collections import deque 
import time



class IsRelavantAgentOccludedStatistics(MetricBase):
    """
    Check if ego trajectory intersects with an agent that is occluded from the ego before they reach the point of intersection.
    """

    def __init__(
        self,
        name: str,
        category: str,
        occlusion_window_before_intersection: float = 5,
        intersection_delta: float = 3,
        buf_distance: float = 1,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initializes the IsRelavantAgentOccludedStatistics class
        :param name: Metric name
        :param category: Metric category
        :param occlusion_window_before_intersection: Metric allowed time window for occlusions to happen before the intersection in seconds
        :param intersection_delta: Metric allowed time delta between target vehicle intersecting ego
        :param buf_distance: Metric buffer used to check for collisions
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        assert occlusion_window_before_intersection > intersection_delta
        self.occlusion_window_before_intersection = occlusion_window_before_intersection
        self.intersection_delta = intersection_delta
        self.buf_distance = buf_distance
        
    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)


    def is_relavant_agent_occluded(self, history: SimulationHistory, scenario: AbstractScenario) -> bool:
        """
        Check if relavant agent occluded becomes occluded from ego before either ego or they reach the intersection point within a time limit
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return Relavant agent occluded status.
        """
        start = time.time()

        list_of_occlusion_masks = history.occlusion_masks

        if list_of_occlusion_masks is None:
            return False
        
        list_of_ego_states = history.extract_ego_state
        list_of_observations = history.extract_observations

        #we only grab the initial track tokens but we could also grab all of them, or from the window that starts at the current iteration till the end, or to some preselected future time horizon
        initial_vehicle_track_tokens = {
            tracked_object.track_token
            for tracked_object in scenario.initial_tracked_objects.tracked_objects
            if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE
        }
    
        
        vehicles_future_tracks = {track_token: [] for track_token in initial_vehicle_track_tokens}
        ego_traj = deque()
        for i in range(scenario.get_number_of_iterations() - 1):
            for tracked_object in list_of_observations[i].tracked_objects:
                if tracked_object.track_token in initial_vehicle_track_tokens:
                    vehicles_future_tracks[tracked_object.track_token].append((tracked_object.center.x, tracked_object.center.y))
            e_center = list_of_ego_states[i].center
            ego_traj.append((e_center.x, e_center.y))
            
        vehicles_linestrings_buffered = {}
        #we convert the tracks to a linestring with a buffer in order to be able to check for intersections later with the ego trajectory
        for token in vehicles_future_tracks.keys():
            if len(vehicles_future_tracks[token]) > 1:
                vehicles_linestrings_buffered[token] = LineString(vehicles_future_tracks[token]).buffer(self.buf_distance)
        
        ego_track_buffered = LineString(ego_traj).buffer(self.buf_distance)
        
        #now we check if ego traj intersects with each agent traj to determine which is relavant
        relavant_agent_tokens = []
        for token in vehicles_linestrings_buffered.keys():
            if ego_track_buffered.intersects(vehicles_linestrings_buffered[token]):
                relavant_agent_tokens.append(token)
        if len(relavant_agent_tokens) == 0:
            return False
                
        # now that we have the relavant agents, we can figure out exactly when they intersect and if they are occluded in the window before then
        occlusion_window_timesteps = int(self.occlusion_window_before_intersection / history.interval_seconds)
        intersection_delta_timesteps = int(self.intersection_delta / history.interval_seconds)
        print('occlusion_window_timesteps', occlusion_window_timesteps)
        print('intersection_delta_timesteps', intersection_delta_timesteps)
        tracked_objects_by_time = []
        print('numrel agents', len(relavant_agent_tokens))
        print('scenario', scenario.scenario_name, scenario.token)
        for i in range(scenario.get_number_of_iterations() - 1):
            tracked_objects_by_time.append([obj 
                                            for obj in list_of_observations[i].tracked_objects
                                            if obj.track_token in relavant_agent_tokens])
        for i in range(scenario.get_number_of_iterations() - 1):
            occluded = {token: False for token in relavant_agent_tokens}
            for j in range(max(i - occlusion_window_timesteps, 0), min(i + intersection_delta_timesteps + 1, scenario.get_number_of_iterations() - 1)):
                for tracked_object in tracked_objects_by_time[j]:
                    time_us_at_j = list_of_ego_states[j].time_us
                    #cross can happen any time in the window but occlusion must happen before the time that ego reaches the crossing point
                    if i > j and tracked_object.track_token not in list_of_occlusion_masks[time_us_at_j]: #if its not in the occlusion mask, then its not visible
                        occluded[tracked_object.track_token] = True
                    distance = ((ego_traj[i][0] - tracked_object.center.x)**2 + (ego_traj[i][1] - tracked_object.center.y)**2)**0.5
                    if abs(i - j) <= intersection_delta_timesteps and distance < self.buf_distance * 2 and occluded[tracked_object.track_token]:
                        print('time elapsed true', time.time() - start)
                        return True
        print('time elapsed false', time.time() - start) 
        return False

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        is_relavant_agent_occluded = self.is_relavant_agent_occluded(history=history, scenario=scenario)
        statistics = [
            Statistic(
                name='is_relavant_agent_occluded',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=is_relavant_agent_occluded,
                type=MetricStatisticsType.BOOLEAN,
            )
        ]

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit
        )
        return results
