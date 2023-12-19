from collections import defaultdict, Counter
from typing import Dict, List, Optional, Type
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import Observation

import numpy as np

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.idm_agents_builder import get_starting_segment
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, LaneConnector, Lane
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType


from shapely.geometry import Point, LineString, Polygon
from collections import deque 
import time



class CanScenarioBeMadeDangerousStatistics(MetricBase):
    """
    Check if ego trajectory intersects with an agent that is occluded from the ego before they reach the point of intersection.
    """

    def __init__(
        self,
        name: str,
        category: str,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initializes the CanScenarioBeMadeDangerousStatistics class
        :param name: Metric name
        :param category: Metric category
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        
    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(metric_statistics[0].value)
    
    def get_non_red_connectors(self, traffic_light_status, map_api, agent) -> set:
        """_summary_

        :param traffic_light_status: _description_
        :param map_api: _description_
        :param agent: _description_
        :return: set of lane connectors that are not red
        """
        closest_segment, progress = get_starting_segment(agent, map_api)
        # Ignore agents for which a closest_segment cannot be found
        if closest_segment is None:
            return {}

        if issubclass(type(closest_segment), LaneConnector):
            if progress > 5:
                return {} # if we have made it more than 5 meters into the intersection, other agents can likely see us now and we are either already in danger, or not in danger, but we will never be in a position to make the scenario dangerous
            connectors = [closest_segment]
        elif issubclass(type(closest_segment), Lane):
            if closest_segment.baseline_path.length - progress > 10: #if we are more than 10 meters from the end of the lane, we are too far away from the intersection to be in danger
                return {}
            connectors = closest_segment.outgoing_edges
        connectors = set(connectors)
        
        to_remove = set() #we need to remove the connectors that have red traffic lights
        for connector in connectors:
            if connector.has_traffic_lights() and connector.id in traffic_light_status[TrafficLightStatusType.RED]:
                to_remove.add(connector)
        connectors = connectors - to_remove
        
        return connectors
    
    def get_traffic_light_status_at_iteration(self, iteration: int, scenario: AbstractScenario):
        """_summary_

        :param iteration: _description_
        :param scenario: _description_
        :return: _description_
        """
        traffic_light_data = scenario.get_traffic_light_status_at_iteration(iteration) #perhaps we should check once a second for the first half of the time?
        # Extract traffic light data into Dict[traffic_light_status, lane_connector_ids]
        traffic_light_status: Dict[TrafficLightStatusType, List[str]] = defaultdict(list)
    
        for data in traffic_light_data:
            traffic_light_status[data.status].append(str(data.lane_connector_id))
        return traffic_light_status

    def can_scenario_be_made_dangerous(self, history: SimulationHistory, scenario: AbstractScenario) -> bool:
        """_summary_

        :param history: _description_
        :param scenario: _description_
        :return: _description_
        """
        map_api = scenario.map_api
        
        
        list_of_ego_states = history.extract_ego_state
        
        
        step_size = 1 #in seconds
        temp_connectors = []
        traffic_light_status_list = {}
        print(history.__len__(), int(step_size / history.interval_seconds), history.interval_seconds)
        for iteration in range(0, history.__len__(), int(step_size / history.interval_seconds)):
            #here, we get the traffic light status at the current iteration
            traffic_light_status_list[iteration] = self.get_traffic_light_status_at_iteration(iteration, scenario)
            
            ego_state = list_of_ego_states[iteration].agent #get agent from ego state
            ego_connectors = self.get_non_red_connectors(traffic_light_status_list[iteration], map_api, ego_state)
            if len(ego_connectors) == 1:
                temp_connectors.extend(ego_connectors)
        
        if len(temp_connectors) == 0:
            return False
        
        ego_connector = max(temp_connectors, key=Counter(temp_connectors).get) # we grab the most common connector. this is to stop us from accedentally selecting a connector that looks like where the ego might go, but is in fact only temporarily aligned
    
        for iteration in range(0, history.__len__(), int(step_size / history.interval_seconds)):
            agent_connectors = dict()
            detections = scenario.get_tracked_objects_at_iteration(iteration)
            agents = detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            ego_state = list_of_ego_states[iteration].agent #get agent from ego state at given iteration

            for agent in agents:
                ## agents should be close enough to ego to matter but far enough to be able to put someone in between: >50m, <10m
                if agent.center.distance_to(ego_state.center) > 50 or agent.center.distance_to(ego_state.center) < 10:
                    agent_connectors[agent.metadata.track_token] = {}
                else:
                    agent_connectors[agent.metadata.track_token] = self.get_non_red_connectors(traffic_light_status_list[iteration], map_api, agent)
                    
            for agent in agents:
                for connector in agent_connectors[agent.metadata.track_token]:
                    if ego_connector.baseline_path.linestring.intersects(connector.baseline_path.linestring) and ego_connector.id != connector.id:
                        return True
        
        return False




    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        can_scenario_be_made_dangerous = self.can_scenario_be_made_dangerous(history=history, scenario=scenario)
        statistics = [
            Statistic(
                name='can_scenario_be_made_dangerous',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=can_scenario_be_made_dangerous,
                type=MetricStatisticsType.BOOLEAN,
            )
        ]

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit
        )
        return results
