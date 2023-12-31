from collections import defaultdict, Counter
from typing import Dict, List, Optional

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic, TimeSeries
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.idm_agents_builder import get_starting_segment
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneConnector, Lane
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.common.actor_state.agent import Agent

from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge

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
    
    def cut(self, line: LineString, distance: float) -> List[LineString]:
        """ Cuts a line at a distance from its starting point
        :param line: LineString to cut
        :param distance: distance from the start of the line to cut
        :return: tuple of LineStrings
        """
        if distance <= 0.0 :#line.length:
            return [None, LineString(line)]
        elif distance >= 1.0:
            return [LineString(line), None]
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p), normalized=True)
            if pd == distance:
                return [
                    LineString(coords[:i+1]),
                    LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance, normalized=True)
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:])]

    def cut_piece(self, line: LineString, distance1: float, distance2: float):
        """Cuts a piece out of a line
        :param line: LineString to cut
        :param distance1: percentage of line to cut from the start
        :param distance2: percentage of line to cut from the end
        :return: LineString of the middle cut piece
        """
        l1 = self.cut(line, distance1)[1]
        l2 = self.cut(line, distance2)[0]
        result = l1.intersection(l2)
        if type(result) is not MultiLineString:
            return result
        return linemerge(result)
    
    def get_non_red_connectors(self, traffic_light_status, map_api, agent: Agent) -> set:
        """ Returns the set of lane connectors that are not red that an agent is in or about to be in
        :param traffic_light_status: dictionary of traffic light status to list of lane connector ids
        :param map_api: map api to get the closest segment from
        :param agent: agent to get the non red connectors for
        :return: set of lane connectors that are not red
        """
        DISTANCE_INTO_INTERSECTION = 5 # meters
        TIME_TO_INTERSECTION = 5 # seconds
        closest_segment, progress = get_starting_segment(agent, map_api)
        # Ignore agents for which a closest_segment cannot be found
        if closest_segment is None:
            return {}

        if issubclass(type(closest_segment), LaneConnector):
            if progress > DISTANCE_INTO_INTERSECTION:
                return {} # if we have made it more than 5 meters into the intersection, other agents can likely see us now and we are either already in danger, or not in danger, but we will never be in a position to make the scenario dangerous
            connectors = [closest_segment]
        elif issubclass(type(closest_segment), Lane):
            distance_to_connector = closest_segment.baseline_path.length - progress
            #if we are moving so slowly, or are so far away that in 5 seconds the agent does not make it to the intersection, we are probably not in danger
            if  distance_to_connector > TIME_TO_INTERSECTION * agent.velocity.magnitude():
                return {}
            connectors = closest_segment.outgoing_edges
        connectors = set(connectors)
        
        to_remove = set() #we need to remove the connectors that have red traffic lights
        for connector in connectors:
            if connector.has_traffic_lights() and connector.id in traffic_light_status[TrafficLightStatusType.RED]:
                to_remove.add(connector)
        connectors = connectors - to_remove
        
        return connectors
    
    def get_traffic_light_status_at_iteration(self, iteration: int, scenario: AbstractScenario) -> Dict[TrafficLightStatusType, List[str]]:
        """Returns the traffic light status at the scene at the given iteration
        :param iteration: iteration to get the traffic light status at
        :param scenario: scenario to get the traffic light status at
        :return: dictionary of traffic light status to list of lane connector ids
        """
        traffic_light_data = scenario.get_traffic_light_status_at_iteration(iteration) #perhaps we should check once a second for the first half of the time?
        # Extract traffic light data into Dict[traffic_light_status, lane_connector_ids]
        traffic_light_status: Dict[TrafficLightStatusType, List[str]] = defaultdict(list)
    
        for data in traffic_light_data:
            traffic_light_status[data.status].append(str(data.lane_connector_id))
        return traffic_light_status
    
    def is_connector_mostly_not_red_over_scenario(self, connector: LaneConnector, traffic_light_status_dict, history: SimulationHistory, step_size: int, threshold: float = 0.1) -> bool:
        d, n = 0, 0
        for iteration in range(0, history.__len__(), int(step_size / history.interval_seconds)):
            d += 1
            if connector.has_traffic_lights() and connector.id in traffic_light_status_dict[iteration][TrafficLightStatusType.RED]:
                n += 1
            elif connector.has_traffic_lights() and connector.id in traffic_light_status_dict[iteration][TrafficLightStatusType.GREEN]:
                return True #if the light is explicitly green at any point, we can assume it is not red for the relavant portion of the scenario
            
        if n / d > threshold:
            return False
        return True
            
    def can_scenario_be_made_dangerous(self, history: SimulationHistory, scenario: AbstractScenario) -> bool:
        """Checks if the scenario can be made dangerous for the ego vehicle (if there are active intersections with vehicles in them)
        :param history: Full scenario history
        :param scenario: Full scenario object
        :return: Returns if the scenario can be made dangerous
        """
        map_api = scenario.map_api
        
        list_of_ego_states = history.extract_ego_state
        
        step_size = 1 # in seconds
        temp_connectors = []
        traffic_light_status_dict = {}
        for iteration in range(0, history.__len__(), int(step_size / history.interval_seconds)):
            # here, we get the traffic light status at the current iteration
            traffic_light_status_dict[iteration] = self.get_traffic_light_status_at_iteration(iteration, scenario)
            
            ego_state = list_of_ego_states[iteration].agent # get agent from ego state
            ego_connectors = self.get_non_red_connectors(traffic_light_status_dict[iteration], map_api, ego_state)
            if len(ego_connectors) == 1:
                temp_connectors.extend(ego_connectors)
        
        if len(temp_connectors) == 0:
            return False
        
        ego_connector = max(temp_connectors, key=Counter(temp_connectors).get) # we grab the most common connector. this is to stop us from accedentally selecting a connector that looks like where the ego might go, but is in fact only temporarily aligned

        if not self.is_connector_mostly_not_red_over_scenario(ego_connector, traffic_light_status_dict, history, step_size):
            return False
        
        UPPER_CUT = 0.95
        LOWER_CUT = 0.30
        MIN_DISTANCE = 5 # meters
        MAX_DISTANCE = 80 # meters
        ego_line = self.cut_piece(ego_connector.baseline_path.linestring, LOWER_CUT, UPPER_CUT)# cuts off first 30% and last 5% of the line
        for iteration in range(0, int(history.__len__() / 2), int(step_size / history.interval_seconds)): # we only observe the first half of the simulation
            agent_connectors = dict()
            detections = scenario.get_tracked_objects_at_iteration(iteration)
            agents = detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            ego_state = list_of_ego_states[iteration].agent #get agent from ego state at given iteration

            for agent in agents:
                ## agents should be close enough to ego to matter but far enough to be able to put someone in between: >50m, <5m
                if agent.center.distance_to(ego_state.center) > MAX_DISTANCE or agent.center.distance_to(ego_state.center) < MIN_DISTANCE:
                    agent_connectors[agent.metadata.track_token] = {}
                else:
                    agent_connectors[agent.metadata.track_token] = self.get_non_red_connectors(traffic_light_status_dict[iteration], map_api, agent)
                    
            for agent in agents:
                for connector in agent_connectors[agent.metadata.track_token]:
                    if self.is_connector_mostly_not_red_over_scenario(connector, traffic_light_status_dict, history, step_size): # we only bother checking the connector if is mostly not red
                        agent_line = self.cut_piece(connector.baseline_path.linestring, LOWER_CUT, UPPER_CUT)# cuts off first and last 5% of the line
                        if ego_line.intersects(agent_line) and ego_connector.id != connector.id:
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
