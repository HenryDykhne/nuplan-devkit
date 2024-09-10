from collections import defaultdict, Counter
import math
from typing import Dict, List, Optional
from nuplan.database.utils.measure import angle_diff

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, Statistic
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.observation.idm.idm_agents_builder import get_starting_segment
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneConnector, Lane
from nuplan.common.maps.maps_datatypes import LaneConnectorType, SemanticMapLayer, TrafficLightStatusType
from nuplan.common.actor_state.agent import Agent
from nuplan.common.maps.nuplan_map.utils import cut_piece


def is_ego_in_left_turn_lane(scenario: AbstractScenario) -> bool:
    lanes = scenario.map_api.get_all_map_objects(scenario.initial_ego_state.center, SemanticMapLayer.LANE)
    if len(lanes) == 0: #there should be exactly one lane that you can be in at once. I think
        return False
    
    connectors = lanes[0].outgoing_edges
    if any([(isinstance(connector, LaneConnector) and connector.turn_type == LaneConnectorType.LEFT) for connector in connectors]): #any is false if list is empty
        return True
    else:
        return False

def does_ego_make_left_turn(history: SimulationHistory, scenario: AbstractScenario) -> bool:
    lanes_after_turning_left = [] # we grab the outgoing lanes from the lane connector that is straight
    for ego_state in history.extract_ego_state[0:(len(history.extract_ego_state) // 2)]:
        returned_lanes = scenario.map_api.get_all_map_objects(scenario.initial_ego_state.center, SemanticMapLayer.LANE)
        if len(returned_lanes) != 0 and len(returned_lanes[0].outgoing_edges) != 0 and isinstance(returned_lanes[0].outgoing_edges[0], LaneConnector):
            break
        
    if len(returned_lanes) == 0 or len(returned_lanes[0].outgoing_edges) == 0 \
        or not isinstance(returned_lanes[0].outgoing_edges[0], LaneConnector): #there should be exactly one lane that you can be in at once. I think
        return False
    
    for outgoing in returned_lanes[0].outgoing_edges:
        if outgoing.turn_type == LaneConnectorType.LEFT:
            lanes_after_turning_left.extend([lane.id for lane in outgoing.outgoing_edges])

    for ego_state in history.extract_ego_state: # we check if the ego vehicle is in one of the lanes after going turning left at some point in the simulation
        lanes = scenario.map_api.get_all_map_objects(ego_state.center, SemanticMapLayer.LANE)
        if any([lane.id in lanes_after_turning_left for lane in lanes]):
            return True
    return False

def does_lead_gap_exist(scenario: AbstractScenario, gap_threshold: float) -> bool:
    ego_state = scenario.initial_ego_state
    ego_closest_segment, ego_progress = get_starting_segment(ego_state.agent, scenario.map_api)
    if ego_closest_segment is None:
        return False
    gap = ego_closest_segment.baseline_path.length - ego_progress - ego_state.agent.box.half_length
    if gap < gap_threshold:
        return False
    else: # we want to make sure that there is no vehicle already right in front of us
        vehicles = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        for vehicle in vehicles:
            vehicle_closest_segment, vehicle_progress = get_starting_segment(vehicle, scenario.map_api)
            if vehicle_closest_segment is None:
                continue
            #if the vehicle is in the same lane and ahead of ego, but not ahead by enough, then we return false since there is not enough room to inject an gap_threshold sized agent
            if vehicle_closest_segment.id == ego_closest_segment.id \
                and vehicle_progress - ego_progress > 0 \
                and vehicle_progress - ego_progress < ego_state.agent.box.half_length + gap_threshold + vehicle.box.half_length: # space between front of ego and back of agent must have enough room for a gap_threshold length vehicle
                return False
    return True

def get_oncoming_vehicles(scenario: AbstractScenario, minimum_oncoming_vehicle_heading_threshold: float) -> List[Agent]:
    oncoming_vehicles = []
    vehicles = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
    for vehicle in vehicles:
        heading_diff = abs(angle_diff(vehicle.center.heading, scenario.initial_ego_state.center.heading, math.pi*2))
        if heading_diff > minimum_oncoming_vehicle_heading_threshold:
            oncoming_vehicles.append(vehicle)
    return oncoming_vehicles

def is_oncoming_vehicle_going_straight(history: SimulationHistory, scenario: AbstractScenario, oncoming_vehicle: Agent) -> bool:
    lanes_after_going_straight = [] # we grab the outgoing lanes from the lane connector that is straight
    returned_lanes = scenario.map_api.get_all_map_objects(oncoming_vehicle.center, SemanticMapLayer.LANE)
    if len(returned_lanes) == 0: #there should be exactly one lane that you can be in at once. I think
        #print('no returned lanes')
        return False
    for outgoing in returned_lanes[0].outgoing_edges:
        if outgoing.turn_type == LaneConnectorType.STRAIGHT:
            lanes_after_going_straight.extend([lane.id for lane in outgoing.outgoing_edges])
            
    if len(lanes_after_going_straight) == 0:
        #print('no straight lanes')
        return False
        
    for observation in history.extract_observations: # we check if the oncoming vehicle is in one of the lanes after going straight at some point in the simulation
        for vehicle in observation.tracked_objects.tracked_objects:
            if oncoming_vehicle.metadata.track_token == vehicle.metadata.track_token:
                lanes = scenario.map_api.get_all_map_objects(vehicle.center, SemanticMapLayer.LANE)
                if any([lane.id in lanes_after_going_straight for lane in lanes]):
                    return True
                break # so we dont need to loop through the rest of the vehicles once we found the right one
            
    #print('vehicle does not go straight')
    return False


def get_traffic_light_status_at_iteration(iteration: int, scenario: AbstractScenario) -> Dict[TrafficLightStatusType, List[str]]:
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


class IsScenarioLeftTurnTagOnCandidateStatistics(MetricBase):
    """
    Check if scenario is a candidate to inject a vehicle and match the left-turn-tag-on catagory of occlusion scenarios.
    """
    MINIMUM_LEAD_GAP = 5.0
    MINIMUM_ONCOMING_VEHICLE_HEADING_THRESHOLD = math.pi - 0.3 # this means about 17 degrees in either direction. so an allowable angle of roughly 34 degrees

    def __init__(
        self,
        name: str,
        category: str,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initializes the IsScenarioLeftTurnTagOnCandidate class
        :param name: Metric name
        :param category: Metric category
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        

        
    def is_scenario_left_turn_tag_on_candidate(self, history: SimulationHistory, scenario: AbstractScenario) -> List[bool]:
        checks = []
        # check that ego makes a left turn
        if not does_ego_make_left_turn(history, scenario):
            checks.append(False)
            #return False
        else:
            checks.append(True)
        
        #construct traffic light status dict
        traffic_light_status_dict = {}
        for iteration in range(0, len(history)):
            # here, we get the traffic light status at the current iteration
            traffic_light_status_dict[iteration] = get_traffic_light_status_at_iteration(iteration, scenario)
        
        # check that the left turn is currently not red for at least a bit of the first half of the simulation
        lanes = scenario.map_api.get_all_map_objects(scenario.initial_ego_state.center, SemanticMapLayer.LANE)
        non_red_left_cons = []
        if len(lanes) != 0:
            lane = scenario.map_api.get_all_map_objects(scenario.initial_ego_state.center, SemanticMapLayer.LANE)[0]
            
            for connector in lane.outgoing_edges:
                if connector.turn_type == LaneConnectorType.LEFT \
                    and any([connector.id not in traffic_light_status_dict[iteration][TrafficLightStatusType.RED] for iteration in range(len(history)//2)]):
                    non_red_left_cons.append(connector)
            if len(non_red_left_cons) == 0:
                checks.append(False)
                #return False
            else:
                checks.append(True)
        else:
            checks.append(False)
            #return False
        
        
        # check that there is enough room ahead of ego to place a vehicle
        if not does_lead_gap_exist(scenario, gap_threshold=self.MINIMUM_LEAD_GAP):
            checks.append(False)
            #return False
        else:
            checks.append(True)
        
        # check that there is an oncoming vehicle in the opposing lane
        oncoming_vehicles = get_oncoming_vehicles(scenario, self.MINIMUM_ONCOMING_VEHICLE_HEADING_THRESHOLD)
        if len(oncoming_vehicles) == 0:
            checks.append(False)
            #return False
        else:
            checks.append(True)
        
        # check that the oncoming vehicles in the opposing lane proceeds straight
        if all([(not is_oncoming_vehicle_going_straight(history, scenario, oncoming_vehicle=oncoming_vehicle)) for oncoming_vehicle in oncoming_vehicles]):
            checks.append(False)
            #return False
        else:
            checks.append(True)
        
        # check that the oncoming vehicle has a non red light in the straight lane
        all_red = True
        all_non_red_oncoming_cons = []
        for oncoming_vehicle in oncoming_vehicles:
            lanes = scenario.map_api.get_all_map_objects(oncoming_vehicle.center, SemanticMapLayer.LANE)
            if len(lanes) == 0:
                continue
            lane = lanes[0]
            oncoming_cons = []
            for connector in lane.outgoing_edges:
                if connector.turn_type == LaneConnectorType.STRAIGHT \
                and any([connector.id not in traffic_light_status_dict[iteration][TrafficLightStatusType.RED] for iteration in range(history.__len__()//2)]):
                    oncoming_cons.append(connector)
                    all_non_red_oncoming_cons.append(connector)#this is used in the next section which is why we dont break
            if len(oncoming_cons) > 0:
                all_red = False
        if all_red:
            checks.append(False)
            #print('g', scenario.token)
            #return False
        else:
            checks.append(True)
            

        all_not_intersecting = True
        for con in non_red_left_cons: #we check to see if any of the oncoming connectors intersect with the ego left turn connectors
            if any([oncoming_con.baseline_path.linestring.intersects(con.baseline_path.linestring) for oncoming_con in all_non_red_oncoming_cons]):
                all_not_intersecting = False
        if all_not_intersecting:
            checks.append(False)
            #print('i', scenario.token)
            #return False
        else:
            checks.append(True)
            
        #print('found a candidate!')
        if all(checks):
            checks.append(True)
        else:    
            checks.append(False)
        return checks
        
    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        checks = self.is_scenario_left_turn_tag_on_candidate(history=history, scenario=scenario)
        statistics = [
            Statistic(
                name='is_scenario_left_turn_tag_on_candidate',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=checks[-1],
                type=MetricStatisticsType.BOOLEAN,
            ),
            Statistic(
                name='does_ego_make_left_turn',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=checks[0],
                type=MetricStatisticsType.BOOLEAN,
            ),
            Statistic(
                name='left_turn_is_not_red',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=checks[1],
                type=MetricStatisticsType.BOOLEAN,
            ),
            Statistic(
                name='enough_room_ahead_of_ego',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=checks[2],
                type=MetricStatisticsType.BOOLEAN,
            ),
            Statistic(
                name='does_oncoming_vehicle_exist',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=checks[3],
                type=MetricStatisticsType.BOOLEAN,
            ),
            Statistic(
                name='does_oncoming_go_straight',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=checks[4],
                type=MetricStatisticsType.BOOLEAN,
            ),
            Statistic(
                name='does_oncoming_have_non_red_light',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=checks[5],
                type=MetricStatisticsType.BOOLEAN,
            )
            ,
            Statistic(
                name='does_oncoming_intersect_ego',
                unit=MetricStatisticsType.BOOLEAN.unit,
                value=checks[6],
                type=MetricStatisticsType.BOOLEAN,
            )
        ]

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=statistics, time_series=None, scenario=scenario, metric_score_unit=self.metric_score_unit
        )
        return results