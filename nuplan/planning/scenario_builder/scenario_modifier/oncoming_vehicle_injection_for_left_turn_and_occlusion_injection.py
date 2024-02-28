from __future__ import annotations
from collections import defaultdict
import copy
import math
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType

from nuplan.common.maps.abstract_map_objects import LaneConnector, LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import LaneConnectorType, SemanticMapLayer
from nuplan.database.utils.measure import angle_diff
from shapely.geometry import Polygon, Point, MultiPolygon, MultiLineString, LineString, MultiPoint
from shapely import union_all

from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractModification, AbstractScenarioModifier
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.scenario_builder.scenario_modifier.occlusion_injection_modifier import OcclusionInjectionModifier
from nuplan.planning.simulation.occlusion.wedge_occlusion_manager import WedgeOcclusionManager

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation

class OncomingInjectionForLeftTurnAndOcclusionInjectionModifier(OcclusionInjectionModifier):
    STRAIGHT_RADIUS = 50.0
    def __init__(self):
        super().__init__() #maybe we will need this later
        
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        scenario = runner.scenario
        #TODO: for now, this is a simple way to make sure we are only modifying left turn scenarios but in practice, we should do a real check since some scenarios could be left turns without being labeled as such
        if scenario.scenario_type != "starting_left_turn":
            print(f'Warning: scenario {scenario.token} is of type {scenario.scenario_type} and not "starting_left_turn" so we are not modifying it.')
            return []
        
        historical_egostate_generator = scenario.get_expert_ego_trajectory()
        traffic_light_status = self.get_traffic_light_status_at_iteration(0, scenario)
        initial_ego_agent_object = scenario.get_ego_state_at_iteration(0)
        initial_ego_agent_state = initial_ego_agent_object.agent
        
        map_api = scenario.map_api
        _, ego_lane_level_route_plan = runner.simulation._observations._get_roadblock_path(
                    initial_ego_agent_state,
                    scenario.get_expert_goal_state()
                )
        
        if ego_lane_level_route_plan is None:
            print(f'Warning: ego in scenario {scenario.token} has no route plan')
            return []
        
        left_turn = None
        distance_to_left_turn = 0.0
        for i, map_object in enumerate(ego_lane_level_route_plan):
            
            if isinstance(map_object, LaneConnector) and map_object.turn_type == LaneConnectorType.LEFT:
                left_turn = map_object
                break
            if i == 0:
                progress_along_map_object = map_object.baseline_path.get_nearest_arc_length_from_position(initial_ego_agent_state.center)
                distance_to_left_turn += map_object.baseline_path.length - progress_along_map_object
            else:
                distance_to_left_turn += map_object.baseline_path.length
            
            if i > 3:
                print(f'Warning: ego in scenario {scenario.token} has its left turn too far away from the start of the route plan.')
                return []
        
        if left_turn is None:
            print(f'Warning: ego in scenario {scenario.token} has no left turn in route plan')
            return []
        
        first = left_turn.baseline_path.linestring.coords[0]
        first = Point2D(*first)
        ego_heading_before_turn = left_turn.baseline_path.get_nearest_pose_from_position(first).heading
        connectors = map_api.get_proximal_map_objects(first, self.STRAIGHT_RADIUS, [SemanticMapLayer.LANE_CONNECTOR])[SemanticMapLayer.LANE_CONNECTOR]
        straight_connectors = [connector for connector in connectors if connector.turn_type == LaneConnectorType.STRAIGHT]
        intersecting_straight_connectors = [connector for connector in straight_connectors if connector.baseline_path.linestring.intersects(left_turn.baseline_path.linestring)]
        oncoming_intersecting_straight_lanes = []
        oncoming_connectors = []
        for connector in intersecting_straight_connectors:
            first = connector.baseline_path.linestring.coords[0]
            first = Point2D(*first)
            potential_oncoming_connector_heading = connector.baseline_path.get_nearest_pose_from_position(first).heading
            if abs(angle_diff(ego_heading_before_turn, potential_oncoming_connector_heading, 2 * np.pi)) > np.pi - (np.pi/4):
                oncoming_intersecting_straight_lanes.extend(connector.incoming_edges)
                oncoming_connectors.append(connector)
                
        if len(oncoming_intersecting_straight_lanes) == 0:
            print(f'Warning: ego in scenario {scenario.token} has no oncoming straight lanes intersecting its left turn')
            return []
        elif len(oncoming_intersecting_straight_lanes) > 1:
            print(f'Notice: ego in scenario {scenario.token} has {len(oncoming_intersecting_straight_lanes)} oncoming straight lane intersecting its left turn')
        
        #TODO: for now, we are just taking the first one but we should do a real check to make sure we are taking the correct one, or maybe use them all
        oncoming_lane = oncoming_intersecting_straight_lanes[0]
        oncoming_connector = oncoming_connectors[0]
        if oncoming_connector.id in traffic_light_status[TrafficLightStatusType.RED]:
            print(f'Oncoming lane {oncoming_lane.id} in scenario {scenario.token} has a red traffic light')
            return []
        if left_turn.id in traffic_light_status[TrafficLightStatusType.RED]:
            print(f'Left turn {left_turn.id} in scenario {scenario.token} has a red traffic light')
            return []
        
        conflict_point = left_turn.baseline_path.linestring.intersection(oncoming_connector.baseline_path.linestring)
        if isinstance(conflict_point, MultiPoint):
            conflict_point = conflict_point.geoms[0] 
        conflict_point = Point2D(conflict_point.x, conflict_point.y)
        
        distance_to_conflict_from_begining_of_left_turn = left_turn.baseline_path.get_nearest_arc_length_from_position(conflict_point)    
        ego_distance_to_conflict_point = distance_to_left_turn + distance_to_conflict_from_begining_of_left_turn
        distance_to_conflict_from_begining_of_straight = oncoming_connector.baseline_path.get_nearest_arc_length_from_position(conflict_point)
        distance_from_end_of_oncoming_lane = ego_distance_to_conflict_point - distance_to_conflict_from_begining_of_straight
        distance_from_begining_of_oncoming_lane = oncoming_lane.baseline_path.length - distance_from_end_of_oncoming_lane
        potential_oncoming_vehicle_spawn_point = oncoming_lane.baseline_path.linestring.interpolate(distance_from_begining_of_oncoming_lane)
        potential_oncoming_vehicle_spawn_point = Point2D(potential_oncoming_vehicle_spawn_point.x, potential_oncoming_vehicle_spawn_point.y)
        potential_oncoming_vehicle_spawn_pose = oncoming_lane.baseline_path.get_nearest_pose_from_position(potential_oncoming_vehicle_spawn_point)
        
        goal_segment = oncoming_connector.outgoing_edges[0] #finding a goal for our oncoming vehicle. It will automatically be extended if it is too close.
        *_, last = goal_segment.baseline_path.linestring.coords #we use the last point in a linestring far away to start setting our goal
        last = Point(last)
        goal_point = Point2D(last.x, last.y)
        potential_oncoming_vehicle_goal = goal_segment.baseline_path.get_nearest_pose_from_position(goal_point)
        
        potential_oncoming_vehicle_speed = initial_ego_agent_state.velocity.magnitude() #we will use the same speed as the ego, so hopefully the oncoming vehicle will arive at the conflict point at the same time
        potential_oncoming_vehicle_heading = potential_oncoming_vehicle_spawn_pose.heading
        #potential_oncoming_vehicle_heading = math.atan2(potential_oncoming_vehicle_spawn_pose.heading.point.y, potential_oncoming_vehicle_spawn_pose.heading.point.x)
        potential_oncoming_vehicle_velocity = StateVector2D(potential_oncoming_vehicle_speed * math.cos(potential_oncoming_vehicle_heading), potential_oncoming_vehicle_speed * math.sin(potential_oncoming_vehicle_heading))
        
        oncoming_vehicle_token = "oncoming_inserted"
        oncoming_agent_to_insert = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=OrientedBox(potential_oncoming_vehicle_spawn_pose, 5, 2, 2),
            velocity=potential_oncoming_vehicle_velocity,
            metadata=SceneObjectMetadata(scenario.get_time_point(0).time_us, oncoming_vehicle_token, -2, oncoming_vehicle_token),
            angular_velocity=0.0,
        )    
        
        
        objects_to_avoid = list(AGENT_TYPES | STATIC_OBJECT_TYPES) #grab anything we could crash into
        avoid = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(objects_to_avoid)
        avoid_geoms = [initial_ego_agent_state.box.geometry]#to ensure ego gets added to the geometry since it is not a tracked object
        inject_poly = Polygon(oncoming_agent_to_insert.box.all_corners())
        for obj in avoid:
            avoid_geoms.append(obj.box.geometry)
        avoid_geoms = MultiPolygon(avoid_geoms)
        
        if inject_poly.intersects(avoid_geoms.buffer(self.MINIMUM_SPAWNING_DISTANCE)): #if injected agent intersects with other agents
            print(f'Warning: attempted insertion of oncoming vehicle in scenario {scenario.token} intersects with other agents')
            return []
        
        #now that we know there is room, we inject the oncoming vehicle so we can check for occlusion
        self.inject_candidate(oncoming_agent_to_insert, potential_oncoming_vehicle_goal, runner, scenario.get_time_point(0))

        relavant_agent_tokens = [oncoming_vehicle_token]      
        #check which vehicles are currently visible to the ego vehicle
        manager = WedgeOcclusionManager(scenario)
        visible_relavant_agents = set(relavant_agent_tokens).intersection(
            manager._compute_visible_agents(initial_ego_agent_object, runner.simulation._observations.get_observation())
        )
        
        agents_to_insert = [oncoming_agent_to_insert]
        goals_to_insert = [potential_oncoming_vehicle_goal]
        time_points = [scenario.get_time_point(0)]
        
        #if all relavant vehicles are already occluded, then we dont need to try to inject an occludor
        if len(visible_relavant_agents) == 0:
            num = 0
            modifier_string = "_oncoming_vehicle_injection_for_left_turn_and_occlusion_injection_" + str(num)
            new_sim_runner = copy.deepcopy(runner)
            modification = OncomingInjectionForLeftTurnAndOcclusionInjectionModification(agents_to_insert, goals_to_insert, time_points, modifier_string)
            modification.modify(new_sim_runner.simulation)
            new_sim_runner.simulation.modification = modification
            return [new_sim_runner]
        
        #otherwise, we need to try to inject an occludor
            
        
        
        
class OncomingInjectionForLeftTurnAndOcclusionInjectionModification(AbstractModification):
    def __init__(self, inserted_agents: List[Agent], goal_states: List[StateSE2], time_points: List[TimePoint], modifier_string: str):
        super().__init__(modifier_string) #maybe we will need this later
        self.inserted_agents = inserted_agents
        self.goal_states = goal_states
        self.time_points = time_points

    def modify(self, simulation: Simulation) -> None:
        for inserted_agent, goal_state, time_point in zip(self.inserted_agents, self.goal_states, self.time_points):
            simulation._observations.add_agent_to_scene(
                inserted_agent, goal_state, time_point, simulation
            )
        simulation.scenario._modifier = self.modifier_string