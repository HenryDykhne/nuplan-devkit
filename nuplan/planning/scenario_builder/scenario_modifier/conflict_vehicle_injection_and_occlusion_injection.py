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
from shapely.geometry import Polygon, Point, MultiPolygon, MultiLineString, LineString, MultiPoint, GeometryCollection
from shapely import union_all
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractModification, AbstractScenarioModifier
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.scenario_builder.scenario_modifier.occlusion_injection_modifier import OcclusionInjectionModifier
from nuplan.planning.simulation.occlusion.wedge_occlusion_manager import WedgeOcclusionManager

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation

class ConflictInjectionAndOcclusionInjectionModifier(OcclusionInjectionModifier):
    CONFLICT_RADIUS = 50.0
    SPEED_OFFSETS = [-5.0, 0.0, 5.0]
    MIN_EGO_SPEED = 1.0
    MAX_ALLOWED_SPEED = 15.0
    MIN_ALLOWED_SPEED = 1.0
    def __init__(self):
        super().__init__() #maybe we will need this later
        
    def get_lane_object_heading(self, lane_object: LaneGraphEdgeMapObject) -> float:
        line = lane_object.baseline_path.linestring
        first = line.coords[0]
        first = Point2D(*first)
        heading = lane_object.baseline_path.get_nearest_pose_from_position(first).heading
        return heading
        
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        scenario = runner.scenario
        # determine if ego moves through a lane connector
        # determine what kind of intersection transition we make
        # compute potential conflict lane connectors
        
        crossing_lane_connector = self.how_does_ego_cross_intersection(runner)
        if crossing_lane_connector is None:
            #print(f'Warning: Ego does not cross an intersection in scenario with token: {scenario.token}.')
            return [] 
        
        traffic_light_status = self.get_traffic_light_status_at_iteration(0, scenario)
        if crossing_lane_connector.id in traffic_light_status[TrafficLightStatusType.RED]:
            #print(f'Warning: Crossing lane connector {crossing_lane_connector.id} in scenario {scenario.token} has a red traffic light')
            return []   
        
        ego_object = scenario.get_ego_state_at_iteration(0)
        ego_agent = ego_object.agent
        
        map_api = scenario.map_api
        _, ego_lane_level_route_plan = runner.simulation._observations._get_roadblock_path(
                    ego_agent,
                    scenario.get_expert_goal_state()
                )
        
        if ego_lane_level_route_plan is None:
            #print(f'Warning: ego in scenario {scenario.token} has no route plan')
            return []
        
        first = crossing_lane_connector.baseline_path.linestring.coords[0]
        first = Point2D(*first)
        
        potential_conflicting_connectors = map_api.get_proximal_map_objects(first, self.CONFLICT_RADIUS, [SemanticMapLayer.LANE_CONNECTOR])[SemanticMapLayer.LANE_CONNECTOR]
        conflicting_connectors = [connector for connector in potential_conflicting_connectors if connector.baseline_path.linestring.intersects(crossing_lane_connector.baseline_path.linestring)]
        valid_conflicting_connectors = [connector for connector in conflicting_connectors if connector.id not in traffic_light_status[TrafficLightStatusType.RED]]
        
        if crossing_lane_connector.turn_type == LaneConnectorType.STRAIGHT: #### HANDCRAFTED SAFEGUARD
            # if we are going straight, we want to remove any conflicting connectors that are also going straight to avoid cars rolling out perpendicular to us where they realisticly would not
            valid_conflicting_connectors = [connector for connector in valid_conflicting_connectors if connector.turn_type != LaneConnectorType.STRAIGHT]
        elif crossing_lane_connector.turn_type == LaneConnectorType.LEFT: #### HANDCRAFTED SAFEGUARD
            #if we are turning left, we only really want oncoming traffic
            temp = []
            for connector in valid_conflicting_connectors:
                if connector.turn_type == LaneConnectorType.STRAIGHT:
                    connector_heading = self.get_lane_object_heading(connector)
                    crossing_lane_connector_heading = self.get_lane_object_heading(crossing_lane_connector)
                    angle = angle_diff(connector_heading, crossing_lane_connector_heading, 2 * np.pi) # returns signed angle, to, from
                    if abs(angle) < np.pi/4: #same direction
                        continue
                    elif abs(angle) > np.pi - (np.pi/4): #oncoming
                        temp.append(connector)
                    #continue if its not oncoming
            valid_conflicting_connectors = temp
                    
        
        if len(valid_conflicting_connectors) == 0:
            #print(f'Warning: ego in scenario {scenario.token} has no conflict lanes intersecting its path')
            return []
        
        distance_to_crossing_lane_connector = 0.0
        for i, map_object in enumerate(ego_lane_level_route_plan):
            if map_object.id == crossing_lane_connector.id:
                break
            if i == 0:
                progress_along_map_object = map_object.baseline_path.get_nearest_arc_length_from_position(ego_agent.center)
                distance_to_crossing_lane_connector += map_object.baseline_path.length - progress_along_map_object
            else:
                distance_to_crossing_lane_connector += map_object.baseline_path.length
            
            if i > 3:
                #print(f'Warning: ego in scenario {scenario.token} has its lane connector too far away from the start of the route plan.')
                return []
        
        all_modified_simulation_runners = []
        valid_conflicting_connectors = [connector for connector in valid_conflicting_connectors if connector.id != crossing_lane_connector.id] #removing the connector ego is using
        for conflict_connector in valid_conflicting_connectors:
            conflict_point = crossing_lane_connector.baseline_path.linestring.intersection(conflict_connector.baseline_path.linestring)
            if isinstance(conflict_point, GeometryCollection) or isinstance(conflict_point, MultiPoint) or isinstance(conflict_point, MultiLineString):
                conflict_point = conflict_point.geoms[0]
            if isinstance(conflict_point, LineString):
                conflict_point = conflict_point.coords[0]
                conflict_point = Point2D(*first)
            conflict_point = Point2D(conflict_point.x, conflict_point.y)
            
            distance_to_conflict_from_begining_of_conflict_connector = conflict_connector.baseline_path.get_nearest_arc_length_from_position(conflict_point)   
            ego_distance_to_conflict_point = distance_to_crossing_lane_connector + distance_to_conflict_from_begining_of_conflict_connector
            
            ego_speed = max(ego_agent.velocity.magnitude(), self.MIN_EGO_SPEED)
            time_to_conflict = ego_distance_to_conflict_point / ego_speed
            for ego_speed_offset in self.SPEED_OFFSETS:
                conflict_vehicle_speed = ego_speed + ego_speed_offset
                if conflict_vehicle_speed < self.MIN_ALLOWED_SPEED or conflict_vehicle_speed > self.MAX_ALLOWED_SPEED:
                    continue
                distance_to_conflict_point_for_conflict_vehicle = conflict_vehicle_speed * time_to_conflict
                distance_to_conflict_from_begining_of_conflict_connector = conflict_connector.baseline_path.get_nearest_arc_length_from_position(conflict_point)
                distance_from_end_of_conflict_connector_parent_lane = distance_to_conflict_point_for_conflict_vehicle - distance_to_conflict_from_begining_of_conflict_connector
                
                conflict_connector_parent_lane = conflict_connector.incoming_edges[0]
                distance_from_begining_of_conflict_connector_parent_lane = conflict_connector_parent_lane.baseline_path.length - distance_from_end_of_conflict_connector_parent_lane
                
                potential_conflict_vehicle_spawn_point = conflict_connector_parent_lane.baseline_path.linestring.interpolate(distance_from_begining_of_conflict_connector_parent_lane)
                potential_conflict_vehicle_spawn_point = Point2D(potential_conflict_vehicle_spawn_point.x, potential_conflict_vehicle_spawn_point.y)
                potential_conflict_vehicle_spawn_pose = conflict_connector_parent_lane.baseline_path.get_nearest_pose_from_position(potential_conflict_vehicle_spawn_point)
                
                #finding a goal for our conflict vehicle. It will automatically be extended if it is too close.
                goal_segment = conflict_connector.outgoing_edges[0] 
                *_, last = goal_segment.baseline_path.linestring.coords #we use the last point in a linestring far away to start setting our goal
                last = Point(last)
                goal_point = Point2D(last.x, last.y)
                potential_conflict_vehicle_goal = goal_segment.baseline_path.get_nearest_pose_from_position(goal_point)
        
                potential_conflict_vehicle_speed = ego_agent.velocity.magnitude() #we will use the same speed as the ego, so hopefully the oncoming vehicle will arive at the conflict point at the same time
                potential_conflict_vehicle_heading = potential_conflict_vehicle_spawn_pose.heading
                potential_conflict_vehicle_velocity = StateVector2D(potential_conflict_vehicle_speed * math.cos(potential_conflict_vehicle_heading), potential_conflict_vehicle_speed * math.sin(potential_conflict_vehicle_heading))
        
                conflict_vehicle_token = "conflict_inserted"
                conflict_agent_to_insert = Agent(
                    tracked_object_type=TrackedObjectType.VEHICLE,
                    oriented_box=OrientedBox(potential_conflict_vehicle_spawn_pose, 5, 2, 2),
                    velocity=potential_conflict_vehicle_velocity,
                    metadata=SceneObjectMetadata(scenario.get_time_point(0).time_us,conflict_vehicle_token, -2, conflict_vehicle_token),
                    angular_velocity=0.0,
                )    
        
        
                objects_to_avoid = list(AGENT_TYPES | STATIC_OBJECT_TYPES) #grab anything we could crash into
                avoid = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(objects_to_avoid)
                avoid_geoms = [ego_agent.box.geometry]#to ensure ego gets added to the geometry since it is not a tracked object
                inject_poly = Polygon(conflict_agent_to_insert.box.all_corners())
                for obj in avoid:
                    avoid_geoms.append(obj.box.geometry)
                avoid_geoms = MultiPolygon(avoid_geoms)
        
                if inject_poly.intersects(avoid_geoms.buffer(self.MINIMUM_SPAWNING_DISTANCE)): #if injected agent intersects with other agents
                    continue
        
                #now that we know there is room, we inject the conflict vehicle so we can check for occlusion
                self.inject_candidate(conflict_agent_to_insert, potential_conflict_vehicle_goal, runner, scenario.get_time_point(0))
                
                # for the agent we are about to insert, we want to make sure it has a reasonable time to collision with any vehicle in the scene
                vehicle_agents = [agent for agent in runner.simulation._observations.get_observation().tracked_objects.tracked_objects if agent.tracked_object_type == TrackedObjectType.VEHICLE]
                rough_time_to_collision = self.calculate_rough_min_time_to_collision(ego_agent, vehicle_agents)
                if rough_time_to_collision is not None and rough_time_to_collision < self.MIN_ALLOWED_TIME_TO_COLLISION:
                    #print(f'Warning: vehicle in scenario {scenario.token} is has too low a time to collision')
                    self.remove_candidate(conflict_agent_to_insert, runner)
                    continue

                relavant_agent_tokens = [conflict_vehicle_token]
                #check which vehicles are currently visible to the ego vehicle
                manager = WedgeOcclusionManager(scenario)
                visible_relavant_agents = set(relavant_agent_tokens).intersection(
                    manager._compute_visible_agents(ego_object, runner.simulation._observations.get_observation())
                )
        
                agents_to_insert = [conflict_agent_to_insert]
                goals_to_insert = [potential_conflict_vehicle_goal]
                time_points = [scenario.get_time_point(0)]
        
                #if all relavant vehicles are already occluded, then we dont need to try to inject an occludor
                base_modifier_string = "_conflict_injection_and_occlusion_injection_"
                if len(visible_relavant_agents) == 0:
                    num = 0
                    modifier_string = base_modifier_string + str(num) + "_natural_occlusion_" + str(ego_speed_offset) + "_" + conflict_connector.id
                    new_sim_runner = copy.deepcopy(runner)
                    modification = ConflictInjectionAndOcclusionInjectionModification(agents_to_insert, goals_to_insert, time_points, modifier_string)
                    modification.modify(new_sim_runner.simulation)
                    new_sim_runner.simulation.modification = modification
                    all_modified_simulation_runners.append(new_sim_runner)
                    # remove injected vehicle from original scenario
                    self.remove_candidate(conflict_agent_to_insert, runner)
                    continue
        
                #otherwise, we need to try to inject an occluder
                object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE]
                agents = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(object_types)
                agents.extend(agents_to_insert)

                full_fov_poly = self.generate_full_fov_polygon(ego_agent, agents, relavant_agent_tokens)

                if full_fov_poly.area == 0:
                    continue
        
                #we get a superset of lane ids that we deem valid connectors. to insert agents onto. we want these agents to be leading or adjacent to at least one of the relavant agents or ego to maximize the length of the occlusion
                _, ego_lane_level_route_plan = runner.simulation._observations._get_roadblock_path(
                    scenario.get_ego_state_at_iteration(0).agent,
                    scenario.get_expert_goal_state()
                )
                lane_objects_to_prune_by = []#### HANDCRAFTED SAFEGUARD
                for lane_object in ego_lane_level_route_plan:
                    lane_objects_to_prune_by.append(lane_object)
                    
                relavant_agents = []
                for agent in agents:
                    if agent.track_token in relavant_agent_tokens:
                        relavant_agents.append(agent)

                for agent, goal in zip(agents_to_insert, goals_to_insert):
                    _, relavant_agent_route_plan = runner.simulation._observations._get_roadblock_path(agent, goal)
                    if relavant_agent_route_plan is None:
                        continue
                    for lane_object in relavant_agent_route_plan:
                        lane_objects_to_prune_by.append(lane_object)
                        
                #this is a special case for right turns. we want to allow for a passing occlusion from an  RT: Reveal type scenario
                if crossing_lane_connector.turn_type == LaneConnectorType.RIGHT: #### HANDCRAFTED SAFEGUARD
                    valid_left_turning_conflict_connectors = [connector for connector in valid_conflicting_connectors if connector.turn_type == LaneConnectorType.LEFT]
                    lane_objects_to_prune_by.extend(valid_left_turning_conflict_connectors)
                
        
                lane_objects_to_prune_by = list(dict.fromkeys(lane_objects_to_prune_by)) #remove duplicates
                centerlines, map_polys = self.get_map_geometry(ego_agent, scenario.map_api, traffic_light_status, lane_objects_to_prune_by)
                
                #now that we have the right centerlines, we can find the potential occlusion points. here we make sure our centerlines are within the full_fov_poly
                potential_occlusion_centerlines: MultiLineString = centerlines.intersection(
                    full_fov_poly
                )
                
                if potential_occlusion_centerlines.is_empty:
                    return []
        
                discretized_points = self.discretize_centerline_segments(potential_occlusion_centerlines)
                
                # sample from around feasible points to introduce more variety of scenes we are able to create
                potential_spawn_points = ( #TODO, maybe sample for each point after checking if original point is too close
                    discretized_points if (not self.ADD_NOISE)
                    else self.sample_around_points(discretized_points, self.SAMPLE_SPAWN_POINT_STDEV)
                )
        
                objects_to_avoid = list(AGENT_TYPES | STATIC_OBJECT_TYPES) #grab anything we could crash into
                avoid = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(objects_to_avoid)
                avoid_geoms = [ego_agent.box.geometry]#to ensure ego gets added to the geometry since it is not a tracked object
                for obj in avoid:
                    avoid_geoms.append(obj.box.geometry)
                avoid_geoms.append(conflict_agent_to_insert.box.geometry) #dont want to crash into this either
                avoid_geoms = MultiPolygon(avoid_geoms)
                candidate_occluding_spawn_points = self._filter_to_valid_spawn_points(
                    potential_spawn_points,
                    avoid_geoms,
                    map_polys
                )
        
                #now we try to inject the potential occluders with the goal of occluding our newly inseted relavant agents
                modified_simulation_runners = []
                modifier_number = 0
                #check which vehicles are currently visible to the ego vehicle
                manager = WedgeOcclusionManager(scenario)
                visible_relavant_agents = set(relavant_agent_tokens).intersection(
                    manager._compute_visible_agents(ego_object, runner.simulation._observations.get_observation())
                )
                points_injected_at = MultiPoint()
                iteration = runner.simulation._time_controller.get_iteration()
                for point in candidate_occluding_spawn_points:
                    # check if the point is too close to other injection sites
                    
                    dist = points_injected_at.distance(point)
                    if not math.isnan(dist) and dist < self.MIN_DISTANCE_BETWEEN_INJECTIONS:
                        continue
                    # inject vehicle at point,
                    candidate, goal = self.generate_injection_candidate(point, runner, scenario.map_api, traffic_light_status, [conflict_agent_to_insert])
                    
                    if candidate is None:
                        continue
                    
                    inject_poly = Polygon(candidate.box.all_corners())
                    if inject_poly.intersects(avoid_geoms.buffer(self.MINIMUM_SPAWNING_DISTANCE)): #if injected agent intersects with other agents
                        continue
                    
                    self.inject_candidate(candidate, goal, runner, iteration.time_point)
                    # for the agent we are about to insert, we want to make sure it has a reasonable time to collision with any vehicle in the scene
                    vehicle_agents = [agent for agent in runner.simulation._observations.get_observation().tracked_objects.tracked_objects if agent.tracked_object_type == TrackedObjectType.VEHICLE]
                    rough_time_to_collision = self.calculate_rough_min_time_to_collision(ego_agent, vehicle_agents)
                    if rough_time_to_collision is not None and rough_time_to_collision < self.MIN_ALLOWED_TIME_TO_COLLISION:
                        #print(f'Warning: vehicle in scenario {scenario.token} is has too low a time to collision')
                        self.remove_candidate(candidate, runner)
                        continue
                    
                    # check if new occlusion is created among relavant vehicles
                    new_visible_relavant_agents = set(relavant_agent_tokens).intersection(
                        manager._compute_visible_agents(ego_object, runner.simulation._observations.get_observation())
                    )
                    
                    # remove injected vehicle from original scenario
                    self.remove_candidate(candidate, runner)
                    
                    if len(visible_relavant_agents.difference(new_visible_relavant_agents)) > 0:
                        new_sim_runner = copy.deepcopy(runner)
                        ai = copy.deepcopy(agents_to_insert)
                        gi = copy.deepcopy(goals_to_insert)
                        ti = copy.deepcopy(time_points)
                        ai.append(candidate)
                        gi.append(goal)
                        ti.append(iteration.time_point)
                        
                        modifier_string = base_modifier_string + str(modifier_number) + "_" + str(ego_speed_offset) + "_" + conflict_connector.id
                        modification = ConflictInjectionAndOcclusionInjectionModification(ai, gi, ti, modifier_string)
                        modification.modify(new_sim_runner.simulation)
                        new_sim_runner.simulation.modification = modification
                        modified_simulation_runners.append(new_sim_runner)
                        points_injected_at = points_injected_at.union(point)
                        modifier_number += 1
                
                self.remove_candidate(conflict_agent_to_insert, runner)        
                all_modified_simulation_runners.extend(modified_simulation_runners)
                
        return all_modified_simulation_runners
        
class ConflictInjectionAndOcclusionInjectionModification(AbstractModification):
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