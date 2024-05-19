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

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneConnector, LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import LaneConnectorType, SemanticMapLayer
from nuplan.database.utils.measure import angle_diff
from shapely.geometry import Polygon, Point, MultiPolygon, MultiLineString, LineString, MultiPoint, GeometryCollection
from shapely import union_all
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from nuplan.planning.scenario_builder.scenario_modifier.cross_conflict_with_occlusion_injection_modifier import CrossConflictWithOcclusionInjectionModification

from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractModification, AbstractScenarioModifier
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.scenario_builder.scenario_modifier.conflict_vehicle_injection_and_occlusion_injection import ConflictInjectionAndOcclusionInjectionModifier
from nuplan.planning.scenario_builder.scenario_modifier.occlusion_injection_modifier import OcclusionInjectionModifier
from nuplan.planning.simulation.occlusion.wedge_occlusion_manager import WedgeOcclusionManager

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation

class CrossConflictWithOcclusionInjectionModifierCopy(ConflictInjectionAndOcclusionInjectionModifier):
    CONFLICT_RADIUS = 50.0
    SPEED_OFFSETS = [-5.0, 0.0, 5.0]
    MIN_EGO_SPEED = 1.0
    MAX_ALLOWED_SPEED = 15.0
    MIN_ALLOWED_SPEED = 1.0
    
    def __init__(self, cfg):
        super().__init__() #maybe we will need this later
        
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        scenario = runner.scenario
        map_api = scenario.map_api
        all_modified_simulation_runners = []
        ego_object = scenario.get_ego_state_at_iteration(0)
        ego_agent = ego_object.agent
        traffic_light_status = self.get_traffic_light_status_at_iteration(0, scenario)

        crossing_lane_connector = self.how_does_ego_cross_intersection(runner)
        traffic_light_status = self.get_traffic_light_status_at_iteration(0, scenario)

        _, ego_lane_level_route_plan = runner.simulation._observations._get_roadblock_path(
                    ego_agent,
                    scenario.get_expert_goal_state()
                )

        if crossing_lane_connector is None or crossing_lane_connector.id in traffic_light_status[TrafficLightStatusType.RED] or ego_lane_level_route_plan is None:
            return [] 


        
        first = crossing_lane_connector.baseline_path.linestring.coords[0]
        first = Point2D(*first)
        
        potential_conflicting_connectors = map_api.get_proximal_map_objects(first, self.CONFLICT_RADIUS, [SemanticMapLayer.LANE_CONNECTOR])[SemanticMapLayer.LANE_CONNECTOR]
        conflicting_connectors = [connector for connector in potential_conflicting_connectors if connector.baseline_path.linestring.intersects(crossing_lane_connector.baseline_path.linestring)]
        # removing diverging and sequential connectors
        conflicting_connectors_without_shared_incoming_edge = [connector for connector in conflicting_connectors if not (connector.incoming_edges[0].id == crossing_lane_connector.incoming_edges[0].id)]
        # removing merging connectors
        conflicting_connectors_without_shared_incoming_or_outgoing_edge = [connector for connector in conflicting_connectors_without_shared_incoming_edge if not (connector.outgoing_edges[0].id == crossing_lane_connector.outgoing_edges[0].id)]
        valid_conflicting_connectors = [connector for connector in conflicting_connectors_without_shared_incoming_or_outgoing_edge if connector.id not in traffic_light_status[TrafficLightStatusType.RED]]
    
        if len(valid_conflicting_connectors) == 0:
            return []


        vehicle_agents = [agent for agent in runner.simulation._observations.get_observation().tracked_objects.tracked_objects if agent.tracked_object_type == TrackedObjectType.VEHICLE]


        for conflict_agent_to_insert in vehicle_agents:
            
                                        
            relavant_agent_tokens = [conflict_agent_to_insert.metadata.track_token]
        
            #if all relavant vehicles are already occluded, then we dont need to try to inject an occludor
            base_modifier_string = "_cross_occlusion_injection_"
            #check which vehicles are currently visible to the ego vehicle
            manager = WedgeOcclusionManager(scenario)
            visible_relavant_agents = set(relavant_agent_tokens).intersection(
                manager._compute_visible_agents(ego_object, runner.simulation._observations.get_observation())
            )

            #otherwise, we need to try to inject an occluder
            object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE]
            agents = []
            agents = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(object_types)

            full_fov_poly = self.generate_full_fov_polygon(ego_agent, agents, relavant_agent_tokens)

            if full_fov_poly.area == 0:
                self.remove_candidate(conflict_agent_to_insert, runner)
                continue
            
            centerlines, map_polys = self.get_map_geometry(ego_agent, scenario.map_api, traffic_light_status)
            
            #now that we have the right centerlines, we can find the potential occlusion points. here we make sure our centerlines are within the full_fov_poly
            potential_occlusion_centerlines: MultiLineString = centerlines.intersection(
                full_fov_poly
            )
            
            if potential_occlusion_centerlines.is_empty:
                self.remove_candidate(conflict_agent_to_insert, runner)
                continue

            discretized_points = self.discretize_centerline_segments(potential_occlusion_centerlines)
            
            # sample from around feasible points to introduce more variety of scenes we are able to create
            potential_spawn_points = ( #TODO, maybe sample for each point after checking if original point is too close
                discretized_points if (not self.ADD_NOISE)
                else self.sample_around_points(discretized_points, self.SAMPLE_SPAWN_POINT_STDEV)
            )

            avoid = runner.simulation._observations.get_observation().tracked_objects.tracked_objects #grab anything we could crash into
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
                candidate, goal = self.generate_occlusion_candidate_cross(point, runner, scenario.map_api, traffic_light_status, [conflict_agent_to_insert])
                
                if candidate is None:
                    continue
                
                inject_poly = Polygon(candidate.box.all_corners())
                if inject_poly.intersects(avoid_geoms.buffer(self.MINIMUM_SPAWNING_DISTANCE)): #if injected agent intersects with other agents
                    continue
                
                self.inject_candidate(candidate, goal, runner, iteration.time_point)
                # for the agent we are about to insert, we want to make sure it has a reasonable time to collision with any vehicle in the scene
                vehicle_agents = [agent for agent in runner.simulation._observations.get_observation().tracked_objects.tracked_objects if \
                                    agent.tracked_object_type == TrackedObjectType.VEHICLE]
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
                    
                    modifier_string = base_modifier_string + str(modifier_number) + "_" + str(round(ego_speed_offset, 1)) + "_" + conflict_connector.id
                    modification = CrossConflictWithOcclusionInjectionModification(ai, gi, ti, modifier_string)
                    modification.modify(new_sim_runner.simulation)
                    new_sim_runner.simulation.modification = modification
                    modified_simulation_runners.append(new_sim_runner)
                    points_injected_at = points_injected_at.union(point)
                    modifier_number += 1
            
            self.remove_candidate(conflict_agent_to_insert, runner)
            all_modified_simulation_runners.extend(modified_simulation_runners)
                    
        return all_modified_simulation_runners
            
    def generate_occlusion_candidate_cross(self, point: Point, runner: SimulationRunner, map_api: AbstractMap, traffic_light_status: Dict[TrafficLightStatusType, List[str]], optional_extra_agents: List[Agent] = []) -> Tuple[Agent, StateSE2]:
        """We generate the agent state and goal pair for the occlusion agent for the cross conflict
        :param point: xy coords of the injection point
        :param runner: SimulationRunner to inject into
        :param map_api: map_api of simulation
        :param traffic_light_status: traffic light status at the time of injection
        :return: tuple of agent state and goal
        """
        scenario = runner.scenario
        point2d = Point2D(point.x, point.y)
        if map_api.is_in_layer(point2d, SemanticMapLayer.LANE):
            layer = SemanticMapLayer.LANE
        elif map_api.is_in_layer(point2d, SemanticMapLayer.INTERSECTION):
            layer = SemanticMapLayer.LANE_CONNECTOR
        
        segments: List[LaneGraphEdgeMapObject] = map_api.get_all_map_objects(point2d, layer)
        if not segments:
            return None, None #failed to inject agent
        
        segments = [segment for segment in segments if segment.id not in traffic_light_status[TrafficLightStatusType.RED]]
        
        sorted_segments = sorted(segments, key=lambda segment: abs(segment.baseline_path.get_nearest_pose_from_position(point2d).distance_to(point2d)), reverse=False)
        
        ego_agent = scenario.initial_ego_state.agent
        
        closest_segment = sorted_segments[0]
        heading = closest_segment.baseline_path.get_nearest_pose_from_position(point2d).heading
        
        ego_center = Point(ego_agent.center.x, ego_agent.center.y)
        
        speed_setter_catch = closest_segment.polygon
        for segment in closest_segment.outgoing_edges:
            speed_setter_catch = speed_setter_catch.union(segment.polygon)
        if isinstance(closest_segment, LaneConnector) and len(closest_segment.incoming_edges) > 0 and closest_segment.incoming_edges[0].polygon.contains(ego_center):#if we are right in a lane connector, we might want to check the incoming lane for people behind us as well
            for segment in closest_segment.incoming_edges:
                speed_setter_catch = speed_setter_catch.union(segment.polygon)
        
        # we dont want to spawn directly behind ego since even though it might cause an occlusion, such an occlusion is unlikely to result in a collision for ego    
        if closest_segment.polygon.contains(ego_center):
            ego_progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(ego_center)
            self_progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(point2d)
            if ego_progress > self_progress:
                return None, None
        elif any([outgoing.polygon.contains(ego_center) for outgoing in closest_segment.outgoing_edges]):
            return None, None
        
        moving_agents = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(AGENT_TYPES) #its fine to use initial object here because even if we removed them, they can still provide important contextual information about realistic car speeds in the area
        moving_agents.append(scenario.initial_ego_state.agent)
        moving_agents.extend(optional_extra_agents)
        agents_within_range = []
        vehicles_within_range = []
        agents_within_range_and_roadblock = []
        for agent in moving_agents:
            center_poly = Point(agent.center.x, agent.center.y)
            if agent.center.distance_to(point2d) < self.LEAD_FOLLOW_AGENT_RANGE and speed_setter_catch.contains(center_poly):
                agents_within_range.append(agent)
                if agent.tracked_object_type == TrackedObjectType.VEHICLE or agent.tracked_object_type == TrackedObjectType.EGO:
                    vehicles_within_range.append(agent)
            if agent.center.distance_to(point2d) < self.SIDE_AGENT_RANGE \
                    and (agent.tracked_object_type == TrackedObjectType.VEHICLE or agent.tracked_object_type == TrackedObjectType.BICYCLE or agent.tracked_object_type == TrackedObjectType.EGO) \
                    and closest_segment.parent.contains_point(agent.center):
                agents_within_range_and_roadblock.append(agent)
        sorted_agents = sorted(agents_within_range, key=lambda x: (x.center.distance_to(point2d)), reverse=False) #sorts closest to furthest
        sorted_vehicles = sorted(vehicles_within_range, key=lambda x: (x.center.distance_to(point2d)), reverse=False) #sorts closest to furthest


        if len(sorted_agents) > 0:
            speed = sorted_agents[0].velocity.magnitude()
            velocity = StateVector2D(speed * math.cos(heading), speed * math.sin(heading))#select closest agent in lane region (this is to avoid crashing into pedestriens)
        else:
            speed_limit = closest_segment.speed_limit_mps
            if speed_limit is None or speed_limit <= 1:
                speed_limit = self.DEFAULT_SPEED_LIMIT_MPS
            velocity = StateVector2D(speed_limit * math.cos(heading), speed_limit * math.sin(heading))
            
        all_outgoing_edges_red = False
        if len(closest_segment.outgoing_edges) > 0:
            all_outgoing_edges_red = all(edge.id in traffic_light_status[TrafficLightStatusType.RED] for edge in closest_segment.outgoing_edges)
            
        if all_outgoing_edges_red: #if all outgoing edges are red, we need to take care to bring the vehicle to a stop to avoid running into the road.
            progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(point2d)
            distance_to_end = closest_segment.baseline_path.length - progress
            speed = min(velocity.magnitude(), distance_to_end / self.TIME_TO_END_OF_SEGMENT)
            velocity = StateVector2D(speed * math.cos(heading), speed * math.sin(heading))
        
        goal = None
        if len(sorted_vehicles) > 0:
            current_iter = 0
            if sorted_vehicles[0].metadata.track_token != ego_agent.metadata.track_token and sorted_vehicles[0].metadata.track_token not in [agent.metadata.track_token for agent in optional_extra_agents]:
                goal = runner.simulation._observations._get_historical_agent_goal(sorted_vehicles[0], current_iter)
                goal = copy.deepcopy(goal)
            else: # in case this vehicle is the ego or the conflict vehicle, we actually want it to take a different path than whatever agent it is copying in order to get out of the way
                _, lane_level_route_plan = runner.simulation._observations._get_roadblock_path(
                    ego_agent,
                    scenario.get_expert_goal_state()
                )
                temp = closest_segment
                while temp.id in [x.id for x in lane_level_route_plan]:
                    if len(temp.outgoing_edges) == 1:
                        temp = temp.outgoing_edges[0]
                    elif len(temp.outgoing_edges) == 0:
                        return None, None
                    else:
                        for outgoing in temp.outgoing_edges:
                            if outgoing.id not in [x.id for x in lane_level_route_plan]:
                                temp = outgoing
                                break
                goal_segment = temp
                *_, last = goal_segment.baseline_path.linestring.coords #we use the last point in a linestring far away to start setting our goal
                last = Point(last)
                goal_point = Point2D(last.x, last.y)
                goal = goal_segment.baseline_path.get_nearest_pose_from_position(goal_point)

        if goal is None and len(agents_within_range_and_roadblock) > 0:
            goal_segment = closest_segment #if we cant find a vehicle to copy the goal of, we make our own and let the automated path extension handle the rest
            *_, last = goal_segment.baseline_path.linestring.coords #we use the last point in a linestring far away to start setting our goal
            last = Point(last)
            goal_point = Point2D(last.x, last.y)
            goal = goal_segment.baseline_path.get_nearest_pose_from_position(goal_point)
            
        if goal is None:
            return None, None #if still have no goal, this is likely not a great inection spot
        
        #craft agent to inject
        agent_to_insert = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=OrientedBox(StateSE2(point2d.x, point2d.y, heading), 5, 2, 2),
            velocity=StateVector2D(velocity.x, velocity.y),
            metadata=SceneObjectMetadata(scenario.get_time_point(0).time_us, "oclusion_inserted", -2, "oclusion_inserted"),
            angular_velocity=0.0,
        )

        return agent_to_insert, goal
    