from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

import copy

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, MultiLineString, LineString, MultiPoint
from shapely import union_all

from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import Lane, LaneConnector, LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.database.nuplan_db_orm.traffic_light_status import TrafficLightStatus

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractScenarioModifier
from nuplan.planning.simulation.observation.ml_planner_agents import MLPlannerAgents

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint

from nuplan.planning.simulation.occlusion.wedge_occlusion_manager import WedgeOcclusionManager



import math

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
class OcclusionInjectionModifier(AbstractScenarioModifier):
    DISTANCE_BETWEEN_DISCRETIZED_POINTS = 0.5
    TOLERANCE = 0.2
    MAP_RADIUS = 200.0
    SAMPLE_SPAWN_POINT_STDEV = 0.20
    ADD_NOISE = True
    MINIMUM_SPAWNING_DISTANCE = 0.2 # REPLACE THIS CONSTANT with a function of the speed of the agent you are checking
    MIN_DISTANCE_BETWEEN_INJECTIONS = 0.3
    LEAD_FOLLOW_AGENT_RANGE = 15.0
    EXTENSION_STEPS = 3
    TIME_TO_END_OF_SEGMENT = 2.0 #seconds
    
    def __init__(self):
        super().__init__() #maybe we will need this later
        np.random.seed(0)
        
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        """We convert one abstract scenario into many abstract scenarios by modifying the scenario in some way.
        :param runner: a scenario
        :return: we return a list of runners that are modified versions of the input scenario
        """
        scenario = runner.scenario
        relavant_agent_tokens = self.find_relavant_agents(runner.simulation._observations, scenario)
        
        # here we generate the field of view polygons for each relavant agent, taking into account potential occlusions even from nonrelavant agents
        ego_object = scenario.get_ego_state_at_iteration(0)
        ego_agent = ego_object.agent
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE]
        agents = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(object_types)
        
        full_fov_poly = self.generate_full_fov_polygon(ego_agent, agents, relavant_agent_tokens)
                
        if full_fov_poly.area == 0:
            return []
        
        traffic_light_status = self.get_traffic_light_status_at_iteration(0, scenario)
        centerlines, map_polys = self.get_map_geometry(ego_agent, scenario.map_api, traffic_light_status)
        
        potential_occlusion_centerlines: MultiLineString = centerlines.intersection(
            full_fov_poly
        )
        
        discretized_points = self.discretize_centerline_segments(potential_occlusion_centerlines)
        
        # sample from around feasible points to introduce more variety of scenes we are able to create
        potential_spawn_points = ( #TODO, maybe sample for each point after checking if original point is too close
            discretized_points if (not self.ADD_NOISE)
            else self.sample_around_points(discretized_points, self.SAMPLE_SPAWN_POINT_STDEV)
        )
        
        objects_to_avoid = list(AGENT_TYPES | STATIC_OBJECT_TYPES) #grab anything we could crash into
        avoid = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(objects_to_avoid)
        avoid_geoms = []
        for obj in avoid:
            avoid_geoms.append(obj.box.geometry)
        avoid_geoms = MultiPolygon(avoid_geoms)
        candidate_occluding_spawn_points = self._filter_to_valid_spawn_points(
            potential_spawn_points,
            avoid_geoms,
            map_polys
        )
        
        print(len(candidate_occluding_spawn_points))
        modified_simulation_runners = []
        modifier_string = "occlusion_injection_"
        modifier_number = 0
        #check which vehicles are currently visible to the ego vehicle
        manager = WedgeOcclusionManager(scenario)
        print(len(runner.simulation._observations._agents))
        visible_relavant_agents = set(relavant_agent_tokens).intersection(
            manager._compute_visible_agents(ego_object, runner.simulation._observations)
        )
        points_injected_at = MultiPoint()
        iter = runner.simulation._time_controller.get_iteration()
        for point in candidate_occluding_spawn_points:
            # check if the point is too close to other injection sites
            dist = points_injected_at.distance(point)
            if not math.isnan(dist) and dist < self.MIN_DISTANCE_BETWEEN_INJECTIONS:
                continue
            # inject vehicle at point,
            candidate, goal = self.generate_injection_candidate(point, runner, scenario.map_api, traffic_light_status)
            
            if candidate is None:
                continue
            
            inject_poly = Polygon(candidate.box.all_corners)
            if inject_poly.intersect(avoid_geoms): #if injected intersects with other agents
                continue
            
            self.inject_candidate(candidate, goal, runner, iter.time_point)
            
            # check if new occlusion is created among relavant vehicles
            new_visible_relavant_agents = set(relavant_agent_tokens).intersection(
                manager._compute_visible_agents(ego_object, runner.simulation._observations)
            )
            if len(visible_relavant_agents.difference(new_visible_relavant_agents)) > 0:
                new_sim_runner = copy.deepcopy(runner)
                new_sim_runner.scenario._modifier = modifier_string + str(modifier_number)
                modified_simulation_runners.append(new_sim_runner)
                points_injected_at = points_injected_at.union(point)
                modifier_number += 1
                
            # remove injected vehicle from original scenario
            self.remove_candidate(candidate, runner)
                
        return modified_simulation_runners
    
    def generate_injection_candidate(self, point: Point, runner: SimulationRunner, map_api: AbstractMap, traffic_light_status: Dict[TrafficLightStatusType, List[str]]) -> Tuple[Agent, StateSE2]:
        """_summary_
        :param point: _description_
        :param runner: _description_
        :param map_api: _description_
        :param traffic_light_status: _description_
        :return: _description_
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
        
        distance = [
            segment.baseline_path.get_nearest_pose_from_position(point2d).distance_to(point2d)
            for segment in segments
        ]
        closest_segment = segments[np.argmin(np.abs(distance))]
        heading = closest_segment.baseline_path.get_nearest_pose_from_position(point2d).heading
        
        speed_setter_catch = closest_segment.polygon
        for segment in (closest_segment.incoming_edges + closest_segment.outgoing_edges):
            speed_setter_catch = speed_setter_catch.union(segment.polygon)
        
        moving_agents = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(AGENT_TYPES)
        
        agents_within_range = []
        vehicles_within_range = []
        for agent in moving_agents:
            center_poly = Point(agent.center.x, agent.center.y)
            if agent.center.distance_to(point2d) < self.LEAD_FOLLOW_AGENT_RANGE and speed_setter_catch.contains(center_poly):
                agents_within_range.append(agent)
                if agent.tracked_object_type == TrackedObjectType.VEHICLE:
                    vehicles_within_range.append(agent)
        sorted_agents = sorted(agents_within_range, key=lambda x: (x.center.distance_to(point2d)), reverse=False) #sorts closest to furthest
        sorted_vehicles = sorted(vehicles_within_range, key=lambda x: (x.center.distance_to(point2d)), reverse=False) #sorts closest to furthest

        if len(sorted_agents) > 0:
            velocity = sorted_agents[0].velocity
        elif segment.outgoing_edges[0].id in traffic_light_status[TrafficLightStatus.RED]:
            progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(agent.center)
            distance_to_end = closest_segment.baseline_path.length - progress
            speed_limit_velocity = StateVector2D(segment.speed_limit_mps * math.cos(heading), segment.speed_limit_mps * math.sin(heading))
            velocity = min(speed_limit_velocity, distance_to_end / self.TIME_TO_END_OF_SEGMENT)
        else:
            velocity = StateVector2D(segment.speed_limit_mps * math.cos(heading), segment.speed_limit_mps * math.sin(heading))

        
        if len(sorted_vehicles) > 0:
            current_iter = 0
            goal = runner.simulation.observations._get_historical_agent_goal(sorted_vehicles[0], current_iter)
        else:
            goal_segment = segment #if we cant find a vehicle to copy the goal of, we make our own randomly
            for _ in range(self.EXTENSION_STEPS):
                out = segment.outgoing_edges
                if len(out) > 0:
                    goal_segment = out[0]
            
            first, last = goal_segment.baseline_path.linestring.boundary
            goal_point = Point2D(last.x, last.y)
            goal = goal_segment.baseline_path.get_nearest_pose_from_position(goal_point)
        
        #craft agent to inject
        agent_to_insert = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=OrientedBox(StateSE2(point2d.x, point2d.y, heading), 5, 2, 2),
            velocity=StateVector2D(velocity[0], velocity[1]),
            metadata=SceneObjectMetadata(scenario.get_time_point(0).time_us, "inserted", -2, "inserted"),
            angular_velocity=0.0,
        )

        return agent_to_insert, goal
    
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
    
    def inject_candidate(self, candidate: Agent, goal: StateSE2, runner: SimulationRunner, time_point: TimePoint) -> None:
        """_summary_
        :param candidate: _description_
        :param goal: _description_
        :param runner: _description_
        :param time_point: _description_
        """
        runner.simulation._observations.add_agent_to_scene(candidate, goal, time_point)
    
    def remove_candidate(self, candidate: Agent, runner: SimulationRunner) -> None:
        runner.simulation._observations._agents.pop(candidate.metadata.track_token)
    
    def _filter_to_valid_spawn_points(self,
    potential_spawn_points: np.ndarray,
    no_spawn_polys: MultiPolygon,
    map_polys: MultiPolygon,
    ) -> List[Point]:
        """Helper to remove points from potential_spawn_points if they are too close to other vehicles or edges of the road."""
        # expand geometries to account for vehicle overlaps and drivable areas
        injection_shape = (5,2)#####TODO pull out these magic numbers
        half_shape_diag = ((injection_shape[0] ** 2 + injection_shape[1] ** 2) ** (1/2)) / 2  # pythag to fetch longest dist from center point, used to make area for headings
        potential_spawns = MultiPoint(potential_spawn_points)
        potential_spawn_areas = potential_spawns.buffer(half_shape_diag)
        too_close_to_other_vehicles_poly = no_spawn_polys.buffer(self.MINIMUM_SPAWNING_DISTANCE)
        
        # filter to valid areas
        drivable_spawn_areas = potential_spawn_areas.intersection(map_polys)  # on the road
        valid_spawn_areas = drivable_spawn_areas.difference(too_close_to_other_vehicles_poly)  # not too close to others
        
        # throw out points which cannot spawn vehicles (no matter the heading)
        width = injection_shape[1] # the smallest radius from center that must be valid for vehicle to the space occupy is width/2
        valid_spawn_points = [p for p in potential_spawn_points if Point(p).buffer(width/2).covered_by(valid_spawn_areas)]
        #we still need to check for intersections when attempting injections, but this should reduce the number of points we need to check
        return valid_spawn_points
    
    def sample_around_points(self, points: np.ndarray, stdev: float) -> np.ndarray:
        """Helper to draw other potential spawn points from gaussian"""
        m, n = points.shape
        noise = np.random.normal(np.zeros((1, n)), stdev * np.ones((1, n)), (m, n))
        return points + noise
    
    def find_relavant_agents(self, observations: MLPlannerAgents, scenario: AbstractScenario):
        relevant_agent_tokens = []
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE]
        for agent in scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(object_types):
            # Sets agent goal to be it's last known point in the simulation. 
            current_iter = 0
            goal = observations._get_historical_agent_goal(agent, current_iter)

            if goal:
                if observations._is_parked_vehicle(agent, goal, scenario.map_api):
                    continue

                route_plan = observations._get_roadblock_path(agent, goal)

                if not observations._irrelevant_to_ego(route_plan, scenario):
                    relevant_agent_tokens.append(agent.track_token)
        return relevant_agent_tokens
    
    def generate_full_fov_polygon(self, ego_object: AgentState, agents: List[AgentState], relavant_agent_tokens: List[str]) -> Polygon:
        """Generates the full fov polygon for all relavant agent taking occlusions into account
        :param ego_object: _description_
        :param agents: _description_
        :return: _description_
        """
        sorted_agents = sorted(agents, key=lambda x: (x.center.distance_to(ego_object.center)), reverse=True) #sorts farthest to closest
        full_fov_poly = Polygon()
        max_range = sorted_agents[0].center.distance_to(ego_object.center)
        for agent in sorted_agents:
            fov, fov_through, target_poly = self.generate_fov_polygon(ego_object, agent, max_range=max_range)
            if agent.tracked_object_type == TrackedObjectType.VEHICLE: #if its a vehicle, it can cause occlusions, so we subtract the large through polygon to remove everything behind the target before adding the smaller fov polygon back in
                full_fov_poly = full_fov_poly.difference(fov_through)
            if agent.track_token in relavant_agent_tokens:
                full_fov_poly = full_fov_poly.union_all(fov)
        return full_fov_poly
    
    def generate_fov_polygon(self, observer: AgentState, target: AgentState, max_range: int = 1000) -> Tuple[Polygon, Polygon, Polygon]:
        """Generates a polygon that represents the field of view from the observer to the target, as well as the field of view from the observer through the target to a given range.
        :param observer: observation agent
        :param target: target agent
        :param max_range: max range of extension, defaults to 1000
        :return: the fov polygon, the fov polygon that extends through the target and the polygon representing the target itself
        """
        observer_origin = (observer.center.x, observer.center.y)
        corners_list = target.box.all_corners() #Return 4 corners of oriented box (FL, RL, RR, FR) Point2D
        corners = []
        corners_centered = []
        for corner in corners_list:
            corners.append((corner.x, corner.y))
            #corners centered on the observer make later calcualtions easier as long as we remember to transform our results back afterwards
            corners_centered.append((corner.x - observer.center.x, corner.y - observer.center.y)) #we shift the corners and move them to a different data structure we can play with
        horizon_points = []
        range_to_ego = target.center.distance_to(observer.center)
        fov_through_range = max(max_range, range_to_ego)
        for corner in corners_centered:
            fov_through_range_multiplier = fov_through_range / (corner[0]**2 + corner[1]**2)**0.5
            horizon_points.append((corner[0]*fov_through_range_multiplier + observer.center.x, corner[1]*fov_through_range_multiplier + observer.center.y))
        target_poly = Polygon(corners) #the order they get returned in means we dont need to do a convex hull (FL, RL, RR, FR)
        fov_poly = Polygon(corners.append(observer_origin)).convex_hull.difference(target_poly)
        fov_through_poly = Polygon(horizon_points.append(observer_origin)).convex_hull
        return fov_poly, fov_through_poly, target_poly
    
    def discretize_centerline_segments(self,
    centerlines: Union[MultiLineString, LineString],
    ) -> np.ndarray:
        """Helper function to generate discretized centerline points.
        :param centerlines: _description_
        :return: _description_
        """
        # discretize and return coordinates for potential occluding spawns
        #centerlines = centerlines.simplify(self.TOLERANCE, preserve_topology=True).geoms  #simplifiy to reduce density of points
        ### this adds segments, but wont remove any if there are already too many
        centerlines_segments = centerlines.segmentize(self.DISTANCE_BETWEEN_DISCRETIZED_POINTS) # discretize and break into individual geometries
        return np.asarray(
        [
            coord
            for sequence in centerlines_segments
            for coord in sequence.coords
        ]
    )
    
    def get_map_geometry(self, ego_object: AgentState, map_api: AbstractMap, traffic_light_status: Dict[TrafficLightStatusType, List[str]]) -> Tuple[MultiLineString, MultiPolygon]:
        """Helper function to get map geometry from a map.
        :param map_api: _description_
        :return: _description_
        """
        # get centerlines
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        map_object_dict = map_api.get_proximal_map_objects(ego_object.center.point, self.MAP_RADIUS, layers)
        centerlines = []
        map_polys = []
        for layer in layers:
            for obj in map_object_dict[layer]:
                if obj.id not in traffic_light_status[TrafficLightStatus.RED]:
                    centerlines.append(obj.baseline_path.linestring)
                    map_polys.append(obj.polygon)
        centerlines = union_all(centerlines)
        map_polys = union_all(map_polys)
        return centerlines, map_polys