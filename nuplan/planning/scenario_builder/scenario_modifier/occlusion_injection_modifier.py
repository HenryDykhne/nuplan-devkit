from __future__ import annotations
from collections import defaultdict, deque
import itertools
from typing import Dict, List, Set, Tuple, Union

import copy

import math
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, MultiLineString, LineString, MultiPoint
from shapely import union_all, affinity

from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import Lane, LaneConnector, LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.database.nuplan_db_orm.traffic_light_status import TrafficLightStatus
from nuplan.database.utils.measure import angle_diff

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractModification, AbstractScenarioModifier
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.ml_planner_agents import MLPlannerAgents

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint

from nuplan.planning.simulation.occlusion.wedge_occlusion_manager import WedgeOcclusionManager

from nuplan.common.maps.nuplan_map.utils import cut_piece


from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
class OcclusionInjectionModifier(AbstractScenarioModifier):
    DISTANCE_BETWEEN_DISCRETIZED_POINTS = 0.5
    TOLERANCE = 0.2
    MAP_RADIUS = 200.0
    SAMPLE_SPAWN_POINT_STDEV = 0.20
    ADD_NOISE = True
    MINIMUM_SPAWNING_DISTANCE = 1.0 # REPLACE THIS CONSTANT with a function of the speed of the agent you are checking
    MIN_DISTANCE_BETWEEN_INJECTIONS = 1.0
    LEAD_FOLLOW_AGENT_RANGE = 20.0 #agents we copy the speed and goal from that we are leading or following
    SIDE_AGENT_RANGE = 10.0 #agents we check exist in case we cant find a goal from leading and following agents, just to make sure there is something that could be worth occluding in a lane beside us
    EXTENSION_STEPS = 3
    TIME_TO_END_OF_SEGMENT = 2.0 #seconds
    RELAVANT_PLAN_DEPTH = 3
    UPPER_CUT = 0.95
    LOWER_CUT = 0.30
    DEFAULT_SPEED_LIMIT_MPS = 15.0 #equates to roughly 54kph
    MIN_ALLOWED_TIME_TO_COLLISION = 2.0
    
    def __init__(self):
        super().__init__() #maybe we will need this later
        np.random.seed(0)
        
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        """We convert one abstract scenario into many abstract scenarios by modifying the scenario in some way.
        :param runner: a scenario
        :return: we return a list of runners that are modified versions of the input scenario
        """
        scenario = runner.scenario
        traffic_light_status = self.get_traffic_light_status_at_iteration(0, scenario)
        relavant_agent_tokens = self.find_relavant_agents(runner.simulation._observations, scenario, traffic_light_status)
        #print('Number of relavant agent tokens', len(relavant_agent_tokens))
        # here we generate the field of view polygons for each relavant agent, taking into account potential occlusions even from nonrelavant agents
        ego_object = scenario.get_ego_state_at_iteration(0)
        ego_agent = ego_object.agent
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE]
        agents = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(object_types)
        
        if len(agents) == 0:
            print('No initial tracked agents in scenario', scenario.token)
            return []

        full_fov_poly = self.generate_full_fov_polygon(ego_agent, agents, relavant_agent_tokens)

        if full_fov_poly.area == 0:
            return []
        
        #we get a superset of lane ids that we deem valid connectors. to insert agents onto. we want these agents to be leading or adjacent to at least one of the relavant agents or ego to maximize the length of the occlusion
        _, ego_lane_level_route_plan = runner.simulation._observations._get_roadblock_path(
            scenario.get_ego_state_at_iteration(0).agent,
            scenario.get_expert_goal_state()
        )
        lane_objects_to_prune_by = []
        for lane_object in ego_lane_level_route_plan:
            lane_objects_to_prune_by.append(lane_object)
            
        relavant_agents = []
        for agent in agents:
            if agent.track_token in relavant_agent_tokens:
                relavant_agents.append(agent)

        for relavant_agent in relavant_agents:
            current_iter = 0
            goal = runner.simulation._observations._get_historical_agent_goal(relavant_agent, current_iter)
            _, relavant_agent_route_plan = runner.simulation._observations._get_roadblock_path(relavant_agent, goal)
            for lane_object in relavant_agent_route_plan:
                lane_objects_to_prune_by.append(lane_object)
        
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
        avoid_geoms = MultiPolygon(avoid_geoms)
        candidate_occluding_spawn_points = self._filter_to_valid_spawn_points(
            potential_spawn_points,
            avoid_geoms,
            map_polys
        )
        
        modified_simulation_runners = []
        modifier_string = "occlusion_injection_"
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
            candidate, goal = self.generate_injection_candidate(point, runner, scenario.map_api, traffic_light_status)
            
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
                modification = OcclusionInjectionModification(candidate, goal, iteration.time_point, modifier_string + str(modifier_number))
                modification.modify(new_sim_runner.simulation)
                new_sim_runner.simulation.modification = modification

                modified_simulation_runners.append(new_sim_runner)
                points_injected_at = points_injected_at.union(point)
                modifier_number += 1

            
                
        return modified_simulation_runners
    
    def how_does_ego_cross_intersection(self, runner: SimulationRunner ) -> LaneConnector:
        """This returns the lane connector that the ego vehicle uses to cross an intersection. If the ego vehicle is already inside the intersection, it returns None.
        :param runner: runner that containst the scenario to use to determine how the ego crosses the intersection
        :return: LaneConnector that the ego uses to cross the intersection
        """
        scenario = runner.scenario
        ego_object = scenario.get_ego_state_at_iteration(0)
        ego_agent = ego_object.agent
        _, ego_lane_level_route_plan = runner.simulation._observations._get_roadblock_path(
                    ego_agent,
                    scenario.get_expert_goal_state()
                )
        if ego_lane_level_route_plan is None:
            return None
        
        depth = 0
        suspected_connector = None
        for lane_object in ego_lane_level_route_plan:
            if depth > 3:
                return None
            if isinstance(lane_object, LaneConnector):
                suspected_connector = lane_object
                break
            depth += 1
        
        if suspected_connector is None:
            return None
        
        historical_egostate_generator = scenario.get_expert_ego_trajectory()
        for ego_state in historical_egostate_generator:
            current_lane_objects = scenario.map_api.get_all_map_objects(ego_state.center.point, SemanticMapLayer.LANE_CONNECTOR)
            current_lane_object_ids = [lane_object.id for lane_object in current_lane_objects]
            if suspected_connector.id in current_lane_object_ids:
                return suspected_connector
        return None
    
    def generate_injection_candidate(self, point: Point, runner: SimulationRunner, map_api: AbstractMap, traffic_light_status: Dict[TrafficLightStatusType, List[str]], optional_extra_agents: List[Agent] = []) -> Tuple[Agent, StateSE2]:
        """We generate the agent state and goal pair for the newly injected agent
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
        
        distance = [
            segment.baseline_path.get_nearest_pose_from_position(point2d).distance_to(point2d)
            for segment in segments
        ]
        closest_segment = segments[np.argmin(np.abs(distance))]
        heading = closest_segment.baseline_path.get_nearest_pose_from_position(point2d).heading
        
        ego_agent = scenario.initial_ego_state.agent
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
        
        moving_agents = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(AGENT_TYPES)
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
            goal = runner.simulation._observations._get_historical_agent_goal(sorted_vehicles[0], current_iter)
            goal = copy.deepcopy(goal)

        if goal is None and len(agents_within_range_and_roadblock) > 0:
            # print('how often does this happen?')
            goal_segment = closest_segment #if we cant find a vehicle to copy the goal of, we make our own and let the automated path extension handle the rest
            *_, last = goal_segment.baseline_path.linestring.coords #we use the last point in a linestring far away to start setting our goal
            last = Point(last)
            goal_point = Point2D(last.x, last.y)
            goal = goal_segment.baseline_path.get_nearest_pose_from_position(goal_point)
            
        if goal is None:
            return None, None #if we cant find a vehicle to copy the goal of, then we arent following or leading another vehicle, so this is likely not a great inection spot
        
        #craft agent to inject
        agent_to_insert = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=OrientedBox(StateSE2(point2d.x, point2d.y, heading), 5, 2, 2),
            velocity=StateVector2D(velocity.x, velocity.y),
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
        runner.simulation._observations.add_agent_to_scene(candidate, goal, time_point, runner.simulation)
    
    def remove_candidate(self, candidate: Agent, runner: SimulationRunner) -> None:
        runner.simulation._observations.remove_agent_from_scene(candidate, runner.simulation)
    
    def _filter_to_valid_spawn_points(self,
        potential_spawn_points: np.ndarray,
        no_spawn_polys: MultiPolygon,
        map_polys: MultiPolygon,
    ) -> List[Point]:
        """Helper to remove points from potential_spawn_points if they are too close to other vehicles or edges of the road.
        :param potential_spawn_points: what it says on the tin
        :param no_spawn_polys: polys we should not spawn inside of like other agents
        :param map_polys: polygon describing the lanes we can drive on
        :return: list of valid spawn points
        """
        
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
        valid_spawn_points = [Point(p) for p in potential_spawn_points if Point(p).buffer(width/2).covered_by(valid_spawn_areas)]
        #we still need to check for intersections when attempting injections, but this should reduce the number of points we need to check
        return valid_spawn_points
    
    def sample_around_points(self, points: np.ndarray, stdev: float) -> np.ndarray:
        """Helper to draw other potential spawn points from gaussian"""
        m, n = points.shape
        noise = np.random.normal(np.zeros((1, n)), stdev * np.ones((1, n)), (m, n))
        return points + noise
    
    def find_relavant_agents(self, observations: MLPlannerAgents, scenario: AbstractScenario, traffic_light_status: Dict[TrafficLightStatusType, List[str]]) -> List[str]:
        """Returns a list of tokens of agents that might be worth occluding
        :param observations: observations, specifically the MLP planner agents. no point in using anything else since no other observations properly react to occlusions
        :param scenario: our scenario
        :param traffic_light_status: traffic light status at the time of injection
        :return: list of relavant agent tokens
        """
        relevant_agent_tokens = []
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BICYCLE]
        _, ego_lane_level_route_plan = observations._get_roadblock_path(
                    scenario.get_ego_state_at_iteration(0).agent,
                    scenario.get_expert_goal_state()
                )
        
        for agent in scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(object_types):
            # Sets agent goal to be it's last known point in the simulation. 
            current_iter = 0
            goal = observations._get_historical_agent_goal(agent, current_iter)

            if goal:
                if observations._is_parked_vehicle(agent, goal, scenario.map_api):
                    continue

                _, lane_level_route_plan = observations._get_roadblock_path(agent, goal)
                if lane_level_route_plan is None:
                    continue
                
                heading_diff = angle_diff(agent.center.heading, scenario.get_ego_state_at_iteration(0).agent.center.heading, math.pi*2)
                check_range = observations._optimization_cfg.mixed_agent_heading_check_range if observations._optimization_cfg.mixed_agent_heading_check else 0.2
                if abs(heading_diff) <= check_range:#if the agent is aligned with us, its likely not relavant
                    continue
                
                plan_shape = self.get_relavant_plan_shape(lane_level_route_plan, self.RELAVANT_PLAN_DEPTH, self.LOWER_CUT, self.UPPER_CUT, traffic_light_status)

                ego_plan_shape = self.get_relavant_plan_shape(ego_lane_level_route_plan, self.RELAVANT_PLAN_DEPTH, self.LOWER_CUT, self.UPPER_CUT, traffic_light_status)
                    
                if plan_shape.intersects(ego_plan_shape):
                    relevant_agent_tokens.append(agent.track_token)
        return relevant_agent_tokens
    
    def get_relavant_plan_shape(self, lane_level_route_plan: List[LaneGraphEdgeMapObject],
                                relavant_plan_depth: int,
                                lower_cut: float,
                                upper_cut: float,
                                traffic_light_status: Dict[TrafficLightStatusType, List[str]]
                                ) -> MultiLineString:
        """returns multilinestring that represents the relavant portions of the future plan (where lane connectors are since thats where agents cross each other)
        :param lane_level_route_plan: list of lane graph objects that make up the plan
        :param relavant_plan_depth: how deep to search. 0 imples to only look at the first lane connector
        :param lower_cut: how much to remove from the start of the connector
        :param upper_cut: how much to remove from the end of the connector (we do this to try and avoid intersections where lane connectors merge together)
        :param traffic_light_status: traffic light status at the time of injection
        :return: the multilinestring that represents the relavant portions of the future plan
        """
        plan_shape = MultiLineString()
        for i, lane_object in enumerate(lane_level_route_plan):
            if isinstance(lane_object, LaneConnector) and lane_object.id not in traffic_light_status[TrafficLightStatusType.RED]:
                linestring = cut_piece(lane_object.baseline_path.linestring, lower_cut, upper_cut)# cuts off first 30% and last 5% of the line
                plan_shape = plan_shape.union(linestring)
            if i == relavant_plan_depth:
                break
        return plan_shape
    
    def generate_full_fov_polygon(self, ego_object: AgentState, agents: List[AgentState], relavant_agent_tokens: List[str]) -> Polygon:
        """Generates the full fov polygon for all relavant agent taking occlusions into account
        :param ego_object: ego
        :param agents: all agents in the scene
        :param relavant_agent_tokens: tokens of relavant agents
        :return: polygon representing the full fov
        """
        sorted_agents = sorted(agents, key=lambda x: (x.center.distance_to(ego_object.center)), reverse=True) #sorts farthest to closest
        full_fov_poly = Polygon()
        max_range = sorted_agents[0].center.distance_to(ego_object.center)
        for agent in sorted_agents:
            fov, fov_through, target_poly = self.generate_fov_polygon(ego_object, agent, max_range=max_range)
            if agent.tracked_object_type == TrackedObjectType.VEHICLE: #if its a vehicle, it can cause occlusions, so we subtract the large through polygon to remove everything behind the target before adding the smaller fov polygon back in
                full_fov_poly = full_fov_poly.difference(fov_through)
            if agent.track_token in relavant_agent_tokens:
                full_fov_poly = full_fov_poly.union(fov)
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
        fov_poly = Polygon(corners + [observer_origin]).convex_hull.difference(target_poly)
        fov_through_poly = Polygon(horizon_points + [observer_origin]).convex_hull
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
        discretized = centerlines.segmentize(self.DISTANCE_BETWEEN_DISCRETIZED_POINTS)
        if not isinstance(discretized, MultiLineString): #makes sure its a multilinestring
            discretized = MultiLineString([discretized])
        centerlines_segments = discretized.geoms # discretize and break into individual geometries
        return np.asarray(
        [
            coord
            for sequence in centerlines_segments
            for coord in sequence.coords
        ]
    )
    
    def get_map_geometry(self, ego_object: AgentState, map_api: AbstractMap, traffic_light_status: Dict[TrafficLightStatusType, List[str]], lane_objects_to_prune_by: List[LaneGraphEdgeMapObject] = None) -> Tuple[MultiLineString, MultiPolygon]:
        """Helper function to get map geometry from a map.
        :param ego_object: ego object to center map on
        :param map_api: what it says on the tin
        :param traffic_light_status: traffic light status at the time of injection
        :param ids_to_prune_by: list of laneobjects that must be matched to if we want to keep a particular lane centerline. lane connectors must match exactly and lanes must have their parent roadblock match to allow for occlusions by cars in adjacent lanes. if it is None, we do not prune
        :return: A multilinestring of all the centerlines, and a multipolygon of all the map polygons
        """
        
        # get centerlines
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        map_object_dict = map_api.get_proximal_map_objects(ego_object.center.point, self.MAP_RADIUS, layers)
        centerlines = []
        map_polys = []
        for layer in layers:
            for obj in map_object_dict[layer]:
                if lane_objects_to_prune_by is not None:
                    if isinstance(obj, LaneConnector) and (obj.id not in [lane_object.id for lane_object in lane_objects_to_prune_by]):
                        continue
                    elif (obj.parent.id not in [lane_object.parent.id for lane_object in lane_objects_to_prune_by]): #parent block must match
                        continue
                if (obj.id not in traffic_light_status[TrafficLightStatusType.RED]):
                    centerlines.append(obj.baseline_path.linestring)
                    map_polys.append(obj.polygon)
        centerlines = union_all(centerlines)
        map_polys = union_all(map_polys)
        return centerlines, map_polys
    
    def calculate_rough_min_time_to_collision(self, ego_agent: Agent, other_agents: List[Agent], interval: float = 0.1, horizon: float = 3) -> float:
        """Helper function to calculate the rough minimum time to collision between ego and other agents.
        :param ego_agent: ego agent
        :param other_agents: all other agents
        :param interval: interval between checks. ideally, should be 0.1s which would imply 10Hz
        :param horizon: maximum time to check to in seconds. 
        :return: rough minimum time to collision, or None if above horizon
        """
        agents = [ego_agent] + other_agents
        steps = int(horizon / interval)
        # print(len(agents))
        # print(steps)
        # for agent in agents:
        #     print(agent.center.x, agent.center.y, agent.velocity.x, agent.velocity.y)
        #     print(agent.center.x, agent.center.y, agent.velocity.magnitude() * math.cos(agent.center.heading), agent.velocity.magnitude() * math.sin(agent.center.heading))

        for step in range(1, steps):
            for agent1, agent2 in itertools.combinations(agents, 2):
                time = step * interval
                curr_poly1 = affinity.translate(agent1.box.geometry, xoff=agent1.velocity.magnitude() * math.cos(agent1.center.heading) * time, yoff=agent1.velocity.magnitude() * math.sin(agent1.center.heading) * time)
                curr_poly2 = affinity.translate(agent2.box.geometry, xoff=agent2.velocity.magnitude() * math.cos(agent2.center.heading) * time, yoff=agent2.velocity.magnitude() * math.sin(agent2.center.heading) * time)
                if curr_poly1.intersects(curr_poly2):
                    return time
        return None


class OcclusionInjectionModification(AbstractModification):
    def __init__(self, inserted_agent: Agent, goal_state: StateSE2, time_point: TimePoint, modifier_string: str):
        super().__init__(modifier_string) #maybe we will need this later
        self.inserted_agent = inserted_agent
        self.goal_state = goal_state
        self.time_point = time_point

    def modify(self, simulation: Simulation) -> None:
        simulation._observations.add_agent_to_scene(
            self.inserted_agent, self.goal_state, self.time_point, simulation
        )
        simulation.scenario._modifier = self.modifier_string