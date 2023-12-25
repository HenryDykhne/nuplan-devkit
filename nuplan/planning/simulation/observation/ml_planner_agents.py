from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occlusion.wedge_occlusion_manager import WedgeOcclusionManager
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner

from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.training.modeling.models.pdm_offset_model import PDMOffsetModel
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_hybrid_planner import PDMHybridPlanner

from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy

from nuplan.common.maps.maps_datatypes import SemanticMapLayer

OPEN_LOOP_DETECTION_TYPES = [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE, \
                             TrackedObjectType.CZONE_SIGN, TrackedObjectType.BARRIER, \
                             TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.GENERIC_OBJECT]

IDM_AGENT_CONFIG = {  
    "target_velocity": 10,             # Desired velocity in free traffic [m/s]
    "min_gap_to_lead_agent": 1.0,      # Minimum relative distance to lead vehicle [m]
    "headway_time": 1.5,               # Desired time headway. The minimum possible time to the vehicle in front [s]
    "accel_max": 1.0,                  # Maximum acceleration [m/s^2]
    "decel_max": 3.0,                  # Maximum deceleration (positive value) [m/s^2]
    "planned_trajectory_samples": 16,  # Number of trajectory samples to generate
    "planned_trajectory_sample_interval": 0.5,  # The sampling time interval between samples [s]
    "occupancy_map_radius": 40,        # The range around the ego to add objects to be considered [m]
}


PDM_CLOSED_AGENT_CONFIG = {  
    "trajectory_sampling": TrajectorySampling(num_poses=80, interval_length= 0.1),
    "proposal_sampling": TrajectorySampling(num_poses=40, interval_length= 0.1),
    "idm_policies": BatchIDMPolicy(speed_limit_fraction= [0.2,0.4,0.6,0.8,1.0], 
                                    fallback_target_velocity= 15.0, 
                                    min_gap_to_lead_agent= 1.0,
                                    headway_time= 1.5,
                                    accel_max= 1.5,
                                    decel_max= 3.0),
    "lateral_offsets": [-1.0, 1.0], 
    "map_radius": 50,
}

PDM_HYBRID_AGENT_CONFIG = {  
    "model":PDMOffsetModel(trajectory_sampling=TrajectorySampling(num_poses=16, interval_length=0.5), 
                        history_sampling=TrajectorySampling(num_poses=10, interval_length=0.2),
                        planner=None,
                        centerline_samples=120,
                        centerline_interval=1.0,
                        hidden_dim=512),
    "correction_horizon":2.0,
}


class MLPlannerAgents(AbstractObservation):
    """
    Simulate agents based on an ML model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario, occlusions: bool, planner_type: str, pdm_hybrid_ckpt: str) -> None:
        """
        Initializes the MLPlannerAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        self.current_iteration = 0
        self.model = model
        self.planner_type = planner_type
        self.pdm_hybrid_ckpt = pdm_hybrid_ckpt
        self._scenario = scenario
        self._occlusions = occlusions
        self._ego_state_history: Dict = {}
        self._agents: Dict = None    
        self._trajectory_cache: Dict = {}
        self._inference_frequency: float = 0.2
        self._full_inference_distance: float = 30

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0
        self._agents = None
        self._trajectory_cache = {}
        self._ego_state_history = {}
        
    def _get_agents(self):
        """
        Gets dict of tracked agents, or lazily creates them it 
        from vehicles at simulation start if it does not exist.
        """

        if not self._agents:
            self._agents = {}
            for agent in self._scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):

                
                # TODO: Support ego controllers - right now just doing perfect tracking.
                #       Revist whether there is a better way of translating agent states to ego states. 
                #       Revist whether there is a better way of setting agent goals.
                #       Filter out impossible/off-road initial detections.

                # Sets agent goal to be it's last known point in the simulation. This results in some strange driving behaviour
                # if the agent disappears early in a scene.
                goal = self._get_historical_agent_goal(agent, self.current_iteration)
                if goal:

                    route_plan = self._get_roadblock_path(agent, goal)

                    if route_plan:
                        self._agents[agent.metadata.track_token] = self._build_agent_record(agent, self._scenario.start_time)

                        # Initialize planner.
                        planner_init = PlannerInitialization(
                            route_roadblock_ids=route_plan,
                            mission_goal=goal,
                            map_api=self._scenario.map_api,
                        )

                        self._agents[agent.metadata.track_token]['planner'].initialize(planner_init)

        return self._agents

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass
            
    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        agents = [self._build_agent_from_ego_state(v['ego_state'], v['metadata']) for v in self._get_agents().values()]
        open_loop_detections = self._get_open_loop_track_objects(self.current_iteration)
        open_loop_detections.extend(agents)
        return DetectionsTracks(tracked_objects=TrackedObjects(open_loop_detections))
    
    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
        #self._add_newly_detected_agents(next_iteration) #- Adds new agents. This causes some weird behaviour, so commented out for now.
        self.propagate_agents(iteration, next_iteration, history)

    def _get_open_loop_track_objects(self, iteration: int) -> List[TrackedObject]:
        """
        Get open-loop tracked objects from scenario.
        :param iteration: The simulation iteration.
        :return: A list of TrackedObjects.
        """

        detections = self._scenario.get_tracked_objects_at_iteration(iteration) 
        return detections.tracked_objects.get_tracked_objects_of_types(OPEN_LOOP_DETECTION_TYPES) 

    def propagate_agents(self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer):
        """
        Propagates agents into next timestep by constructing input for each agent planner, then interpolating new agent
        states from the predicted trajectory for each from their respective planners. Caches computed agent states.
        """

        traffic_light_data = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))

        # TODO: Find way to parallelize.
        # TODO: Propagate non-ego and lower frequency to improve performance.
        for agent_token, agent_data in self._agents.items():
            if agent_token in self._trajectory_cache and \
                (next_iteration.time_s - self._trajectory_cache[agent_token][0]) < self._inference_frequency and \
            (((agent_data['ego_state'].center.x - history.current_state[0].center.x) ** 2 + \
                (agent_data['ego_state'].center.y - history.current_state[0].center.y) ** 2) ** 0.5) >= self._full_inference_distance:
                     trajectory = self._trajectory_cache[agent_token][1]
            else:
                history_input = self._build_history_input(agent_token, agent_data['ego_state'], history)

                if agent_data['occlusion'] is not None:
                    history_input = agent_data['occlusion'].occlude_input(history_input)

                planner_input = PlannerInput(iteration=iteration, history=history_input, traffic_light_data=traffic_light_data)                    
                trajectory = agent_data['planner'].compute_trajectory(planner_input)
                self._trajectory_cache[agent_token] = (next_iteration.time_point.time_s, trajectory)

            agent_data['ego_state'] = trajectory.get_state_at_time(next_iteration.time_point)
            self._ego_state_history[agent_token][next_iteration.time_point] = agent_data['ego_state']
            
    def _build_ego_state_from_agent(self, agent: Agent, time_point: TimePoint) -> EgoState:
        """
        Builds ego state from corresponding agent state. Since this process is imperfect, it uses cached ego states from the propagation
        so casting is only required for the beginning agent states for which we have no propagation information.
        """

        if agent.metadata.track_token in self._ego_state_history:
            if time_point in self._ego_state_history[agent.metadata.track_token]:
                return self._ego_state_history[agent.metadata.track_token][time_point]
        else:
            self._ego_state_history[agent.metadata.track_token] = {}

        # Most of this is just eyeballed, so there may be a more principled way of setting these values.
        output = EgoState.build_from_center(
            center=agent.center,
            center_velocity_2d=agent.velocity if self.planner_type != "idm" else StateVector2D(agent.velocity.magnitude(), 0),
            center_acceleration_2d=StateVector2D(0, 0),
            tire_steering_angle=0,
            time_point=time_point,
            vehicle_parameters=VehicleParameters(
                        vehicle_name=agent.track_token,
                        vehicle_type="gen1",
                        width=agent.box.width,
                        front_length=agent.box.length * 4 / 5,
                        rear_length=agent.box.length * 1 / 5,
                        wheel_base=agent.box.length * 3 / 5,
                        cog_position_from_rear_axle=agent.box.length * 1.5 / 5,
                        height=agent.box.height,
                    ),
        )

        self._ego_state_history[agent.metadata.track_token][time_point] = output
        return output
    
    def _build_agent_from_ego_state(self, ego_state: EgoState, scene_object_metadata: SceneObjectMetadata) -> Agent:
        """
        Builds agent state from corresponding ego state. Unlike the inverse this process is well-defined.
        """

        agent_state = Agent(
            metadata=scene_object_metadata,
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=ego_state.car_footprint.oriented_box,
            velocity=ego_state.dynamic_car_state.center_velocity_2d,
        )
        return agent_state
    
    def _build_history_input(self, agent_track_token: str, current_state: EgoState, history: SimulationHistoryBuffer) -> SimulationHistoryBuffer:
        """
        Builds the planner history input for a given agent. This requires us the interchange the ego states of the actual ego with the 
        constructed ego states of the agent of interest, and create observations corresponding to the ego in the observation history buffer.
        """
        ego_state_buffer = history.ego_state_buffer
        observation_buffer = history.observation_buffer

        new_observations = []
        faux_ego_obervations = []

        # Construct a timestep/track_token to observation loopup table for faster proessing
        track_token_agent_dict = {}
        for t, observation in enumerate(observation_buffer):
            track_token_agent_dict[t] = {}
            for agent in observation.tracked_objects.tracked_objects:
                track_token_agent_dict[t][agent.metadata.track_token] = agent

        # Loop through history buffer
        for t, (ego_state, observation) in enumerate(zip(ego_state_buffer, observation_buffer)):

            # Convert actual ego state into agent object
            ego_agent_object = self._build_agent_from_ego_state(ego_state, \
                                                                SceneObjectMetadata(token='ego', track_token="ego", track_id=-1, \
                                                                                    timestamp_us=ego_state.time_us))
            
            # Get agent object corresponding to agent from observation buffer. If one does not exist for current timestep, take from the future,
            # if one does not exist from the future, take the current state. This might occur at the first timestep for observations that have no
            # logged  history prior to simulation start, or observations inserted mid-simulation.
            matched_agent = None

            for i in range(t, len(observation_buffer)):
                if i in track_token_agent_dict and agent_track_token in track_token_agent_dict[i]:
                    matched_agent = track_token_agent_dict[i][agent_track_token]
                    break
                    
            # Convert agent state to a corresponding "ego state" object, or pull it from cache if already computed.
            if matched_agent is None:
                faux_ego_observation = deepcopy(current_state)
                faux_ego_observation._time_point = deepcopy(ego_state.time_point)
            else:
                faux_ego_observation = self._build_ego_state_from_agent(matched_agent, deepcopy(ego_state.time_point))


            # Rebuild timestep and buffer - creating a new observations object with old ego appended.
            tracks = [ag for ag in observation.tracked_objects.tracked_objects if ag.metadata.track_token != agent_track_token]
            tracks.append(ego_agent_object)

            new_observations.append(DetectionsTracks(tracked_objects=TrackedObjects(tracks)))
            faux_ego_obervations.append(faux_ego_observation)
    

        output_buffer = SimulationHistoryBuffer(deque(faux_ego_obervations), \
                            deque(new_observations), \
                                history.sample_interval)

        return output_buffer
    
    def add_agent_to_scene(self, agent: Agent, goal: StateSE2, timepoint_record: TimePoint):
        """
        Adds agent to the scene with a given goal during the simulation runtime.
        """
        # TODO: Inject IDM agents (and non-ML agents more broadly)

        self._agents[agent.metadata.track_token] = self._build_agent_record(agent, timepoint_record)

        route_plan = self._get_roadblock_path(agent, goal)

        if route_plan:
            self._agents[agent.metadata.track_token] = self._build_agent_record(agent, self._scenario.start_time)

            # Initialize planner.
            planner_init = PlannerInitialization(
                route_roadblock_ids=route_plan,
                mission_goal=goal,
                map_api=self._scenario.map_api,
            )

            self._agents[agent.metadata.track_token]['planner'].initialize(planner_init)

    def _build_agent_record(self, agent: Agent, timepoint_record: TimePoint):

        if self.planner_type == "ml":
            planner = MLPlanner(self.model)
        elif self.planner_type == "idm":
            planner = IDMPlanner(**IDM_AGENT_CONFIG)
        elif self.planner_type == "pdm_closed":
            planner = PDMClosedPlanner(**PDM_CLOSED_AGENT_CONFIG)
        elif self.planner_type == "pdm_hybrid":
            assert self.pdm_hybrid_ckpt, "Must provide checkpoint path for PDM hybrid planner."
            planner = PDMHybridPlanner(**PDM_CLOSED_AGENT_CONFIG, **PDM_HYBRID_AGENT_CONFIG, checkpoint_path=self.pdm_hybrid_ckpt)
        else:
            raise ValueError("Invalid planner type.")

        return {'ego_state': self._build_ego_state_from_agent(agent, timepoint_record), \
                'metadata': agent.metadata,
                'planner': planner,
                'occlusion': WedgeOcclusionManager(self._scenario) if self._occlusions else None}
    
    def _get_historical_agent_goal(self, agent: Agent, iteration_index: int):
        for frame in range(self._scenario.get_number_of_iterations()-1, iteration_index, -1):
            last_scenario_frame = self._scenario.get_tracked_objects_at_iteration(frame)
            for track in last_scenario_frame.tracked_objects.tracked_objects:
                if track.metadata.track_token == agent.metadata.track_token:
                    return track.center

        return None
    
    def _get_roadblock_path(self, agent: Agent, goal: StateSE2, max_depth: int = 10):

        start_edge, _ = self._get_target_segment(agent.center, self._scenario.map_api)
        end_edge, _ = self._get_target_segment(goal, self._scenario.map_api)

        if start_edge is None:
            return None
        
        if end_edge is not None:
            gs = BreadthFirstSearch(start_edge)
            route_plan, path_found = gs.search(end_edge, max_depth)
        else:
            route_plan = [start_edge]

        route_plan = self._extend_path(route_plan, max_depth)        
        route_plan = [edge.get_roadblock_id() for edge in route_plan]
        route_plan = list(dict.fromkeys(route_plan))
        
        if len(route_plan) == 1:
            route_plan = route_plan + route_plan

        return route_plan
    
    def _extend_path(self, route_plan: List[str], min_path_length: 10):
        """
        Extends a route plan to a given depth by continually going forward.
        """
        while len(route_plan) < min_path_length:
            outgoing_edges = route_plan[-1].outgoing_edges

            if not outgoing_edges:
                break

            curvatures = [abs(edge.baseline_path.get_curvature_at_arc_length(0.0)) for edge in outgoing_edges]
            idx = np.argmin(curvatures)
            route_plan.append(outgoing_edges[idx])

        return route_plan

    def _get_target_segment(
        self, target_state: StateSE2, map_api: AbstractMap
    ) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[float]]:
        """
        Gets the map object that the agent is on and the progress along the segment.
        :param agent: The agent of interested.
        :param map_api: An AbstractMap instance.
        :return: GraphEdgeMapObject and progress along the segment. If no map object is found then None.
        """
        if map_api.is_in_layer(target_state, SemanticMapLayer.LANE):
            layer = SemanticMapLayer.LANE
        elif map_api.is_in_layer(target_state, SemanticMapLayer.INTERSECTION):
            layer = SemanticMapLayer.LANE_CONNECTOR
        else:
            return None, None

        segments: List[LaneGraphEdgeMapObject] = map_api.get_all_map_objects(target_state, layer)
        if not segments:
            return None, None

        # Get segment with the closest heading to the agent
        heading_diff = [
            segment.baseline_path.get_nearest_pose_from_position(target_state).heading - target_state.heading
            for segment in segments
        ]
        closest_segment = segments[np.argmin(np.abs(heading_diff))]

        progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(target_state)
        return closest_segment, progress


class BreadthFirstSearch:
 

    def __init__(self, start_edge: LaneGraphEdgeMapObject):

        self._queue = deque([start_edge, None])
        self._parent: Dict[str, Optional[LaneGraphEdgeMapObject]] = dict()
        self._visited = set()

    def search(
        self, target_edge: LaneGraphEdgeMapObject, max_depth: int
    ) -> Tuple[List[LaneGraphEdgeMapObject], bool]:

        start_edge = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_edge: LaneGraphEdgeMapObject = start_edge
        end_depth: int = 1
        depth: int = 1

        self._parent[start_edge.id + f"_{depth}"] = None

        while self._queue:
            current_edge = self._queue.popleft()
            if current_edge is not None:
                self._visited.add(current_edge.id)

            # Early exit condition
            if self._check_end_condition(depth, max_depth):
                break

            # Depth tracking
            if current_edge is None:
                depth += 1
                self._queue.append(None)
                if self._queue[0] is None:
                    break
                continue

            # Goal condition
            if self._check_goal_condition(current_edge, target_edge):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                break

            # Populate queue
            for next_edge in current_edge.outgoing_edges:
                if next_edge.id not in self._visited:
                    self._queue.append(next_edge)
                    self._parent[next_edge.id + f"_{depth + 1}"] = current_edge
                    end_edge = next_edge
                    end_depth = depth + 1

        return self._construct_path(end_edge, end_depth), path_found

    @staticmethod
    def _check_end_condition(depth: int, target_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: True if:
            - The current depth exceeds the target depth.
        """
        return depth > target_depth

    @staticmethod
    def _check_goal_condition(
        current_edge: LaneGraphEdgeMapObject,
        target_edge: LaneGraphEdgeMapObject,
    ) -> bool:
        return current_edge.id == target_edge.id

    def _construct_path(self, end_edge: LaneGraphEdgeMapObject, depth: int) -> List[LaneGraphEdgeMapObject]:
        """
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of LaneGraphEdgeMapObject
        """
        path = [end_edge]
        while self._parent[end_edge.id + f"_{depth}"] is not None:
            path.append(self._parent[end_edge.id + f"_{depth}"])
            end_edge = self._parent[end_edge.id + f"_{depth}"]
            depth -= 1
        path.reverse()

        return path
