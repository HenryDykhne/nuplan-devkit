from collections import deque
from copy import deepcopy
import copy
import math
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from omegaconf import DictConfig

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.transform import translate_longitudinally
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject

from nuplan.database.utils.measure import angle_diff

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.occlusion_manager_builder import build_occlusion_manager
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.mlpa.max_depth_breadth_first_search import MaxDepthBreadthFirstSearch
from nuplan.planning.simulation.observation.mlpa.planner_config_constants import IDM_AGENT_CONFIG, OPEN_LOOP_DETECTION_TYPES, PDM_CLOSED_AGENT_CONFIG, \
                                                                                    PDM_HYBRID_AGENT_CONFIG, PDM_BATCH_IDM_CONFIG, \
                                                                                    PDM_OFFSET_MODEL_CONFIG
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner

from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from tuplan_garage.planning.training.modeling.models.pdm_offset_model import PDMOffsetModel
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_hybrid_planner import PDMHybridPlanner

from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.database.utils.measure import angle_diff

class MLPlannerAgents(AbstractObservation):
    """
    Simulate agents based on an ML model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario, occlusion_cfg: dict, optimization_cfg: dict, planner_type: str, \
                 pdm_hybrid_ckpt: str, tracker_cfg: dict) -> None:
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

        self._occlusion_cfg = DictConfig(occlusion_cfg)
        self._optimization_cfg = DictConfig(optimization_cfg)
        self._ego_state_history: Dict = {}
        self._agents: Dict = None    
        self._trajectory_cache: Dict = {}
        self._static_agents: List[Agent] = []

        self._tracker_cfg = tracker_cfg

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0
        self._agents = None
        self._trajectory_cache = {}
        self._ego_state_history = {}
        self._static_agents = []
        if self.model is not None:
            self.model = self.model.cpu() #we do this to avoid the model being on the gpu when we save it and try to reload it later, since it only moves onto the gpu when we make the agents

    def _get_agents(self):
        """
        Gets dict of tracked agents, or lazily creates them it 
        from vehicles at simulation start if it does not exist.
        """

        if self._agents is None:
            self._agents = {}
            for agent in self._scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):

                # Sets agent goal to be it's last known point in the simulation. 
                goal = self._get_historical_agent_goal(agent, self.current_iteration)


                if goal:

                    if self._is_parked_vehicle(agent, goal, self._scenario.map_api):
                        self._static_agents.append(agent)
                        continue

                    route_plan, _ = self._get_roadblock_path(agent, goal)

                    if not self._irrelevant_to_ego(route_plan, self._scenario):
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
        open_loop_detections.extend(self._static_agents)
        return DetectionsTracks(tracked_objects=TrackedObjects(open_loop_detections))
    
    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index
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
        for agent_token, agent_data in self._agents.items():
            if agent_token in self._trajectory_cache and \
                (next_iteration.time_s - self._trajectory_cache[agent_token][0]) < self._optimization_cfg.subsample_inference_frequency and \
            (((agent_data['ego_state'].center.x - history.current_state[0].center.x) ** 2 + \
                (agent_data['ego_state'].center.y - history.current_state[0].center.y) ** 2) ** 0.5) >= self._optimization_cfg.subsample_full_inference_distance:
                    trajectory = self._trajectory_cache[agent_token][1]
            else:
                history_input = self._build_history_input(agent_token, agent_data['ego_state'], history)

                if agent_data['occlusion'] is not None:
                    history_input = agent_data['occlusion'].occlude_input(history_input)

                planner_input = PlannerInput(iteration=iteration, history=history_input, traffic_light_data=traffic_light_data)                    
                trajectory = agent_data['planner'].compute_trajectory(planner_input)
                self._trajectory_cache[agent_token] = (next_iteration.time_point.time_s, trajectory)
                
            agent_data['ego_state'] = self._get_new_state_from_trajectory(iteration, next_iteration, agent_data['ego_state'], trajectory, agent_data['controller'])
            self._ego_state_history[agent_token][next_iteration.time_point] = agent_data['ego_state']


    def _get_new_state_from_trajectory(self, current_iteration: SimulationIteration,
                                                next_iteration: SimulationIteration,
                                                ego_state: EgoState,
                                                trajectory: AbstractTrajectory,
                                                controller: AbstractEgoController) -> EgoState:
        """
        Gets the state of the agent at a given timepoint from a trajectory.
        """

        controller.reset()
        controller.update_state(current_iteration, next_iteration, ego_state, trajectory)
        return controller.get_state()
            
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
            
        output = EgoState.build_from_rear_axle(
            rear_axle_pose=translate_longitudinally(agent.center, agent.box.length * 1 / 5 - agent.box.length / 2),
            rear_axle_velocity_2d=StateVector2D(agent.velocity.magnitude(), 0), #EgoState and Agent uses different velocity representations. It's very weird.
            rear_axle_acceleration_2d=StateVector2D(0, 0),
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

        track_heading = ego_state.car_footprint.oriented_box.center.heading
        velocity = ego_state.dynamic_car_state.center_velocity_2d

        agent_state = Agent(
            metadata=scene_object_metadata,
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=ego_state.car_footprint.oriented_box,
            velocity=StateVector2D(np.cos(track_heading) * velocity.magnitude(), np.sin(track_heading) * velocity.magnitude())
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
    
    def remove_all_of_object_types_from_scene(self, agent_types: List[TrackedObjectType], simulation: Simulation):
        """Removes all agents of a given type from the scene
        """
        if simulation._history_buffer is None:
            simulation._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
                simulation._history_buffer_size, simulation._scenario, simulation._observations.observation_type()
            )
        objects = copy.deepcopy(simulation._observations.get_observation().tracked_objects.tracked_objects)
        for obs in simulation.history_buffer.observation_buffer:
            objects.extend(copy.deepcopy(obs.tracked_objects.tracked_objects))
        
        deduped_objects = list({track.metadata.track_token:track for track in objects}.values())

        for obj in deduped_objects:
            if obj.tracked_object_type in agent_types:
                simulation._observations.remove_agent_from_scene(obj, simulation)
    
    def remove_agent_from_scene(self, agent: Agent, simulation: Simulation):
        """Removes an agent from the scene
        """
        if simulation._history_buffer is None:
            simulation._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
                simulation._history_buffer_size, simulation._scenario, simulation._observations.observation_type()
            )
        
        if agent.metadata.track_token in simulation._observations._get_agents():
            simulation._observations._get_agents().pop(agent.metadata.track_token)

        simulation._observations._static_agents = [a for a in simulation._observations._static_agents if a.metadata.track_token != agent.metadata.track_token]
        
        history_buffer = simulation._history_buffer
        new_observation_buffer = deque()
        for observations in history_buffer.observation_buffer:
            tracks = []
            for track in observations.tracked_objects.tracked_objects:
                if track.metadata.track_token != agent.metadata.track_token:
                    tracks.append(track)

            new_observation_buffer.append(DetectionsTracks(TrackedObjects(tracks)))
            
        if agent.metadata.track_token in simulation._observations._ego_state_history:
            simulation._observations._ego_state_history.pop(agent.metadata.track_token)
        
        simulation._history_buffer = SimulationHistoryBuffer(history_buffer.ego_state_buffer, new_observation_buffer, history_buffer.sample_interval)

    
    def add_agent_to_scene(self, agent: Agent, goal: StateSE2, timepoint_record: TimePoint, simulation: Simulation):
        """
        Adds agent to the scene with a given goal during the simulation runtime.
        Gets dict of tracked agents, or lazily creates them it 
        from vehicles at simulation start if it does not exist.
        """
        assert 'inserted' in agent.metadata.track_token
        
        if simulation._history_buffer is None:
            simulation._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
                simulation._history_buffer_size, simulation._scenario, simulation._observations.observation_type()
            )

        self._agents = self._get_agents() #this action is idempotent
            
        route_plan, _ = self._get_roadblock_path(agent, goal)

        if route_plan:
            self._agents[agent.metadata.track_token] = self._build_agent_record(agent, timepoint_record)

            # Initialize planner.
            planner_init = PlannerInitialization(
                route_roadblock_ids=route_plan,
                mission_goal=goal,
                map_api=self._scenario.map_api,
            )

            self._agents[agent.metadata.track_token]['planner'].initialize(planner_init)

        history_buffer = simulation._history_buffer
        new_observation_buffer = deque()

        for ego_states, observations in zip(history_buffer.ego_state_buffer, history_buffer.observation_buffer):
            past_time_point = ego_states.time_point
            tracks = deepcopy(observations.tracked_objects.tracked_objects)

            new_center = StateSE2(x = agent.box.center.x - agent.velocity.x * (timepoint_record.time_s - past_time_point.time_s), y=agent.box.center.y - agent.velocity.y * (timepoint_record.time_s - past_time_point.time_s), heading=agent.box.center.heading)

            new_metadata = SceneObjectMetadata(timestamp_us=past_time_point.time_us, token=agent.metadata.token, track_id=agent.metadata.track_id, track_token=agent.metadata.track_token, category_name=agent.metadata.category_name)
            new_box = OrientedBox(center=new_center, length=agent.box.length, width=agent.box.width, height=agent.box.height)

            new_agent = Agent(
                    metadata=new_metadata,
                    tracked_object_type=agent.tracked_object_type,
                    oriented_box=new_box,
                    velocity=agent.velocity,
                    angular_velocity=agent.angular_velocity
                )
            
            tracks.append(new_agent)

            #list.sort(tracks, key=lambda agent: agent.tracked_object_type.value)

            new_observation_buffer.append(DetectionsTracks(TrackedObjects(tracks)))


        simulation._history_buffer = SimulationHistoryBuffer(history_buffer.ego_state_buffer, new_observation_buffer, history_buffer.sample_interval)

    def _build_agent_record(self, agent: Agent, timepoint_record: TimePoint):
        """
        Create a record for an agent that contains the agent's ego state, metadata, and planner. 
        This is propagated through the simulation.
        """

        if self._optimization_cfg.mixed_agents:
            selected_planner_type = self._select_mixed_agent_type(agent, self._scenario) 
        else:
            selected_planner_type = self.planner_type

        if selected_planner_type == "ml":
            assert self.model is not None, "Must provide model for ML planner."
            planner = MLPlanner(self.model)
        elif selected_planner_type == "idm":
            planner = IDMPlanner(**IDM_AGENT_CONFIG)
        elif selected_planner_type == "pdm_closed":
            planner = PDMClosedPlanner(**PDM_CLOSED_AGENT_CONFIG, idm_policies=BatchIDMPolicy(**PDM_BATCH_IDM_CONFIG))
        elif selected_planner_type == "pdm_hybrid":
            assert self.pdm_hybrid_ckpt, "Must provide checkpoint path for PDM hybrid planner."
            planner = PDMHybridPlanner(**PDM_CLOSED_AGENT_CONFIG, idm_policies=BatchIDMPolicy(**PDM_BATCH_IDM_CONFIG), \
                                       **PDM_HYBRID_AGENT_CONFIG, model= PDMOffsetModel(**PDM_OFFSET_MODEL_CONFIG), checkpoint_path=self.pdm_hybrid_ckpt)
        else:
            raise ValueError("Invalid planner type.")

        built_ego_state = self._build_ego_state_from_agent(agent, timepoint_record)
        return {'ego_state': built_ego_state, \
                'metadata': agent.metadata,
                'planner': planner,
                'occlusion': build_occlusion_manager(self._occlusion_cfg, self._scenario) if self._occlusion_cfg.occlusion else None,
                'controller': TwoStageController(self._scenario, LQRTracker(**self._tracker_cfg, vehicle=built_ego_state.car_footprint.vehicle_parameters), KinematicBicycleModel(built_ego_state.car_footprint.vehicle_parameters))}
    
    def _get_historical_agent_goal(self, agent: Agent, iteration_index: int):
        """
        Gets the last known state of an agent and returns it.
        """

        for frame in range(self._scenario.get_number_of_iterations()-1, iteration_index, -1):
            last_scenario_frame = self._scenario.get_tracked_objects_at_iteration(frame)
            for track in last_scenario_frame.tracked_objects.tracked_objects:
                if track.metadata.track_token == agent.metadata.track_token:
                    return track.center

        return None
    
    def _get_roadblock_path(self, agent: Agent, goal: StateSE2, max_depth: int = 10):
        """
        Gets a path from the agent's current position to a goal position using a max depth BFS.
        """

        start_edge, _ = self._get_target_state_segment(agent.center, self._scenario.map_api)
        end_edge, _ = self._get_target_state_segment(goal, self._scenario.map_api)

        if start_edge is None:
            return None, None
        
        if end_edge is not None:
            gs = MaxDepthBreadthFirstSearch(start_edge)
            route_plan, path_found = gs.search(end_edge, max_depth)
        else:
            route_plan = [start_edge]

        route_plan = self._extend_path(route_plan, max_depth)
        lane_level_route_plan = route_plan 
        route_plan = [edge.get_roadblock_id() for edge in route_plan]
        route_plan = list(dict.fromkeys(route_plan))    #deduplicates
        
        if len(route_plan) == 1:
            route_plan = route_plan + route_plan

        return route_plan, lane_level_route_plan
    
    def _extend_path(self, route_plan: List[LaneGraphEdgeMapObject], min_path_length: int = 10, path_direction_offset: int = 0):
        """
        Extends a route plan to a given depth by continually going forward.
        """

        while len(route_plan) < min_path_length:
            outgoing_edges = route_plan[-1].outgoing_edges

            if not outgoing_edges:
                break

            sorted_outgoing_edges = sorted(outgoing_edges, key= lambda edge: edge.baseline_path.get_curvature_at_arc_length(0.0))
            absolute_curvatures = [abs(edge.baseline_path.get_curvature_at_arc_length(0.0)) for edge in sorted_outgoing_edges]
            idx = np.argmin(absolute_curvatures) + path_direction_offset
            idx = min(max(idx, 0), len(sorted_outgoing_edges)-1)
            route_plan.append(sorted_outgoing_edges[idx])

        return route_plan

    def _get_target_state_segment(self, target_state: StateSE2, map_api: AbstractMap
    ) -> Tuple[Optional[LaneGraphEdgeMapObject], Optional[float]]:
        """
        Gets the map object that the target state is on and the progress along the segment.
        :param target_state: The target_state of interest.
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
            angle_diff(segment.baseline_path.get_nearest_pose_from_position(target_state).heading, target_state.heading, math.pi*2)
            for segment in segments
        ]
        closest_segment = segments[np.argmin(np.abs(heading_diff))]

        progress = closest_segment.baseline_path.get_nearest_arc_length_from_position(target_state)
        return closest_segment, progress

    def _is_parked_vehicle(self, agent: Agent, goal: StateSE2, map_api: AbstractMap):
        """
        Checks if an agent is a parked vehicle.
        """

        if map_api.is_in_layer(agent.center, SemanticMapLayer.CARPARK_AREA):
            if goal.distance_to(agent.center) < self._optimization_cfg.parked_car_threshold:
                return True
            
        return False

    def _irrelevant_to_ego(self, route_plan: List[str], scenario: AbstractScenario):
        """
        Checks if an agent is irrelevant to ego.
        """
        
        if not route_plan:
            return True
        
        if not self._optimization_cfg.route_plan_culling:
            return False
        
        ego_route_plan = set(scenario.get_route_roadblock_ids())
        route_plan = set(route_plan)

        return len(ego_route_plan.intersection(route_plan)) == 0
        

    def _select_mixed_agent_type(self, agent: Agent, scenario: AbstractScenario):
        """
        Selects a planner type for a mixed planner.
        """
        if 'inserted' in agent.metadata.track_token:
            return self.planner_type
            
        ego_state_at_start = scenario.get_ego_state_at_iteration(0)
        
        if self._optimization_cfg.mixed_agent_heading_check:
            if abs(angle_diff(ego_state_at_start.rear_axle.heading, agent.center.heading, math.pi*2)) <= self._optimization_cfg.mixed_agent_heading_check_range:
                return "idm"

        for index in range(scenario.get_number_of_iterations()):
            tracks = scenario.get_tracked_objects_at_iteration(index)
            ego_state = scenario.get_ego_state_at_iteration(index)

            for copy_agent in tracks.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
                if agent.metadata.track_token == copy_agent.metadata.track_token:
                    if copy_agent.center.distance_to(ego_state.rear_axle) <= self._optimization_cfg.mixed_agent_relevance_distance:
                        return self.planner_type
        return "idm"
