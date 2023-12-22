from collections import deque
from copy import deepcopy
from typing import Dict, List, Type

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters

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
        self._agent_presence_threshold: float = 10

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
                #print(goal)
                if goal:
                    # Estimates ego states from agent state at simulation starts, stores metadata and creates planner for each agent
                    self._agents[agent.metadata.track_token] = self._build_agent_record(agent, self._scenario.start_time)

                    # Initialize planner.
                    planner_init = PlannerInitialization(
                            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
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
                    planner_input = agent_data['occlusion'].occlude_input(history_input)

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
            center_velocity_2d=agent.velocity,
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
                faux_ego_observation._time_point = ego_state.time_point
            else:
                faux_ego_observation = self._build_ego_state_from_agent(matched_agent, ego_state.time_point)


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

        planner_init = PlannerInitialization(
                route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
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
        for frame in range(self._scenario.get_number_of_iterations()-1, iteration_index+self._agent_presence_threshold, -1):
            last_scenario_frame = self._scenario.get_tracked_objects_at_iteration(frame)
            for track in last_scenario_frame.tracked_objects.tracked_objects:
                if track.metadata.track_token == agent.metadata.track_token:
                    return track.center

        return None
    
    def _add_newly_detected_agents(self, next_iteration: SimulationIteration):
        for agent in self._scenario.get_tracked_objects_at_iteration(next_iteration.index).tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
            if agent.metadata.track_token not in self._agents:
                goal = self._get_historical_agent_goal(agent, next_iteration.index)

                if goal:
                    # Estimates ego states from agent state at simulation starts, stores metadata and creates planner for each agent
                    self._agents[agent.metadata.track_token] = self._build_agent_record(agent, next_iteration.time_point)

                    # Initialize planner.
                    planner_init = PlannerInitialization(
                            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
                            mission_goal=goal,
                            map_api=self._scenario.map_api,
                        )
                    
                    self._agents[agent.metadata.track_token]['planner'].initialize(planner_init)
