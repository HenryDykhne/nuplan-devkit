from collections import deque
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
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper


OPEN_LOOP_DETECTION_TYPES = [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE, TrackedObjectType.CZONE_SIGN, TrackedObjectType.BARRIER, TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.GENERIC_OBJECT]

class MLPlannerAgents(AbstractObservation):
    """
    Simulate agents based on an ML model.
    """

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario) -> None:
        """
        Initializes the MLPlannerAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        self.current_iteration = 0
        self.model = model
        self._scenario = scenario
        self._ego_state_history = {}
        self._agents: Dict = None

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0
        self._agents = None

    def _get_agents(self):


        if not self._agents:
            self._agents = {}
            for agent in self._scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
                self._agents[agent.metadata.track_token] = {'ego_state': self._build_ego_state_from_agent(agent, self._scenario.start_time), \
                                                            'metadata': agent.metadata,
                                                            'planner': MLPlanner(self.model)}
                # TODO: Support ego controllers - right now just doing perfect tracking
                # TODO: Handle ego-states better
                # TODO: Handle goals better (lol)

                goal = None
                frame_off=1
                while goal is None:
                    last_scenario_frame = self._scenario.get_tracked_objects_at_iteration(self._scenario.get_number_of_iterations()-frame_off)
                    tracked_set = [track for track in last_scenario_frame.tracked_objects.tracked_objects if track.metadata.track_token == agent.metadata.track_token]

                    if tracked_set:
                        goal = tracked_set[0].center
                    else: 
                        frame_off += 1

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
        agents = [self._build_agent_from_ego_state(v['ego_state'], v['metadata']) for k, v in self._get_agents().items()]
        open_loop_detections = self._get_open_loop_track_objects(self.current_iteration)
        open_loop_detections.extend(agents)
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

        traffic_light_data = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))

        for agent_token, agent_data in self._agents.items():
            history_input = self._build_history_input(agent_token, agent_data['ego_state'], history)
            planner_input = PlannerInput(iteration=iteration, history=history_input, traffic_light_data=traffic_light_data)
            trajectory = agent_data['planner'].compute_trajectory(planner_input)
            agent_data['ego_state'] = trajectory.get_state_at_time(next_iteration.time_point)
            self._ego_state_history[agent_token][next_iteration.time_point] = agent_data['ego_state']
            
    def _build_ego_state_from_agent(self, agent: Agent, time_point: TimePoint) -> EgoState:

        if agent.metadata.track_token in self._ego_state_history:
            if time_point in self._ego_state_history[agent.metadata.track_token]:
                return self._ego_state_history[agent.metadata.track_token][time_point]
        else:
            self._ego_state_history[agent.metadata.track_token] = {}

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
            #angular_vel=agent.angular_velocity if agent.angular_velocity is not None else 0.0
        )

        self._ego_state_history[agent.metadata.track_token][time_point] = output
        return output
    
    def _build_agent_from_ego_state(self, ego_state: EgoState, scene_object_metadata: SceneObjectMetadata) -> Agent:
        agent_state = Agent(
            metadata=scene_object_metadata,
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=ego_state.car_footprint.oriented_box,
            velocity=ego_state.dynamic_car_state.center_velocity_2d,
            #angular_velocity=ego_state.dynamic_car_state.angular_velocity
        )
        return agent_state
    
    def _build_history_input(self, agent_track_token: str, current_state: EgoState, history: SimulationHistoryBuffer) -> SimulationHistoryBuffer:

        ego_state_buffer = history.ego_state_buffer
        observation_buffer = history.observation_buffer

        new_observations = []
        faux_ego_obervations = []

        for t, (ego_state, observation) in enumerate(zip(ego_state_buffer, observation_buffer)):
            ego_agent_object = self._build_agent_from_ego_state(ego_state, SceneObjectMetadata(token='ego', track_token="ego", track_id=-1, timestamp_us=ego_state.time_us))
            
            i = t
            matched_obs = []
            while not matched_obs:
                matched_obs = [ag for ag in observation_buffer[i].tracked_objects.tracked_objects if ag.metadata.track_token == agent_track_token]
                i += 1
            
            faux_ego_observation = self._build_ego_state_from_agent(matched_obs[0], ego_state.time_point)

            tracks = [ag for ag in observation.tracked_objects.tracked_objects if ag.metadata.track_token != agent_track_token]
            tracks.append(ego_agent_object)

    
            new_observations.append(DetectionsTracks(tracked_objects=TrackedObjects(tracks)))
            faux_ego_obervations.append(faux_ego_observation)
    

        output_buffer = SimulationHistoryBuffer(deque(faux_ego_obervations), \
                            deque(new_observations), \
                                history.sample_interval)

        return output_buffer
    
    def add_agent_to_scene(self, agent: Agent, goal: StateSE2, timepoint: TimePoint):

        self._agents[agent.metadata.track_token] = {'ego_state': self._build_ego_state_from_agent(agent, timepoint), \
                                                    'metadata': agent.metadata,
                                                    'planner': MLPlanner(self.model)}
        planner_init = PlannerInitialization(
                route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
                mission_goal=goal,
                map_api=self._scenario.map_api,
            )
        
        self.agents[agent.metadata.track_token]['planner'].initialize(planner_init)
