from typing import List

import copy
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractModification, AbstractScenarioModifier

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
import math

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
class LeftAndRightModifier(AbstractScenarioModifier):
    def __init__(self):
        super().__init__() #maybe we will need this later
        
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        """We convert one abstract scenario into many abstract scenarios by modifying the scenario in some way.
        :param runner: a scenario
        :return: we return a list of runners that are modified versions of the input scenario
        """
        modified_simulation_runners = []
        left = copy.deepcopy(runner)
        right = copy.deepcopy(runner)
        
        scenario = runner.scenario
        iteration = runner.simulation._time_controller.get_iteration()
        
        left_angle = scenario.initial_ego_state.center.heading
        left_inserted_agent = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=OrientedBox(StateSE2(scenario.initial_ego_state.center.x - 4, scenario.initial_ego_state.center.y, left_angle), 5, 2, 2),
            velocity=StateVector2D(scenario.initial_ego_state.agent._velocity.x, scenario.initial_ego_state.agent._velocity.y),
            metadata=SceneObjectMetadata(scenario.get_time_point(0).time_us, "inserted_left", -2, "inserted_left"),
            angular_velocity=0.0,
        )

        left_inserted_goal = StateSE2(scenario.initial_ego_state.center.x, scenario.initial_ego_state.center.y, 1.25)
        
        right_angle = scenario.initial_ego_state.center.heading
        right_inserted_agent = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=OrientedBox(StateSE2(scenario.initial_ego_state.center.x + 4, scenario.initial_ego_state.center.y, right_angle), 5, 2, 2),
            velocity=StateVector2D(scenario.initial_ego_state.agent._velocity.x, scenario.initial_ego_state.agent._velocity.y),
            metadata=SceneObjectMetadata(scenario.get_time_point(0).time_us, "inserted_right", -2, "inserted_right"),
            angular_velocity=0.0,
        )

        right_inserted_goal = StateSE2(scenario.initial_ego_state.center.x, scenario.initial_ego_state.center.y, 1.25)
        
        left_modification = LeftAndRightModification(left_inserted_agent, left_inserted_goal, iteration.time_point, "left")
        right_modification = LeftAndRightModification(right_inserted_agent, right_inserted_goal, iteration.time_point, "right")
        
        left.simulation.modification = left_modification
        right.simulation.modification = right_modification
        
        left_modification.modify(left.simulation)
        right_modification.modify(right.simulation)
        
        modified_simulation_runners.append(left)
        modified_simulation_runners.append(right)
        
        return modified_simulation_runners
    
    
class LeftAndRightModification(AbstractModification):
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