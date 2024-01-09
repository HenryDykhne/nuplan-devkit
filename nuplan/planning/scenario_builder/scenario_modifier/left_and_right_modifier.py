from typing import List

import copy
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractScenarioModifier


from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
import math

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
class LeftAndRightModifier(AbstractScenarioModifier):
    def __init__(self):
        super().__init__() #maybe we will need this later
        
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        """We convert one abstract scenario into many abstract scenarios by modifying the scenario in some way.
        :param scenario: a scenario
        :return: we return a list of scenarios that are modified versions of the input scenario
        """
        modified_simulation_runners = []
        left = copy.deepcopy(runner)
        right = copy.deepcopy(runner)
        
        scenario = runner.scenario
        
        
        angle = scenario.initial_ego_state.center.heading
        inserted_agent = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=OrientedBox(StateSE2(scenario.initial_ego_state.center.x - 4, scenario.initial_ego_state.center.y, angle), 5, 2, 2),
            velocity=StateVector2D(scenario.initial_ego_state.agent._velocity.x, scenario.initial_ego_state.agent._velocity.y),
            metadata=SceneObjectMetadata(1623707858950113, "inserted_left", -2, "inserted_left"),
            angular_velocity=0.0,
        )

        inserted_goal = StateSE2(scenario.initial_ego_state.center.x, scenario.initial_ego_state.center.y, 1.25)
        
        iter = runner.simulation._time_controller.get_iteration()
        left.simulation._observations.add_agent_to_scene(
            inserted_agent, inserted_goal, iter.time_point
        )
        
        left.scenario._modifier = "left"
        
        ##############################
        
        angle = scenario.initial_ego_state.center.heading
        inserted_agent = Agent(
            tracked_object_type=TrackedObjectType.VEHICLE,
            oriented_box=OrientedBox(StateSE2(scenario.initial_ego_state.center.x + 4, scenario.initial_ego_state.center.y, angle), 5, 2, 2),
            velocity=StateVector2D(scenario.initial_ego_state.agent._velocity.x, scenario.initial_ego_state.agent._velocity.y),
            metadata=SceneObjectMetadata(1623707858950113, "inserted_right", -2, "inserted_right"),
            angular_velocity=0.0,
        )

        inserted_goal = StateSE2(scenario.initial_ego_state.center.x, scenario.initial_ego_state.center.y, 1.25)
        
        iter = runner.simulation._time_controller.get_iteration()
        right.simulation._observations.add_agent_to_scene(
            inserted_agent, inserted_goal, iter.time_point
        )
        
        right.scenario._modifier = "right"
        #######################################
        
        modified_simulation_runners.append(left)
        modified_simulation_runners.append(right)
        
        return modified_simulation_runners