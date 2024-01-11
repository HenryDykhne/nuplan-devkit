from typing import List

import copy
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractScenarioModifier
from nuplan.planning.simulation.observation.ml_planner_agents import MLPlannerAgents

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
import math

from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
class OcclusionInjectionModifier(AbstractScenarioModifier):
    def __init__(self):
        super().__init__() #maybe we will need this later
        
    def modify_scenario(self, runner: SimulationRunner) -> List[SimulationRunner]:
        """We convert one abstract scenario into many abstract scenarios by modifying the scenario in some way.
        :param runner: a scenario
        :return: we return a list of runners that are modified versions of the input scenario
        """
        scenario = runner.scenario
        relavant_agent_tokens = self.find_relavant_agents(runner.simulation._observations, scenario)
        pass
    
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

                route_plan = observations.get_roadblock_path(agent, goal)

                if not observations._irrelevant_to_ego(route_plan, scenario):
                    relevant_agent_tokens.append(agent.track_token)
        return relevant_agent_tokens