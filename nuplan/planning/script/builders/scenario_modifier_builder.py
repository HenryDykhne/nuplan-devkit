from typing import List

from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractScenarioModifier
from nuplan.planning.scenario_builder.scenario_modifier.left_and_right_modifier import LeftAndRightModifier
from nuplan.planning.scenario_builder.scenario_modifier.occlusion_injection_modifier import OcclusionInjectionModifier


def build_scenario_modifiers(scenario_modifier_types: List[str]) -> List[AbstractScenarioModifier]:
    modifiers = []
    for type in scenario_modifier_types:
        if type == "left-and-right":
            modifiers.append(LeftAndRightModifier()) #this one is an example and just injects an agent to the left of ego in one modifiecation of the scenario and to the right in another
        elif type == "occlusion-injection":
             modifiers.append(OcclusionInjectionModifier())
        # elif type == "occludie-injection":
        #     modifiers.append(OccludieInjection())
        else:
            raise ValueError(f"Unknown scenario modifier type: {type}")
    
    return modifiers