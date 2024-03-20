from typing import List

from omegaconf import DictConfig

from nuplan.planning.scenario_builder.scenario_modifier.abstract_scenario_modifier import AbstractScenarioModifier
from nuplan.planning.scenario_builder.scenario_modifier.left_and_right_modifier import LeftAndRightModifier
from nuplan.planning.scenario_builder.scenario_modifier.occlusion_injection_modifier import OcclusionInjectionModifier
from nuplan.planning.scenario_builder.scenario_modifier.oncoming_vehicle_injection_for_left_turn_and_occlusion_injection import OncomingInjectionForLeftTurnAndOcclusionInjectionModifier
from nuplan.planning.scenario_builder.scenario_modifier.conflict_vehicle_injection_and_occlusion_injection import ConflictInjectionAndOcclusionInjectionModifier
from nuplan.planning.scenario_builder.scenario_modifier.sequential_conflict_with_occlusion_injection_modifier import SequentialConflictWithOcclusionInjectionModifier


def build_scenario_modifiers(scenario_modifier_types: List[str], cfg: DictConfig = None) -> List[AbstractScenarioModifier]:
    modifiers = []
    for type in scenario_modifier_types:
        if type == "left-and-right":
            modifiers.append(LeftAndRightModifier()) #this one is an example and just injects an agent to the left of ego in one modifiecation of the scenario and to the right in another
        elif type == "occlusion-injection":
             modifiers.append(OcclusionInjectionModifier())
        elif type == "oncoming-left-turn-and-occlusion-injection":
            modifiers.append(OncomingInjectionForLeftTurnAndOcclusionInjectionModifier())
        elif type == "conflict-and-occlusion-injection":
            modifiers.append(ConflictInjectionAndOcclusionInjectionModifier())
        elif type == "sequential-conflict-with-occlusion-injection":
            modifiers.append(SequentialConflictWithOcclusionInjectionModifier(cfg))
        # elif type == "occludie-injection":
        #     modifiers.append(OccludieInjection())
        else:
            raise ValueError(f"Unknown scenario modifier type: {type}")
    
    return modifiers