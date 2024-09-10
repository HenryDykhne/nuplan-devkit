
from omegaconf import DictConfig

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.occlusion.abstract_occlusion_manager import AbstractOcclusionManager
from nuplan.planning.simulation.occlusion.range_occlusion_manager import RangeOcclusionManager
from nuplan.planning.simulation.occlusion.shadow_occlusion_manager import ShadowOcclusionManager
from nuplan.planning.simulation.occlusion.wedge_occlusion_manager import WedgeOcclusionManager



def build_occlusion_manager(occlusion_cfg: DictConfig, scenario: AbstractScenario) -> AbstractOcclusionManager:
    """
    Instantiate occlusion_manager
    :param occlusion_cfg: config of a occlusion_manager
    :param scenario: scenario
    :return occlusion_cfg
    """
    occlusion_manager: AbstractOcclusionManager
    if 'notice_threshold' in occlusion_cfg.keys() and 'uncloak_reaction_time' in occlusion_cfg.keys():
        uncloak_reaction_time = occlusion_cfg.uncloak_reaction_time
        notice_threshold = occlusion_cfg.notice_threshold
    else:
        uncloak_reaction_time, notice_threshold = AbstractOcclusionManager.__init__.__defaults__
    if occlusion_cfg.manager_type == 'range': #masks everyone further away than a set threshold
        occlusion_manager = RangeOcclusionManager(scenario, uncloak_reaction_time=uncloak_reaction_time, notice_threshold=notice_threshold)
    elif occlusion_cfg.manager_type == 'shadow': #masks everyone who is is occluded by shadows
        occlusion_manager = ShadowOcclusionManager(scenario, uncloak_reaction_time=uncloak_reaction_time, notice_threshold=notice_threshold)
    elif occlusion_cfg.manager_type == 'wedge': #calculates occlusions via wedges
        occlusion_manager = WedgeOcclusionManager(scenario, uncloak_reaction_time=uncloak_reaction_time, notice_threshold=notice_threshold)
    else:
        raise ValueError(f"Invalid manager_type selected. Got {occlusion_cfg.manager_type}")


    return occlusion_manager
