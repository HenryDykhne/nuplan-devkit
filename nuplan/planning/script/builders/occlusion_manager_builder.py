
from omegaconf import DictConfig

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.occlusion.occlusion_manager import AbstractOcclusionManager
from nuplan.planning.simulation.occlusion.range_occlusion_manager import RangeOcclusionManager

def build_occlusion_manager(occlusion_cfg: DictConfig, scenario: AbstractScenario) -> AbstractOcclusionManager:
    """
    Instantiate occlusion_manager
    :param occlusion_cfg: config of a occlusion_manager
    :param scenario: scenario
    :return occlusion_cfg
    """
    # Placeholder
    occlusion_manager: AbstractOcclusionManager = RangeOcclusionManager(scenario)

    return occlusion_manager
