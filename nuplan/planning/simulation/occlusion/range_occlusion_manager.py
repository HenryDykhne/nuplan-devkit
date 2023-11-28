from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occlusion.abstract_occlusion_manager import AbstractOcclusionManager


class RangeOcclusionManager(AbstractOcclusionManager):
    """
    Range occlusion manager. Occludes all objects outside of a given
    range of the ego.
    """

    def __init__(
        self,
        scenario: AbstractScenario,
        range_threshold: float = 25
    ):
        super().__init__(scenario)
        self.range_threshold = range_threshold

    def _compute_mask(self, ego_state: EgoState, observations: DetectionsTracks) -> set:
        not_occluded = set()
        for track in observations.tracked_objects.tracked_objects:
            if ((ego_state.center.x - track.center.x) ** 2 + \
                (ego_state.center.y - track.center.y) ** 2) ** 0.5 <= self.range_threshold:
                not_occluded.add(track.metadata.track_token)

        return not_occluded
    