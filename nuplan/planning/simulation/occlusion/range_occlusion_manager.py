from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.occlusion.occlusion_manager import AbstractOcclusionManager


class RangeOcclusionManager(AbstractOcclusionManager):
    def _compute_mask(self, ego_state: EgoState, observations: DetectionsTracks) -> set:
        not_occluded = set()
        for track in observations.tracked_objects.tracked_objects:
            if (ego_state.center.x - track.center.x) ** 2 + (ego_state.center.y - track.center.y) ** 2 <= 500:
                not_occluded.add(track.metadata.track_token)

        return not_occluded