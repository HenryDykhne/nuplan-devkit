from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occlusion.abstract_occlusion_manager import AbstractOcclusionManager
from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import unary_union
from typing import List
from collections import deque 
import math
import numpy as np

import time


class WedgeOcclusionManager(AbstractOcclusionManager):
    """
    Range occlusion manager. Occludes all objects outside of a given
    range of the ego.
    """

    ORIG = (0,0)

    def __init__(
        self,
        scenario: AbstractScenario,
        horizon_threshold: float = 1000, # meters since that is how far a standing human can see unblocked before the curvature of the earth cuts your line of sight
        num_wedges: float = 180 # gives wedge width of roughly 2 degrees

    ):
        super().__init__(scenario)
        self.horizon_threshold = horizon_threshold
        self.num_wedges = num_wedges

    def _compute_visible_agents(self, ego_state: EgoState, observations: DetectionsTracks) -> set:
        """
        Returns set of track tokens that represents the observations visible to the ego
        at this time step.
        """
        # Visible track token set
        return self._determine_occlusions(ego_state.agent, observations.tracked_objects.tracked_objects)

    # wedge based occlusion implementation. about half as fast and the occlusions flicker more but it should scale better if you have tons of occluders
    def _determine_occlusions(self, observer: AgentState, targets:List[AgentState]) -> set:
        start = time.time()
        rads = np.linspace(0,2*math.pi,self.num_wedges+1)
        wedges = set()

        for i in range(len(rads)-1):
            d1 = rads[i] # create wedge
            d2 = rads[i+1]
            p1 = (self.horizon_threshold * math.cos(d1), self.horizon_threshold * math.sin(d1))
            p2 = (self.horizon_threshold * math.cos(d2), self.horizon_threshold * math.sin(d2))
            wedge = Polygon([ORIG, p1, p2])

            wedges.add(wedge)

        sorted_targets = sorted(targets, key=lambda x: (x.center.x - observer.center.x)**2 + (x.center.y - observer.center.y)**2) #sorts closest to farthest

        not_occluded = set() # Visible track token set

        for target in sorted_targets:
            corners_list = target.box.all_corners() #Return 4 corners of oriented box (FL, RL, RR, FR) Point2D
            corners = []
            for corner in corners_list:
                corners.append((corner.x - observer.center.x, corner.y - observer.center.y)) #we shift the corners and move them to a different data structure we can play with
            target_poly = Polygon(corners)

            to_remove = set()
            for wedge in wedges:
                if wedge.intersects(target_poly):
                    if target.tracked_object_type == TrackedObjectType.VEHICLE or target.tracked_object_type == TrackedObjectType.EGO:
                        to_remove.add(wedge)
                    not_occluded.add(target.metadata.track_token)
            wedges -= to_remove


        print('elapsed time:', time.time() - start)
        return not_occluded
    
    def _get_two_neighbors(self, point, polygon) -> List[Point]:
        """retrieve the two neighboring points of point in the polygon
        :point: a tuple representing a point of the polygon
        :polygon: a shapely Polygon
        return: a tuple of the two points immediately neighbors of point 
        """
        points = list(polygon.exterior.coords)
        ndx = points.index(point)
        two_neighbors = [points[(ndx-1)%len(points)], points[(ndx+1)%len(points)]]
        if two_neighbors[0] == point:
            two_neighbors[0] = points[(ndx-2)%len(points)]
        if two_neighbors[1] == point:
            two_neighbors[1] = points[(ndx+2)%len(points)]
        two_neighbors[0] = Point(two_neighbors[0])
        two_neighbors[1] = Point(two_neighbors[1])
        return two_neighbors
