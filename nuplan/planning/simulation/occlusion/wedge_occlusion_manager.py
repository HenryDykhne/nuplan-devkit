from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occlusion.abstract_occlusion_manager import AbstractOcclusionManager
from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from shapely.geometry import Polygon, Point, MultiPoint
from typing import List
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
        num_wedges: float = 360 # 360 gives wedge width of roughly 1 degrees

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
        wedges = dict()

        for i in range(len(rads)-1):
            d1 = rads[i] # create wedge
            d2 = rads[i+1]
            p1 = (self.horizon_threshold * math.cos(d1), self.horizon_threshold * math.sin(d1))
            p2 = (self.horizon_threshold * math.cos(d2), self.horizon_threshold * math.sin(d2))
            wedge = Polygon([self.ORIG, p1, p2])

            wedges[i] = wedge

        sorted_targets = sorted(targets, key=lambda x: (x.center.x - observer.center.x)**2 + (x.center.y - observer.center.y)**2) #sorts closest to farthest

        not_occluded = set() # Visible track token set

        for target in sorted_targets:
            corners_list = target.box.all_corners() #Return 4 corners of oriented box (FL, RL, RR, FR) Point2D
            corners = []
            for corner in corners_list:
                corners.append((corner.x - observer.center.x, corner.y - observer.center.y)) #we shift the corners and move them to a different data structure we can play with
            target_poly = Polygon(corners)

            angle = math.atan2(target.center.y - observer.center.y, target.center.x - observer.center.x) # we get the angle relative to the observer
            index_of_correct_wedge = int((angle * self.num_wedges) // (2 * math.pi)) # we find the wedge that covers that angle

            to_remove = set()
            if index_of_correct_wedge in wedges:
                wedge = wedges[index_of_correct_wedge]
                not_occluded.add(target.metadata.track_token)
                if target.tracked_object_type == TrackedObjectType.VEHICLE or target.tracked_object_type == TrackedObjectType.EGO:
                    to_remove.add(index_of_correct_wedge)
                else:
                    continue # if it is not a vehicle, it cannot block any wedges and so we should just proceed to the next target
            
            # here, we want to limit the wedges we check to only those that hit the target
            # we do this by calculating the maximum possible angular diameter in the way that can be seen here:
            # https://rechneronline.de/sehwinkel/angular-diameter.php
            max_possible_crossection = ((corners_list[0].x-corners_list[2].x)**2+(corners_list[0].y-corners_list[2].y)**2)**0.5
            dist = ((target.center.x - observer.center.x)**2 + (target.center.y - observer.center.y)**2)**0.5
            angular_diameter = 2 * math.atan(max_possible_crossection/(2 * dist)) # angular diameteris
            
            # with the angular diameter, we can calculate the maximum number of wedges to either side of the center the car can take up
            num_wedges_to_check_to_each_side = int((((angular_diameter / 2) * self.num_wedges) // (2 * math.pi)) + 1)
            
            # here, we check wedges, fanning out from the center of the target till we stop seeing the target
            counterclockwise_off_target = False
            clockwise_off_target = False
            for i in range(1, num_wedges_to_check_to_each_side + 1):
                if counterclockwise_off_target and clockwise_off_target:
                    break
                
                wedge_idx = (index_of_correct_wedge + i) % self.num_wedges # fan out counterclockwise
                if wedge_idx in wedges:
                    wedge = wedges[wedge_idx]
                    if wedge.intersects(target_poly):
                        not_occluded.add(target.metadata.track_token)
                        if target.tracked_object_type == TrackedObjectType.VEHICLE or target.tracked_object_type == TrackedObjectType.EGO:
                            to_remove.add(wedge_idx)
                        else:
                            break
                    else: 
                        counterclockwise_off_target = True

                wedge_idx = (index_of_correct_wedge - i) % self.num_wedges # fan out clockwise
                if wedge_idx in wedges:
                    wedge = wedges[wedge_idx]
                    if wedge.intersects(target_poly):
                        not_occluded.add(target.metadata.track_token)
                        if target.tracked_object_type == TrackedObjectType.VEHICLE or target.tracked_object_type == TrackedObjectType.EGO:
                            to_remove.add(wedge_idx)
                        else: 
                            break
                    else: 
                        clockwise_off_target = True

            for key in to_remove:
                del wedges[key]


        print('elapsed time:', time.time() - start)
        return not_occluded