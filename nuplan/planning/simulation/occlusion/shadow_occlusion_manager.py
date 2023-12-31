from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.occlusion.abstract_occlusion_manager import AbstractOcclusionManager
from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, STATIC_OBJECT_TYPES, TrackedObjectType
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from typing import List
from collections import deque 
import math
import numpy as np

import time


class ShadowOcclusionManager(AbstractOcclusionManager):
    """
    Range occlusion manager. Occludes all objects outside of a given
    range of the ego.
    """

    ORIG = (0,0)

    def __init__(
        self,
        scenario: AbstractScenario,
        uncloak_reaction_time: float = AbstractOcclusionManager.__init__.__defaults__[0],
        notice_threshold: float = AbstractOcclusionManager.__init__.__defaults__[1],
        horizon_threshold: float = 1000, # meters since that is how far a standing human can see unblocked before the curvature of the earth cuts your line of sight
        min_rad: float = 0.026, # minimum radians that the vehicle must take up to be observed (0.026 = aprox 1.5 degrees)
    ):
        super().__init__(scenario, uncloak_reaction_time, notice_threshold)
        self.horizon_threshold = horizon_threshold
        self.min_rad = min_rad

    def _compute_visible_agents(self, ego_state: EgoState, observations: DetectionsTracks) -> set:
        """
        Returns set of track tokens that represents the observations visible to the ego
        at this time step.
        """
        # Visible track token set
        return self._determine_occlusions(ego_state.agent, observations.tracked_objects.tracked_objects)
    
    def _determine_occlusions(self, observer: AgentState, targets:List[AgentState]) -> set:
        # start = time.time()
        not_occluded = set() # Visible track token set
        
        shadow_polys = deque([])
        for target in targets: #first we construct shadows
            if target.tracked_object_type == TrackedObjectType.VEHICLE or target.tracked_object_type == TrackedObjectType.EGO:
                corners_list = target.box.all_corners() #Return 4 corners of oriented box (FL, RL, RR, FR) Point2D
                corners = []
                for corner in corners_list:
                    corners.append((corner.x - observer.center.x, corner.y - observer.center.y)) #we shift the corners and move them to a different data structure we can play with
                horizon_points = []
                range_to_ego = ((target.center.x - observer.center.x)**2 + (target.center.y - observer.center.y)**2)**0.5
                shadow_range_from_ego = max(self.horizon_threshold, range_to_ego) #if horizon_threshold is big enough, we dont even really need to check which is the max. we can just always take the horizon threshold
                for corner in corners:
                    shadow_range_multiplier = shadow_range_from_ego / (corner[0]**2 + corner[1]**2)**0.5
                    horizon_points.append((corner[0]*shadow_range_multiplier, corner[1]*shadow_range_multiplier))
                
                shadow_polygon1 = Polygon([corners[0], horizon_points[0], horizon_points[2], corners[2]])#FL and RR gives one diagonal
                shadow_polygon2 = Polygon([corners[1], horizon_points[1], horizon_points[3], corners[3]])#RL and FR gives the second diagonal
                shadow_polys.append(shadow_polygon1)
                shadow_polys.append(shadow_polygon2)
                
        combined_shadow_poly = unary_union(shadow_polys)
        
        observer_origin = Point(self.ORIG)
        for target in targets: #now we check if each target is contained within the shadows
            corners_list = target.box.all_corners() #Return 4 corners of oriented box (FL, RL, RR, FR) Point2D
            corners = []
            for corner in corners_list:
                corners.append((corner.x - observer.center.x, corner.y - observer.center.y)) #we shift the corners and move them to a different data structure we can play with
            target_poly = Polygon(corners)
            
            diff_poly = target_poly.difference(combined_shadow_poly)
            
            hull = unary_union([diff_poly, observer_origin]).convex_hull
            if isinstance(hull, Polygon):
                try: #if the convex hull formed does not include the origin such as in the case where the target is on top of the observer, an error will be thrown
                    neighbor_1, neighbor_2 = self._get_two_neighbors(self.ORIG, hull)
                    radians = abs(math.atan2(neighbor_1.x*neighbor_2.y - neighbor_1.y*neighbor_2.x, neighbor_1.x*neighbor_2.x + neighbor_1.y*neighbor_2.y ))
                    if radians > self.min_rad:
                        not_occluded.add(target.metadata.track_token)
                except ValueError: # but if the target is on top of you, you can probably see it
                    not_occluded.add(target.metadata.track_token)
                    
        # print('elapsed time:', time.time() - start)
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
