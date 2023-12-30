from collections import deque
from typing import Dict, List, Optional, Tuple
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject


class MaxDepthBreadthFirstSearch:

    def __init__(self, start_edge: LaneGraphEdgeMapObject):
        self._queue = deque([start_edge, None])
        self._parent: Dict[str, Optional[LaneGraphEdgeMapObject]] = dict()
        self._visited = set()

    def search(
        self, target_edge: LaneGraphEdgeMapObject, max_depth: int
    ) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Search for a path from the start edge to the target edge.
        :param target_edge: The target edge to search for.
        :param max_depth: The maximum depth to search for.
        :return: A tuple of:
            - The path as a list of LaneGraphEdgeMapObject
            - A boolean indicating if the path was found.
        """

        start_edge = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_edge: LaneGraphEdgeMapObject = start_edge
        end_depth: int = 1
        depth: int = 1

        self._parent[start_edge.id + f"_{depth}"] = None

        while self._queue:
            current_edge = self._queue.popleft()
            if current_edge is not None:
                self._visited.add(current_edge.id)

            # Early exit condition
            if self._check_end_condition(depth, max_depth):
                break

            # Depth tracking
            if current_edge is None:
                depth += 1
                self._queue.append(None)
                if self._queue[0] is None:
                    break
                continue

            # Goal condition
            if self._check_goal_condition(current_edge, target_edge):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                break

            # Populate queue
            for next_edge in current_edge.outgoing_edges:
                if next_edge.id not in self._visited:
                    self._queue.append(next_edge)
                    self._parent[next_edge.id + f"_{depth + 1}"] = current_edge
                    end_edge = next_edge
                    end_depth = depth + 1

        return self._construct_path(end_edge, end_depth), path_found

    @staticmethod
    def _check_end_condition(depth: int, target_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: True if:
            - The current depth exceeds the target depth.
        """

        return depth > target_depth

    @staticmethod
    def _check_goal_condition(
        current_edge: LaneGraphEdgeMapObject,
        target_edge: LaneGraphEdgeMapObject,
    ) -> bool:
        """
        Checl to see if current edge is the target edge.
        """
            
        return current_edge.id == target_edge.id

    def _construct_path(self, end_edge: LaneGraphEdgeMapObject, depth: int) -> List[LaneGraphEdgeMapObject]:
        """
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of LaneGraphEdgeMapObject
        """

        path = [end_edge]
        while self._parent[end_edge.id + f"_{depth}"] is not None:
            path.append(self._parent[end_edge.id + f"_{depth}"])
            end_edge = self._parent[end_edge.id + f"_{depth}"]
            depth -= 1
        path.reverse()

        return path
