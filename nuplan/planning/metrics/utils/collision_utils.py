from __future__ import annotations

from enum import IntEnum
import math
from typing import Dict, List

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import CollisionData
from nuplan.planning.metrics.metric_result import MetricStatisticsType, Statistic

VRU_types = [
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]

object_types = [
    TrackedObjectType.TRAFFIC_CONE,
    TrackedObjectType.BARRIER,
    TrackedObjectType.CZONE_SIGN,
    TrackedObjectType.GENERIC_OBJECT,
]


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


def ego_delta_v_collision(
    ego_state: EgoState, scene_object: SceneObject, ego_mass: float = 2000, agent_mass: float = 2000
) -> float:
    """
    Compute the ego delta V (loss of velocity during the collision). Delta V represents the intensity of the collision
    of the ego with other agents.
    :param ego_state: The state of ego.
    :param scene_object: The scene_object ego is colliding with.
    :param ego_mass: mass of ego.
    :param agent_mass: mass of the agent.
    :return The delta V measure for ego.
    """
    # Collision metric is defined as the ratio of agent mass to the overall mass of ego and agent, times the changes in velocity defined as
    # sqrt(ego_speed^2 + agent_speed^2 - 2 * ego_speed * agent_speed * cos(heading_difference))
    ego_mass_ratio = agent_mass / (agent_mass + ego_mass)

    scene_object_speed = scene_object.velocity.magnitude() if isinstance(scene_object, Agent) else 0

    sum_speed_squared = ego_state.dynamic_car_state.speed**2 + scene_object_speed**2
    cos_rule_term = (
        2
        * ego_state.dynamic_car_state.speed
        * scene_object_speed
        * np.cos(ego_state.rear_axle.heading - scene_object.center.heading)
    )
    velocity_component = float(np.sqrt(sum_speed_squared - cos_rule_term))

    return ego_mass_ratio * velocity_component

def get_pdof(ego_state: EgoState, scene_object: SceneObject) -> float:
    """
    Compute the PDOF (principal direction of force) for the collision.
    :param ego_state: The state of ego.
    :param scene_object: The scene_object ego is colliding with.
    :return The PDOF measure for the collision.
    """
    scene_object_speed = scene_object.velocity.magnitude() if isinstance(scene_object, Agent) else 0
    scene_object_velocity = (scene_object_speed * np.cos(scene_object.center.heading), scene_object_speed * np.sin(scene_object.center.heading))
    
    ego_speed = ego_state.dynamic_car_state.speed
    ego_velocity = (ego_speed * np.cos(ego_state.rear_axle.heading), ego_speed * np.sin(ego_state.rear_axle.heading))
    
    return math.atan2(ego_velocity[1] - scene_object_velocity[1], ego_velocity[0] - scene_object_velocity[0]) - math.atan2(ego_velocity[1], ego_velocity[0])

def get_probability_of_mais3(collision_delta_v: float, pdof: float, role: str = 'driver') -> float:
    """
    Compute the probability of MAIS3 (Maximum Abbreviated Injury Scale 3) for the collision.
    :param collision_delta_v: The delta V measure for the collision.
    :param pdof: The PDOF (principal direction of force) for the collision.
    :return The probability of MAIS3 for the collision.
    """

    #imputed version MAIS3+ coefs
    INTERCEPT = -9.333
    COS_PDOF = 0.363
    SIN_PDOF = -0.346
    COS_2_PDOF = -0.253
    SIN_2_PDOF = 0.205
    DELTA_V = 0.094
    OBJECT_HIT_SUV_TRUCK_VAN = 0.199
    OBJECT_HIT_LARGE = 1.362
    OBJECT_HIT_OTHER = 0.740
    BODY_TYPE_SUV_TRUCK_VAN = -0.065
    UNBELTED = 1.667
    OTHER_BELT = 0.954
    FEMALE = 0.185
    REAR_SEAT = -0.143
    ROLE_PASSENGER = 0.310
    AGE = 0.040
    MODEL_YEAR_2010 = -0.020
    COS_PDOF_DELTA_V = 0.003
    SIN_PDOF_DELTA_V = -0.003
    COS_2_PDOF_DELTA_V = -0.014
    SIN_2_PDOF_DELTA_V = -0.005

    delta_v = collision_delta_v * 3.6 # change in speed between after and before impact (km/h)
    if role == 'driver':
        rear_seat = 0 # sitting in rear seat?
        role_passenger = 0 # not the driver?
    elif role == 'passenger':
        rear_seat = 0
        role_passenger = 1
    else: #rear seat passenger
        rear_seat = 1
        role_passenger = 1

    mais3_logit = INTERCEPT

    mais3_logit += np.cos(pdof) * COS_PDOF
    mais3_logit += np.sin(pdof) * SIN_PDOF
    mais3_logit += np.cos(2 * pdof) * COS_2_PDOF
    mais3_logit += np.sin(2 * pdof) * SIN_2_PDOF
    mais3_logit += delta_v * DELTA_V
    mais3_logit += 1 * OBJECT_HIT_SUV_TRUCK_VAN
    mais3_logit += 0 * OBJECT_HIT_LARGE
    mais3_logit += 0 * OBJECT_HIT_OTHER
    mais3_logit += 1 * BODY_TYPE_SUV_TRUCK_VAN
    mais3_logit += 0 * UNBELTED
    mais3_logit += 0 * OTHER_BELT
    mais3_logit += 0.54 * FEMALE
    mais3_logit += rear_seat * REAR_SEAT 
    mais3_logit += role_passenger * ROLE_PASSENGER
    mais3_logit += 41 * AGE
    mais3_logit += 0 * MODEL_YEAR_2010
    mais3_logit += np.cos(pdof) * delta_v * COS_PDOF_DELTA_V
    mais3_logit += np.sin(pdof) * delta_v * SIN_PDOF_DELTA_V
    mais3_logit += np.cos(2 * pdof) * delta_v * COS_2_PDOF_DELTA_V
    mais3_logit += np.sin(2 * pdof) * delta_v * SIN_2_PDOF_DELTA_V

    e_logit = np.exp(mais3_logit)
    mais3 = e_logit / (1 + e_logit)
    return mais3
    
def get_severity_class(velocity_range: float, collision_impact_angle: float) -> int:
    if abs(collision_impact_angle) < 0.38: #front
        if velocity_range <= 1.11: #4 km/h
            return 0
        elif velocity_range <= 5.56: #20 km/h
            return 1
        elif velocity_range <= 11.11: #40 km/h
            return 2
        else:
            return 3
    elif abs(collision_impact_angle) > np.pi - 0.38: #back
        if velocity_range <= 1.11:  #4 km/h
            return 0
        elif velocity_range <= 5.56: #20 km/h
            return 1
        elif velocity_range <= 11.11: #40 km/h
            return 2
        else:
            return 3
    else: #side
        if velocity_range <= 0.56: #2 km/h
            return 0
        elif velocity_range <= 2.22: #8 km/h
            return 1
        elif velocity_range <= 4.44: #16 km/h
            return 2
        else:
            return 3

    

def get_fault_type_statistics(
    all_at_fault_collisions: Dict[TrackedObjectType, List[float]],
) -> List[Statistic]:
    """
    :param all_at_fault_collisions: Dict of at_fault collisions.
    :return: List of Statistics for all collision track types.
    """
    statistics = []
    track_types_collisions_energy_dict: Dict[str, List[float]] = {}

    for collision_track_type, collision_name in zip(
        [VRU_types, [TrackedObjectType.VEHICLE], object_types], ['VRUs', 'vehicles', 'objects']
    ):
        track_types_collisions_energy_dict[collision_name] = [
            colision_energy
            for track_type in collision_track_type
            for colision_energy in all_at_fault_collisions[track_type]
        ]
        statistics.extend(
            [
                Statistic(
                    name=f'number_of_at_fault_collisions_with_{collision_name}',
                    unit=MetricStatisticsType.COUNT.unit,
                    value=len(track_types_collisions_energy_dict[collision_name]),
                    type=MetricStatisticsType.COUNT,
                )
            ]
        )
    for collision_name, track_types_collisions_energy in track_types_collisions_energy_dict.items():
        if len(track_types_collisions_energy) > 0:
            statistics.extend(
                [
                    Statistic(
                        name=f'max_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=max(track_types_collisions_energy),
                        type=MetricStatisticsType.MAX,
                    ),
                    Statistic(
                        name=f'min_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=min(track_types_collisions_energy),
                        type=MetricStatisticsType.MIN,
                    ),
                    Statistic(
                        name=f'mean_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=np.mean(track_types_collisions_energy),
                        type=MetricStatisticsType.MEAN,
                    ),
                ]
            )
    return statistics

def get_type_statistics(
    all_ego_collisions: Dict[TrackedObjectType, List[float]],
    ego_collision_data: Dict[TrackedObjectType, List[CollisionData]],
) -> List[Statistic]:
    """
    :param all_ego_collisions: Dict of ego collisions.
    :param ego_collision_data: Dict of ego collisions with all relavant data.
    :return: List of Statistics for all collision track types.
    """
    statistics = []
    track_types_collisions_energy_dict: Dict[str, List[float]] = {}
    track_types_collisions_data_dict: Dict[str, List[CollisionData]] = {}

    for collision_track_type, collision_name in zip(
        [VRU_types, [TrackedObjectType.VEHICLE], object_types], ['VRUs', 'vehicles', 'objects']
    ):
        track_types_collisions_energy_dict[collision_name] = [
            colision_energy
            for track_type in collision_track_type
            for colision_energy in all_ego_collisions[track_type]
        ]
        track_types_collisions_data_dict[collision_name] = [
            colision_data
            for track_type in collision_track_type
            for colision_data in ego_collision_data[track_type]
        ]
        statistics.extend(
            [
                Statistic(
                    name=f'number_of_ego_collisions_with_{collision_name}',
                    unit=MetricStatisticsType.COUNT.unit,
                    value=len(track_types_collisions_energy_dict[collision_name]),
                    type=MetricStatisticsType.COUNT,
                )
            ]
        )
    for collision_name, track_types_collisions_energy in track_types_collisions_energy_dict.items():
        if len(track_types_collisions_energy) > 0:
            first_col_idx = np.argmin([col_data.time for col_data in track_types_collisions_data_dict[collision_name]])
            statistics.extend(
                [
                    Statistic(
                        name=f'max_ego_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=max(track_types_collisions_energy),
                        type=MetricStatisticsType.MAX,
                    ),
                    Statistic(
                        name=f'min_ego_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=min(track_types_collisions_energy),
                        type=MetricStatisticsType.MIN,
                    ),
                    Statistic(
                        name=f'mean_ego_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=np.mean(track_types_collisions_energy),
                        type=MetricStatisticsType.MEAN,
                    ),
                    Statistic(
                        name=f'first_ego_collision_energy_with_{collision_name}',
                        unit="meters_per_second",
                        value=track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v,
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'relative_impact_angle_at_first_ego_collision_with_{collision_name}',
                        unit="radians",
                        value=track_types_collisions_data_dict[collision_name][first_col_idx].collision_impact_angle,
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'relative_heading_angle_at_first_ego_collision_with_{collision_name}',
                        unit="radians",
                        value=track_types_collisions_data_dict[collision_name][first_col_idx].collision_angle,
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'pdof_at_first_ego_collision_with_{collision_name}',
                        unit="radians",
                        value=track_types_collisions_data_dict[collision_name][first_col_idx].collision_pdof,
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'mais3+_driver_at_first_ego_collision_with_{collision_name}',
                        unit="probability",
                        value=get_probability_of_mais3(track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v, track_types_collisions_data_dict[collision_name][first_col_idx].collision_pdof, 'driver'),
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'mais3+_passenger_at_first_ego_collision_with_{collision_name}',
                        unit="probability",
                        value=get_probability_of_mais3(track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v, track_types_collisions_data_dict[collision_name][first_col_idx].collision_pdof, 'passenger'),
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'mais3+_backseat_at_first_ego_collision_with_{collision_name}',
                        unit="probability",
                        value=get_probability_of_mais3(track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v, track_types_collisions_data_dict[collision_name][first_col_idx].collision_pdof, 'backseat'),
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'mais3+_worst_case_driver_at_first_ego_collision_with_{collision_name}',
                        unit="probability",
                        value=get_probability_of_mais3(2 * track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v, track_types_collisions_data_dict[collision_name][first_col_idx].collision_pdof, 'driver'),
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'mais3+_worst_case_passenger_at_first_ego_collision_with_{collision_name}',
                        unit="probability",
                        value=get_probability_of_mais3(2 * track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v, track_types_collisions_data_dict[collision_name][first_col_idx].collision_pdof, 'passenger'),
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'mais3+_worst_case_backseat_at_first_ego_collision_with_{collision_name}',
                        unit="probability",
                        value=get_probability_of_mais3(2 * track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v, track_types_collisions_data_dict[collision_name][first_col_idx].collision_pdof, 'backseat'),
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(
                        name=f'severity_class_at_first_ego_collision_with_{collision_name}',
                        unit="severity_class",
                        value=get_severity_class(track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v, track_types_collisions_data_dict[collision_name][first_col_idx].collision_impact_angle),
                        type=MetricStatisticsType.VALUE,
                    ),
                    Statistic(#we always assume that the two vehicles weight the same ammount in the prior calculation. here, we assume that the other vehicle is much more massive
                        name=f'severity_class_worst_case_at_first_ego_collision_with_{collision_name}',
                        unit="severity_class",
                        value=get_severity_class(2 * track_types_collisions_data_dict[collision_name][first_col_idx].collision_ego_delta_v, track_types_collisions_data_dict[collision_name][first_col_idx].collision_impact_angle),
                        type=MetricStatisticsType.VALUE,
                    )
    
                ]
            )
    return statistics

