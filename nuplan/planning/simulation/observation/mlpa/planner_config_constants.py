
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


OPEN_LOOP_DETECTION_TYPES = [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE, \
                             TrackedObjectType.CZONE_SIGN, TrackedObjectType.BARRIER, \
                             TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.GENERIC_OBJECT]


# Taken from idm_planner.yaml
IDM_AGENT_CONFIG = {  
    "target_velocity": 10,             # Desired velocity in free traffic [m/s]
    "min_gap_to_lead_agent": 1.0,      # Minimum relative distance to lead vehicle [m]
    "headway_time": 1.5,               # Desired time headway. The minimum possible time to the vehicle in front [s]
    "accel_max": 1.0,                  # Maximum acceleration [m/s^2]
    "decel_max": 3.0,                  # Maximum deceleration (positive value) [m/s^2]
    "planned_trajectory_samples": 16,  # Number of trajectory samples to generate
    "planned_trajectory_sample_interval": 0.5,  # The sampling time interval between samples [s]
    "occupancy_map_radius": 40,        # The range around the ego to add objects to be considered [m]
}

# Taken from pdm_closed_planner.yaml in the TuPlan code
PDM_CLOSED_AGENT_CONFIG = {  
    "trajectory_sampling": TrajectorySampling(num_poses=80, interval_length= 0.1),
    "proposal_sampling": TrajectorySampling(num_poses=40, interval_length= 0.1),
    "lateral_offsets": [-1.0, 1.0], 
    "map_radius": 50,
}

PDM_BATCH_IDM_CONFIG = {
    "speed_limit_fraction":[0.2,0.4,0.6,0.8,1.0], 
    "fallback_target_velocity":15.0,#15.0
    "min_gap_to_lead_agent":1.0,#1.0
    "headway_time":1.5,#1.5
    "accel_max":1.5,#1.5
    "decel_max":3.0#3.0
}

# Taken from pdm_hybrid_planner.yaml in the TuPlan code
PDM_OFFSET_MODEL_CONFIG = {
    "trajectory_sampling": TrajectorySampling(num_poses=16, interval_length=0.5), 
    "history_sampling": TrajectorySampling(num_poses=10, interval_length=0.2),
    "planner": None,
    "centerline_samples": 120,
    "centerline_interval": 1.0,
    "hidden_dim": 512
}

PDM_HYBRID_AGENT_CONFIG = {  
    "correction_horizon": 2.0,
}
