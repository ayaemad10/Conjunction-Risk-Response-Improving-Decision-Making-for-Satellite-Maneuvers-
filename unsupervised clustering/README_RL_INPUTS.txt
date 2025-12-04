==============================
RL INPUT FILES DOCUMENTATION
==============================

This package contains all required inputs for building a Reinforcement Learning (RL) environment for conjunction-risk maneuver optimization.

Prepared by: Nada Hesham

------------------------------------------------------------
1. FILE: events_summary_for_RL.csv
------------------------------------------------------------
This is the main dataset containing all required event-level information for RL environment setup.

Includes:
- Raw physical features:
    miss_distance
    relative_speed
    relative_position_r
    relative_position_t
    relative_position_n
    relative_velocity_r
    relative_velocity_t
    relative_velocity_n

- Derived magnitude features:
    rel_pos_mag
    rel_vel_mag

- Log-transformed versions:
    miss_distance_log
    relative_speed_log
    rel_pos_mag_log
    rel_vel_mag_log

- Risk metrics:
    collision_probability
    collision_max_probability

- Unsupervised learning outputs:
    cluster          (KMeans)
    dbscan_label     (DBSCAN)
    anomaly_flag     (combined unified anomaly indicator)

- Metadata fields:
    object1_object_designator
    object2_object_designator
    object1_object_type
    object2_object_type
    object1_maneuverable
    object2_maneuverable


------------------------------------------------------------
2. FILE: rl_states.npy
------------------------------------------------------------
This file contains the final RL state vectors.

Shape:
(185511, 9)

State vector structure:
[
    miss_distance,
    relative_speed,
    rel_pos_mag,
    rel_vel_mag,
    collision_probability,
    collision_max_probability,
    cluster,
    dbscan_label,
    anomaly_flag
]

Load using:
    import numpy as np
    states = np.load("rl_states.npy")


------------------------------------------------------------
3. FILE: high_risk_cases.csv
------------------------------------------------------------
A filtered list of high-risk or anomalous conjunction events.

Criteria:
- collision_max_probability > 0.001  
  OR
- anomaly_flag == 1

Use cases:
- Reward shaping
- Evaluating edge scenarios
- Maneuver optimization stress testing


------------------------------------------------------------
4. RL ENGINEER NOTES
------------------------------------------------------------
Recommended RL definition:

State:
    The 9-element vector from rl_states.npy

Example action space:
    0 = no maneuver
    1 = small maneuver
    2 = large maneuver

Reward design ideas:
- Reward reduction of risk  
- Penalize unnecessary maneuvers  
- Larger penalty for failing to avoid dangerous encounters  
- Possibly incorporate fuel usage  

The RL engineer can freely adjust:
    - policy architecture
    - reward function
    - action definitions
    - transition dynamics

------------------------------------------------------------
END OF DOCUMENT
------------------------------------------------------------
