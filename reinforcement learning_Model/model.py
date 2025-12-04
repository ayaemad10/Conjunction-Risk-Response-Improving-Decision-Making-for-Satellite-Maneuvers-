import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import json
import os

class SatelliteManeuverEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df):
        super().__init__()
        self.df = df.copy() # Work on a copy to avoid modifying original df
        self.num_conjunctions = len(df)
        self.max_steps_per_episode = 10 # Define maximum steps to simulate a single conjunction event

        # Handle NaN in 'object1_maneuverable' by filling with 'N'
        if 'object1_maneuverable' in self.df.columns:
            self.df['object1_maneuverable'] = self.df['object1_maneuverable'].fillna('N').astype(str)
        else:
            # If column doesn't exist, assume not maneuverable for all
            self.df['object1_maneuverable'] = 'N'

        # Define observation space based on relevant columns
        feature_columns = ['miss_distance', 'relative_speed',
                         'relative_position_r', 'relative_position_t', 'relative_position_n',
                         'relative_velocity_r', 'relative_velocity_t', 'relative_velocity_n',
                         'collision_probability', 'collision_max_probability']

        # Calculate bounds and explicitly cast to float32 to avoid UserWarning
        low_bounds = self.df[feature_columns].min().values.astype(np.float32)
        high_bounds = self.df[feature_columns].max().values.astype(np.float32)

        # Ensure logical bounds for specific features
        low_bounds[0] = 0.0 # miss_distance cannot be negative
        low_bounds[1] = 0.0 # relative_speed cannot be negative
        low_bounds[8] = 0.0 # collision_probability min is 0
        low_bounds[9] = 0.0 # collision_max_probability min is 0
        high_bounds[8] = 1.0 # collision_probability max is 1
        high_bounds[9] = 1.0 # collision_max_probability max is 1

        # Ensure all values are finite before creating the Box space, converting inf to float32 max/min
        low_bounds = np.nan_to_num(low_bounds, nan=-np.inf, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        high_bounds = np.nan_to_num(high_bounds, nan=np.inf, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

        self.observation_space = gym.spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # Define action space:
        # 0: no maneuver
        # 1-2: delta-v in R (positive/negative)
        # 3-4: delta-v in T (positive/negative)
        # 5-6: delta-v in N (positive/negative)
        self.delta_v_magnitude = 0.01 # Adjusted example magnitude in km/s (smaller, more realistic)
        self.action_space = gym.spaces.Discrete(7) # 7 discrete actions

        # Internal state that gets updated within an episode
        self._current_state = None
        self._current_conjunction_idx = None # Index of the conjunction from df for the current episode
        self._steps_taken_in_episode = 0

    def _get_obs(self):
        # Return a copy to prevent external modification of internal state
        return self._current_state.copy()

    def _get_info(self):
        # Ensure 'object1_maneuverable' is a string 'Y' or 'N'
        maneuverable_status = self.df.iloc[self._current_conjunction_idx]['object1_maneuverable']
        return {'object1_maneuverable': str(maneuverable_status)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly select a conjunction event from the DataFrame to start a new episode
        self._current_conjunction_idx = self.np_random.integers(self.num_conjunctions)
        self._steps_taken_in_episode = 0

        # Initialize the internal mutable state for the episode
        current_event_data = self.df.iloc[self._current_conjunction_idx]

        feature_columns = ['miss_distance', 'relative_speed',
                         'relative_position_r', 'relative_position_t', 'relative_position_n',
                         'relative_velocity_r', 'relative_velocity_t', 'relative_velocity_n',
                         'collision_probability', 'collision_max_probability']
        self._current_state = current_event_data[feature_columns].values.astype(np.float32)

        # Ensure probabilities are within [0, 1] bounds, and other physical bounds
        self._current_state[8] = np.clip(self._current_state[8], 0.0, 1.0) # collision_probability
        self._current_state[9] = np.clip(self._current_state[9], 0.0, 1.0) # collision_max_probability
        self._current_state[0] = np.maximum(self._current_state[0], 0.0) # miss_distance
        self._current_state[1] = np.maximum(self._current_state[1], 0.0) # relative_speed

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self._steps_taken_in_episode += 1
        terminated = False
        truncated = False
        reward = 0.0

        maneuverable = self._get_info()['object1_maneuverable'] == 'Y'

        # Apply delta-v based on action if object1 is maneuverable
        if maneuverable and action > 0:
            # A temporary copy of velocity components from the current internal state
            relative_velocity = self._current_state[5:8].copy() # Indices for r_v, t_v, n_v

            if action == 1: # +R direction (reduce current R component)
                relative_velocity[0] -= self.delta_v_magnitude
            elif action == 2: # -R direction (increase current R component)
                relative_velocity[0] += self.delta_v_magnitude
            elif action == 3: # +T direction (reduce current T component)
                relative_velocity[1] -= self.delta_v_magnitude
            elif action == 4: # -T direction (increase current T component)
                relative_velocity[1] += self.delta_v_magnitude
            elif action == 5: # +N direction (reduce current N component)
                relative_velocity[2] -= self.delta_v_magnitude
            elif action == 6: # -N direction (increase current N component)
                relative_velocity[2] += self.delta_v_magnitude

            # Update the internal state with new relative velocities
            self._current_state[5:8] = relative_velocity

            # --- Simplified Heuristic for propagation effect of maneuver ---
            # This is NOT physically accurate but demonstrates how action could change collision risk.
            # In a real system, an orbital propagator would re-calculate miss_distance and collision_probability.

            current_miss_dist = self._current_state[0]
            current_prob = self._current_state[8]

            # A very simplistic model: small delta-v directly improves miss distance and reduces probability
            # Factor could be tuned or derived from a more complex model
            miss_dist_increase_per_delta_v = 1000.0 # Example: 1000m increase per km/s delta_v
            prob_decrease_factor_per_delta_v = 0.5 # Example: reduce probability by 50% per delta_v

            # Impact of maneuver on miss_distance and collision_probability
            # The effect is proportional to delta_v_magnitude
            self._current_state[0] += miss_dist_increase_per_delta_v * self.delta_v_magnitude # Increase miss distance
            self._current_state[8] *= (1.0 - prob_decrease_factor_per_delta_v * self.delta_v_magnitude) # Reduce collision probability
            self._current_state[9] *= (1.0 - prob_decrease_factor_per_delta_v * self.delta_v_magnitude) # Reduce max collision probability

            # Clip probabilities to valid range [0, 1]
            self._current_state[8] = np.clip(self._current_state[8], 0.0, 1.0)
            self._current_state[9] = np.clip(self._current_state[9], 0.0, 1.0)

            # Penalize for maneuvering (action > 0)
            reward -= 0.1 # Small penalty for fuel consumption

        # Define reward based on the *updated* internal state
        current_prob = self._current_state[8]
        current_miss_dist = self._current_state[0]

        safe_miss_distance_threshold = 100.0 # km
        critical_miss_distance_threshold = 5.0 # km
        safe_prob_threshold = 1e-7 # Very low probability
        high_prob_threshold = 1e-4 # High risk probability

        if current_prob < safe_prob_threshold and current_miss_dist > safe_miss_distance_threshold:
            reward += 10.0 # Significant positive reward for very safe state
            terminated = True
        elif current_prob > high_prob_threshold or current_miss_dist < critical_miss_distance_threshold:
            reward -= 50.0 # Large negative penalty for high risk/collision
            terminated = True
        else:
            # Continuous reward based on inverse probability and miss distance
            # Higher miss_distance is good, lower probability is good
            # Ensure no division by zero for current_prob
            reward += (1.0 / (current_prob + 1e-12)) * 1e-3 + (current_miss_dist / 1000.0) # Scale appropriately

        # Episode termination conditions
        if self._steps_taken_in_episode >= self.max_steps_per_episode:
            truncated = True # Episode ends due to time limit

        # Ensure observation values are within defined bounds after all updates
        self._current_state[0] = np.clip(self._current_state[0], self.observation_space.low[0], self.observation_space.high[0])
        self._current_state[1] = np.clip(self._current_state[1], self.observation_space.low[1], self.observation_space.high[1])
        # Clip relative positions and velocities too, though they are usually unbounded in Box space
        for i in range(2, 8):
            self._current_state[i] = np.clip(self._current_state[i], self.observation_space.low[i], self.observation_space.high[i])
        self._current_state[8] = np.clip(self._current_state[8], self.observation_space.low[8], self.observation_space.high[8])
        self._current_state[9] = np.clip(self._current_state[9], self.observation_space.low[9], self.observation_space.high[9])

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# Function to load model and make predictions
def load_and_predict(model_path, df, config_path="agent_modules.json", num_eval_episodes=5):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Instantiate the environment
    env = SatelliteManeuverEnv(df)

    # Load the trained model
    loaded_model = PPO.load(model_path, env=env)

    episode_rewards = []
    for i in range(num_eval_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        episode_steps = 0

        print(f"\n--- Starting Evaluation Episode {i+1} ---")
        print(f"Initial Observation: {obs}")
        print(f"Initial Info: {info}")

        while not done and not truncated:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            episode_steps += 1

        episode_rewards.append(total_reward)
        print(f"--- Evaluation Episode {i+1} Finished ---")
        print(f"Total steps taken: {episode_steps}")
        print(f"Total reward received: {total_reward:.2f}")
        print(f"Final Observation: {obs}")
        print(f"Episode terminated: {done}")
        print(f"Episode truncated: {truncated}")

    print(f"\nAverage reward over {num_eval_episodes} episodes: {np.mean(episode_rewards):.2f}")
    return loaded_model, np.mean(episode_rewards)

print("model.py created successfully with SatelliteManeuverEnv class and load_and_predict function.")
