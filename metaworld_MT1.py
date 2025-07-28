# import gymnasium as gym
import metaworld

# seed = 3  # some seed number here
# env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=seed)
# obs, info = env.reset()

# a = env.action_space.sample() # randomly sample an action
# obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action

import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers import TimeLimit


training_period = 10 
seed = 3  # some seed number here
env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=seed, render_mode="rgb_array")
env = TimeLimit(env, max_episode_steps=200)
env = RecordVideo(env, video_folder="/Users/SoumyaJailwala/Desktop/RL_Project/MT1_reach_videos", name_prefix="reach_vid", 
                  episode_trigger=lambda x: x % training_period == 0)
env = RecordEpisodeStatistics(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# model = PPO("MlpPolicy", "CartPole-v1").learn(10_000)
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
env.close()

