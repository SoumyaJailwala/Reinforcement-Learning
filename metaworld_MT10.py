# # import gymnasium as gym
# import metaworld
# import metaworld.envs


# # seed = 3  # some seed number here
# # env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=seed)
# # obs, info = env.reset()

# # a = env.action_space.sample() # randomly sample an action
# # obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action

# import gymnasium as gym
# from stable_baselines3 import PPO
# from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
# from gymnasium.wrappers import TimeLimit

# # print("MT10 registered?", "Meta-World/MT10" in gym.envs.registry)
# # spec = gym.spec("Meta-World/MT10")
# # print("entry_point:", spec.entry_point)
# # print("vector_entry_point:", spec.vector_entry_point)
# # print("vector_env_ok?", spec.vector_entry_point is not None)
# # import sys
# # print("Python:", sys.executable)
# # print("Meta-World/MT10" in gym.envs.registry.keys())  # should print: True
# # print(gym.spec("Meta-World/MT10").vector_entry_point) 

# training_period = 10 
# seed = 3  # some seed number here
# env = gym.make_vec('Meta-World/MT10', seed=45, vector_strategy='sync', render_mode="rgb_array")

# env = RecordVideo(env, video_folder="/Users/SoumyaJailwala/Desktop/RL_Project/MT10_videos", name_prefix="MT10_vid", 
#                   episode_trigger=lambda x: x % training_period == 0)
# env = RecordEpisodeStatistics(env)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000)

# # model = PPO("MlpPolicy", "CartPole-v1").learn(10_000)
# obs, _ = env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     if terminated or truncated:
#         obs, _ = env.reset()
# env.close()







import gymnasium as gym
import metaworld

from stable_baselines3 import PPO
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, TimeLimit

training_period = 10
seed = 3

# Use Meta-World's built-in MT10 environment and sample a task
env = gym.make('Meta-World/MT10', seed=seed, render_mode="rgb_array")
env = TimeLimit(env, max_episode_steps=200)
env = RecordVideo(
    env,
    video_folder="/Users/SoumyaJailwala/Desktop/RL_Project/MT10_videos",
    name_prefix="MT10_vid",
    episode_trigger=lambda x: x % training_period == 0
)
env = RecordEpisodeStatistics(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
