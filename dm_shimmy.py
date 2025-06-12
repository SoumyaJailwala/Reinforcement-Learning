from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers import TimeLimit

from shimmy.registration import DM_CONTROL_SUITE_ENVS
# env_ids = [f"dm_control/{'-'.join(item)}-v0" for item in DM_CONTROL_SUITE_ENVS]
# print(env_ids)


training_period = 10  # record the agent's episode every __
# num_training_episodes = 10_000  # total number of training episodes

env = gym.make("dm_control/ball_in_cup-catch-v0", render_mode="rgb_array")
env = TimeLimit(env, max_episode_steps=250)
env = RecordVideo(env, video_folder="/Users/SoumyaJailwala/Desktop/RL_Project/ball_videos", name_prefix="ball_vid",
                  episode_trigger=lambda x: x % training_period == 0)
env = RecordEpisodeStatistics(env)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=300_000)

# Step 4: Run trained policy
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()
env.close()
