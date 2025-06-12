from stable_baselines3 import PPO
from dm_control import suite
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import imageio


# Step 1: Wrap dm_control env
class DMCWrapper(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, domain="cartpole", task="swingup", render_mode=None):
        self._env = suite.load(domain_name=domain, task_name=task)
        self._action_spec = self._env.action_spec()
        self._obs_spec = self._env.observation_spec()
        self.render_mode = render_mode

        obs_dim = sum(np.prod(v.shape) for v in self._obs_spec.values())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self._action_spec.minimum,
            high=self._action_spec.maximum,
            shape=self._action_spec.shape,
            dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        return np.concatenate([v.ravel() for v in obs_dict.values()])

    def reset(self, seed=None, options=None):
        time_step = self._env.reset()
        return self._flatten_obs(time_step.observation), {}

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0.0
        done = time_step.last()
        return obs, reward, done, False, {}

    def render(self):
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.imshow(self._env.physics.render(camera_id=0))
            plt.axis("off")
            plt.pause(1 / 60)

    def close(self):
        pass


# Step 2: Use it like a Gym env
env = DMCWrapper("cartpole", "swingup", render_mode="human")

# Step 3: Train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Step 4: Run trained policy
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()

env.close()
