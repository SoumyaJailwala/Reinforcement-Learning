### necessary set-up:
# cloned Metaworld github
# ran mjx_setup.py in Metaworld folder
# in the new "cleaned" folder, go into the new xml and remove extra mujoco tags, and any mujoco-includes tags
# change any cylinder types to boxes (similar size)

# increased reward

import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model


import mujoco
from mujoco import mjx

mj_model = mujoco.MjModel.from_xml_path("Metaworld/metaworld/assets/cleaned/mjx_sawyer_reach.xml")
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
while mj_data.time < duration:
  mujoco.mj_step(mj_model, mj_data)
  if len(frames) < mj_data.time * framerate:
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)


media.write_video("mjx_reach_demo.mp4", frames, fps=framerate)
print("🎥 Saved video to: mjx_reach_demo.mp4")


from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp

class ReachEnv(PipelineEnv):
    def __init__(self, **kwargs):
        # Load your custom cleaned XML
        mj_model = mujoco.MjModel.from_xml_path(
            "Metaworld/metaworld/assets/cleaned/mjx_sawyer_reach.xml"
        )

        # Wrap as Brax sys
        sys = mjcf.load_model(mj_model)

        # MJX backend is required
        kwargs['backend'] = 'mjx'
        super().__init__(sys, **kwargs)


        self.hand_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self.goal_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self._reset_qpos = self.sys.qpos0
        self._reset_qvel = jp.zeros(self.sys.nv)

    def reset(self, rng: jp.ndarray) -> State:
        rng, key1, key2 = jax.random.split(rng, 3)
        noise = 0.02
        qpos = self._reset_qpos + noise * jax.random.uniform(key1, shape=self._reset_qpos.shape)
        qvel = self._reset_qvel + noise * jax.random.uniform(key2, shape=self._reset_qvel.shape)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state)
        return State(pipeline_state, obs, reward=jp.zeros(()), done=jp.zeros(()),  metrics={
        'dist': jp.array(1.0),        # default starting distance
        'reward': jp.array(0.0)       # total reward so far
        })


    def step(self, state: State, action: jp.ndarray) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(pipeline_state)

        # Reward: negative distance between hand and goal
        hand_pos = pipeline_state.site_xpos[self.hand_site_id]
        goal_pos = pipeline_state.site_xpos[self.goal_site_id]
        dist = jp.linalg.norm(hand_pos - goal_pos)

        prev_dist = state.metrics.get('dist', 1.0)
        reward = 10 * (prev_dist - dist)

        done = jp.where(dist < 0.05, 1.0, 0.0)  # float32 done
        metrics = {
        'dist': dist,
        'reward': reward  # or cumulative reward if you want
        }
        
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=jp.array(done),
            metrics=metrics
        )

    def _get_obs(self, pipeline_state) -> jp.ndarray:
        return jp.concatenate([
            pipeline_state.qpos,
            pipeline_state.qvel
        ])
    

from brax import envs

# Register your env with Brax
envs.register_environment('reach', ReachEnv)
env = envs.get_environment('reach')

import jax
import jax.numpy as jp

env = envs.get_environment('reach')

# JIT-compiled reset and step
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Reset
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

# Step with dummy action
action = jp.zeros(env.sys.nu)
state = jit_step(state, action)

print("obs shape:", state.obs.shape)
print("reward:", state.reward)
print("done:", state.done)


# Training progress tracking
x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
max_y, min_y = 5, -5  # adjust based on your rewards

def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    plt.figure(figsize=(8, 4))
    plt.clf()
    plt.xlim([0, 1.25 * max(x_data)])
    plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f"Reward: {y_data[-1]:.2f}")
    plt.errorbar(x_data, y_data, yerr=ydataerr)
    plt.grid(True)
    plt.pause(0.01)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=400_000,        # 🔽 smaller training run
    reward_scaling=1.0,
    episode_length=120,           # 🔽 shorter episodes
    normalize_observations=True,
    action_repeat=1,
    unroll_length=5,              # 🔽 less rollout length
    num_minibatches=4,            # 🔽 smaller update steps
    num_updates_per_batch=2,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=64,                  # 🔽 less parallel envs
    batch_size=64,                # 🔽 smaller batch size
    seed=0
)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)


# Recreate env just to be safe
eval_env = envs.get_environment('reach')

# JIT step/reset
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# Init state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

n_steps = 1500
for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    action, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, action)
    rollout.append(state.pipeline_state)




video = eval_env.render(rollout)  # or "side"
media.write_video("trained_policy_reach.mp4", video, fps=60)
print("🎥 Saved trained video to: trained_policy_reach.mp4")
