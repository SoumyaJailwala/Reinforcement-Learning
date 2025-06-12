import dm_control2gym

env_name = "cartpole"


dm_control2gym.create_render_mode("human", show=True, return_pixel=True, height=240, width=320, camera_id=-1, overlays=(),
             depth=False, scene_option=None)
# make the dm_control environment
env = dm_control2gym.make(domain_name=env_name, task_name="swing_up")

# use same syntax as in gym
env.reset()
for t in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    env.render()

