import metaworld

# List all environments in Meta-World
env_names = metaworld.ML1.ENV_NAMES  # For MT1 / ML1 tasks
print("ML1 / MT1 tasks:", sorted(env_names))

env_names = metaworld.MT10.ENV_NAMES  # For MT10 tasks
print("MT10 tasks:", sorted(env_names))

env_names = metaworld.MT50.ENV_NAMES  # For MT50 tasks
print("MT50 tasks:", sorted(env_names))
