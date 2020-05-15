from myutils.model_loader import load_model_and_env_from_rlbz

import os

# Note: To use box2d envs, you need to install box2d box2d-kengz (pip) and swig (apt-get)
# box2d envs include BipedalWalker-v3, LunarLander-v2, ...

# change working directory for access to folder rl-baselines-zoo
import os
print(os.path.abspath(os.curdir))
os.chdir("..")
print(os.path.abspath(os.curdir))

ENV_ID = 'BipedalWalker-v3'

model, env = load_model_and_env_from_rlbz('ppo2', 'rl-baselines-zoo/trained_agents/', ENV_ID, log_dir='log_dir/')

episode_reward = 0
obs = env.reset()
rgb_arrays = []
for _ in range(10000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    #rgb_array = env.render(mode='rgb_array')
    #rgb_arrays.append(rgb_array)
    if done:
        print(episode_reward)
        episode_reward = 0
