import os
import numpy as np
import pandas as pd
from myutils.model_loader import load_model_and_env_from_rlbz

# Note: To use box2d envs, you need to install box2d box2d-kengz (pip) and swig (apt-get)
# box2d envs include BipedalWalker-v3, LunarLander-v2, ...

# change working directory for access to folder rl-baselines-zoo
print('Current working dir: {}'.format(os.path.abspath(os.curdir)))
os.chdir("..")
print('Changed working dir to: {}'.format(os.path.abspath(os.curdir)))

data_dir = 'data/'

ENV_ID = 'BipedalWalker-v3'
NUM_STEPS = 1000

model_names = ['a2c',
               # 'acer',
               'acktr',
               # 'ddpg', # exists
               # 'dqn',
               # 'her',
               'ppo2',
               # 'sac', # exists
               'td3',
               # 'trpo' # exists
               ]


for model_name in model_names:
    model, env = load_model_and_env_from_rlbz(model_name, 'rl-baselines-zoo/trained_agents/', ENV_ID,
                                              log_dir='log_dir/')

    num_obs = len(env.observation_space.sample())
    obs_all = np.zeros(shape=(NUM_STEPS, num_obs))

    num_actions = len(env.action_space.sample())
    actions_all = np.zeros(shape=(NUM_STEPS, num_actions))

    rewards_all = np.zeros(shape=(NUM_STEPS, 1))
    done_all = np.zeros(shape=(NUM_STEPS, 1))
    cumulative_rewards = np.zeros(shape=(NUM_STEPS, 1))

    episode_reward = 0
    obs = env.reset()
    for i in range(NUM_STEPS):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            print(episode_reward)
            episode_reward = 0

        # record all the datapoints
        obs_all[i] = obs
        actions_all[i] = action
        rewards_all[i] = reward
        done_all[i] = done
        cumulative_rewards[i] = episode_reward

    np_all = np.column_stack([done_all, rewards_all, cumulative_rewards, obs_all, actions_all])

    columns = ['done', 'rewards', 'cumulative_reward']
    columns.extend(['obs_{}'.format(i) for i in range(num_obs)])
    columns.extend(['action_{}'.format(i) for i in range(num_actions)])
    df_all = pd.DataFrame(np_all, columns=columns)

    dir_path = os.path.join(data_dir, model_name)
    data_path = os.path.join(dir_path, '{}_{}.csv'.format(model_name, ENV_ID))
    os.makedirs(dir_path, exist_ok=True)
    df_all.to_csv(data_path, index=False)
