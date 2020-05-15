import os
import sys
import warnings

import stable_baselines
from stable_baselines.common import set_global_seeds

from myutils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# Fix for breaking change in v2.6.0
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer


def load_model_and_env_from_rlbz(algo, folder, env_id, n_envs=1, log_dir='', is_atari=False):
    """
    This function loads the pretrained model and environment with all set hyperparameters from the
    rl-baselines-zoo repository. Because the models and their hyperparams are not saved in a .zip format
    (due to an older version of stable-baselines being used; https://github.com/araffin/rl-baselines-zoo/issues/80),
    they cannot be loaded using the simple model.load(path) function.

    Example:
    model, env = load_model_and_env_from_rlbz('ppo2', 'rl-baselines-zoo/trained_agents/', ENV_ID, log_dir='content/')

    :param is_atari: (bool)
    :param algo: (string) algorithm name corresponding to the folder in rl-b-z
    :param folder: (string) directory containing the trained models in the rl-b-z
    :param env_id: (string) name of gym environment
    :param n_envs: (int) number of environments
    :param log_dir: (string) directory for logs saved in the environment
    :return: model and env
    """
    log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    model_path = find_saved_model(algo, log_path, env_id, load_best=False)

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        n_envs = 1

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, test_mode=True)

    env = create_test_env(env_id, n_envs=n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=42, log_dir=log_dir,
                          hyperparams=hyperparams)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    model = ALGOS[algo].load(model_path, env=load_env)

    return model, env


