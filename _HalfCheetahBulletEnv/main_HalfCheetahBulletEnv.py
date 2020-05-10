# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from stable_baselines import PPO2
import numpy as np
import pybullet_envs
import time

print(np.version.version)

def main():
    env = gym.make("HalfCheetahBulletEnv-v0")
    env.render(mode="human")

    # the model was trained for 2e6 steps with the best hyperperams in a notebook on google colab
    model = PPO2.load(currentdir + '/PPO2_HalfCheetahBulletEnv')
    # disable rendering during reset, makes loading much faster
    env.reset()

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()
        while 1:
            time.sleep(1. / 60.)
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            score += reward
            frame += 1
            still_open = env.render("human")
            if still_open == False:
                return
            if not done: continue
            if restart_delay == 0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60 * 2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay == 0: break


if __name__ == "__main__":
    main()
