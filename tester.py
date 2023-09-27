import time

import gymnasium as gym
import pygame

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import grid_world_env.envs.grid_world


vec_env = make_vec_env("GridWorld-v0", n_envs=1)

model = A2C("MultiInputPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("a2c_gridworld")


vec_env = model.get_env()
obs = vec_env.reset()

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_gridworld")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    time.sleep(0.2)

vec_env.close()
