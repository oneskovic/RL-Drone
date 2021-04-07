import numpy as np
import pygame
from DroneEnv import DroneEnv
from time import sleep

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import datetime

# Separate evaluation env
eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(DroneEnv())]), norm_obs=False, norm_reward=True)
timestamp = datetime.datetime.now().strftime('%m-%d-%Y %H-%M-%S')
eval_callback = EvalCallback(eval_env, n_eval_episodes=10, best_model_save_path=f'./training/{timestamp}/best_model',
                             log_path=f'./training/{timestamp}/best_model', eval_freq=5000,
                             deterministic=True, render=True)
checkpoint_callback = CheckpointCallback(save_freq=2500, save_path=f'./training/{timestamp}/all_models')


env = VecNormalize(DummyVecEnv([lambda: Monitor(DroneEnv())]), norm_obs=False, norm_reward=True)
n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#model = TD3.load('training/04-06-2021 19-29-14/all_models/rl_model_1000000_steps.zip', env=env)
model = TD3("MlpPolicy", env, learning_rate=1e-3, buffer_size=200000,
            learning_starts=10000, gamma=0.98, action_noise=action_noise,
            policy_kwargs=dict(net_arch=[400, 300]), verbose=1)

model.learn(total_timesteps=1000000, log_interval=10, callback=[eval_callback, checkpoint_callback])

for i in range(10):
    obs = env.reset()
    done = False
    total_rew = 0.0
    while not done:
        # action = np.random.uniform(0.0,5.0,(2,))
        #action = [9.81*5*0.5,9.81*5*0.5]
        #action = np.array(action, dtype=np.float32)
        action, KURAC = model.predict(obs)
        obs, reward, done, kurac = env.step(action)
        total_rew += reward
        #print(reward)
        env.render()
        # sleep(0.1)
    print(total_rew)