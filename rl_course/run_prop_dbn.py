from pyRDDLGym.core.env import RDDLEnv
import numpy as np

env = RDDLEnv(domain="./prop_dbn/domain.rddl", instance="./prop_dbn/instance.rddl")

state = env.reset()
total_reward = 0

for t in range(env.horizon):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    total_reward += reward
    print(f"Step {t}: action = {action}, reward = {reward}")

print("Total reward:", total_reward)
env.close()
