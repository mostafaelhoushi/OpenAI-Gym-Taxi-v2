from agent import Agent
from monitor import interact
import gym
import numpy as np

overall_avg_reward = 0
N_trials = 100

for i in range(0, N_trials):
    print("Trial #:" + str(i))
    env = gym.make('Taxi-v2')
    agent = Agent()
    avg_rewards, best_avg_reward = interact(env, agent)
    overall_avg_reward += best_avg_reward
    
overall_avg_reward = overall_avg_reward / N_trials
    
print("Overall avg reward over " + str(N_trials) + " trials is: " + str(overall_avg_reward))
