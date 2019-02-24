from agent import Agent
from monitor import interact
import gym
import numpy as np
import sys

update_method = None
if (len(sys.argv) > 1):
    if(sys.argv[1] == "SARSA"):
        update_method = Agent.UpdateMethod.SARSA
    elif(sys.argv[1] == "SARSA_MAX"):
        update_method = Agent.UpdateMethod.SARSA_MAX
    elif(sys.argv[1] == "EXPECTED_SARSA"):
        update_method = Agent.UpdateMethod.EXPECTED_SARSA
    else:
        raise ValueError("update method should be either: SARSA, SARSA_MAX, or EXPECTED_SARSA")
else:
    update_method = Agent.UpdateMethod.SARSA_MAX

        
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
