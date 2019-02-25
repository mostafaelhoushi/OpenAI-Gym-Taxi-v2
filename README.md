# OpenAI-Gym-Taxi-v2
This small repo represents a re-inforcement solution to the Taxi problem in OpenAI Gym: https://github.com/openai/gym/wiki/Leaderboard#taxi-v2

## Steps to Run
1. Clone the repo:
`git clone https://github.com/mostafaelhoushi/OpenAI-Gym-Taxi-v2`

2. cd to the workspace directory:
`cd OpenAI-Gym-Taxi-v2/workspace`

3. Run the main script:
`python main.py`
You may add any of the following arguments when calling the above command to specify the update method: `SARSA`, `SARSA_MAX`, `EXPECTED_SARSA`.

## Source Code:
The repo contains three files in its `workspace` folder:

- `agent.py`: The code I develop the reinforcement learning agent is written here here. This is the only file that I have modified.
- `monitor.py`: The interact function tests how well the agent learns from interaction with the environment. This file has been provided by the creators of the Udacity Reinforcement Learning Nanodegree.
- `main.py`: The main file to run in the terminal to check the performance of the agent. This file has been provided by the creators of the Udacity Reinforcement Learning Nanodegree.

## Results:
The average of running 100 episodes for Sarsa Max (a.k.a. Q-Learning) is 9.2926, Expected Sara is 9.2754. 
