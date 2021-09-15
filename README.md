# Udacity-DRL-Project-2
PPO Algorithm to solve Udacity DRL Project 2 Reacher Environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints.

The environment is considered solved once the average score across 100 episodes is >= 30.

# Running Code
The code is run entirely via the Jupyter Notebook `0_Continuous_Control_Alternative.ipynb`.

# Dependencies
The following is required to run this repository correctly:
- Python 3.6.1 or higher
- Numpy
- Matplotlib
- Unity ML-Agents Toolkit (https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Installation.md)
