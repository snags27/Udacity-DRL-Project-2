# Udacity-DRL-Project-2
PPO Algorithm to solve Udacity DRL Project 2 Reacher Environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints.

The environment is considered solved once the average score across 100 episodes is >= 30.

# Getting Started
1. Download the twenty agent reacher environment via one of the links below (select the correct operating system).
    
    Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
    Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip

2. Place the file in the same location of the repository files and unzip the contents (this should create a folder/path consistent with the path contained within `Report.ipynb`). 

# Running the Code
The code is run entirely within the Jupyter Notebook `Report.ipynb`, this file should be followed sequentially.

# Dependencies
The following is required to run this repository correctly:
- Python 3.6.1 or higher
- Numpy
- Matplotlib
- Unity ML-Agents Toolkit (https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Installation.md)
