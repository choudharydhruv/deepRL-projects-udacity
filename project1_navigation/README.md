[//]: # (Image References)

[image1]: https://github.com/choudharydhruv/deepRL-projects-udacity/blob/master/project1_navigation/plots/demo.gif "Trained Agent"

[Report]: https://github.com/choudharydhruv/deepRL-projects-udacity/blob/master/project1_navigation/Report.md

# Project 1: Navigation

### Introduction

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the file in the GitHub repository, in the `project1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

##### Training an agent and model structure
Follow the instructions in `Navigation.ipynb` to get started with training the agent!  

1. The environment is considered solved if we  receive an average reward (over 100 episodes) of at least +13. 

2. We have implemented two important classes Agent() `agent.py` and DQNetwork() in `model.py`.

3. There is a dqn_run() function that takes in all the hyperparameters and config options to train the agent.

##### Evaluation

Finally we plot two curves: 

1. On-policy average reward for 3 algorithms
    
    a) Deep Q-Learning
    
    b) Double Deep Q-Learning
    
    c) Double Deep Q-Learning with Duelling Network (DuelingDQNetwork() in model.py)
    
2. Off policy average reward over 100 episodes

For the final evaluation report please look at the [Report]


