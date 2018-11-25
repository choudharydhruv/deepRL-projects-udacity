[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

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

### Learning Algorithm

We use Deep Q-Learning(DQN) to solve this environment. DQN is a value based method that uses a neural network to approximate the action-value function for a given state. The Neural network takes the state as an input and estimates the expected average reward of taking one of the possible actions in that state. 

###### Network Architecture
We use a simple neural net

### Evaluation

##### Training an agent and model structure
Follow the instructions in `Navigation.ipynb` to get started with training the agent!  

1. The environment is considered solved if we  receive an average reward (over 100 episodes) of at least +13. 

2. We have implemented two important classes Agent() `agent.py` and DQNetwork() in `model.py`.

### Future Work



3. There is a dqn_run() function that takes in all the hyperparameters and config options to train the agent.

##### Evaluation

Finally we plot two curves: 

1. On-policy average reward for 3 algorithms
    
    a) Deep Q-Learning
    
    b) Double Deep Q-Learning
    
    c) Double Deep Q-Learning with Duelling Network (DuelingDQNetwork() in model.py)
    
2. Off policy average reward over 100 episodes


