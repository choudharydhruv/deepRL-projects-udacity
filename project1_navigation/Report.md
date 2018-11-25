[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: https://github.com/choudharydhruv/deepRL-projects-udacity/blob/master/project1_navigation/plots/DQN_rewards.png

[image3]: https://github.com/choudharydhruv/deepRL-projects-udacity/blob/master/project1_navigation/plots/Offpolicy_rewards.png

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

We use Deep Q-Learning(DQN) to solve this environment. DQN is a value based method that uses a neural network to approximate the action-value function for a given state. The Neural network takes the state as an input and estimates the expected average reward of taking one of the possible actions in that state. DQN learns expected reward using a target and local network.

Additionally, we also implement and evaluate two variants of DQN 

    a) Double Deep Q-Learning (DDQN) which uses the local network to select the next state action and the target network to evalauate that action. DDQN reduces the variance of the estimator because the target reward depends on both the local and target networks.
    
    b) Dueling Network - Dueling network is a different network architecture that has two heads - one estimates the state-value function and the other estimates the advantage of taking an action.

###### Network Architecture
We use a neural net with 2 Fully Connected (FC) layers with 128 and 64 neurons. Input state space has a dimensionality of 37 and output actions space has 4 dimensions. Hence the networ architecture is 37->128->64->4.

For the Dueling Network, we keep the architecture for the advantage function the same as DQN, but we add an additional FC layer for the state value function. The first layer is common to both the branches.

### Evaluation

![Plot comparing on-policy rewards][image2]

![Off-policy rewards for DQN][image2]

### Future Work



3. There is a dqn_run() function that takes in all the hyperparameters and config options to train the agent.

##### Evaluation

Finally we plot two curves: 

1. On-policy average reward for 3 algorithms
    
    a) Deep Q-Learning
    
    b) Double Deep Q-Learning
    
    c) Double Deep Q-Learning with Duelling Network (DuelingDQNetwork() in model.py)
    
2. Off policy average reward over 100 episodes


