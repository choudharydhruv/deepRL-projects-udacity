[//]: # (Image References)

[image1]: https://github.com/choudharydhruv/deepRL-projects-udacity/blob/master/project3_collaborate_compete/plots/demo.gif "Trained Agent"

[image2]: https://github.com/choudharydhruv/deepRL-projects-udacity/blob/master/project3_collaborate_compete/plots/MADDPG_rewards.png

# Project 3: Collaborate and Compete

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Learning Algorithm

We use the Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (Lowe and Wu) to solve this environment. DDPG is a actor-crtic method that uses a an actor to explicitly learn the policy and a critic that estimates the value for a given state and action. Both actor and critic use a target network to make the learning tractable. A multi agent DDPG has multiple agents where actors for the agents can observe the local observation space to produce actions, while critics see the whole observation space as well as the actions taken by other agents. After the learning is done the agents use their respective actors to observe the state and take local actions. Hence the training is centralized but the execution is decentralized.

**Epsilon Decay** : The algorithm needs a lot more exploration in the beginning even after we use Ornstein-Uhlenbeck process noise. Hence we used an epsilon parameter that gets discounted with every learning step. We start the epsilon from a very high value of 10 and then decay over 4000 iterations.

**Soft Updates** : We also used a significantly higher value of softupdate rate for this project. We set the softupdate rate (tau) to 1e-2. This was determined through cross validation.

### Network Architecture

**Actor**: We use a neural net with 2 Fully Connected (FC) layers with 256 and 64 neurons. Input state space has a dimensionality of 24  and output actions space has 2 dimensions for each agent. The output nodes have a non linearity of tanh to bound them to -1,1 range.

**Critic**: The first hidden layer maps from state size to 256, the second layer takes a concatenated output of first layer and the action vectors for all agents and maps that to 64 neurons. The output for the critic is a single dimension (the action value). We use network initializations identical to the original DDPG paper.

**Batch Normalization**: Additionaly we use batch normalization which is a common technique to make learning tractable. We found that unlike the previous two projects, the learning for MADDPG does not converge without batch normalization. We apply batch normalization after all layers. We also used a slightly higher batch size of 128 to make the learning more stable.

### Evaluation

We used the following hyperparamenters for both single agent and multi agent training:

1. Learning Rate: Actor and critic both use 1e-3 (Same as DDPG paper)
2. Batch Size 128 (Chosen by cross validation although 64 performs just as well)
3. Buffer Size 1e6 for each agent.
4. OUNoise variance 0.15.
5. Softupdate rate (tau) = 1e-2
6. Reward discount (gamma) = 0.99

Final reward achieved over 702 episodes is 0.5.

![Plot of rewards][image2]

The training was very sensitive to a few parameters
a) Using Batch Normalization(BN):  BN for the Actors was extremely important, otherwise the learning was intractable. BN for Critics was less important although that helps as well.
b) Epsilon Decay iterations were also quite important. Decaying over 500-1000 iterations performs worse so we choose decay over 4000 iterations even though the environment gets solved in less than 1000 iterations.
c) Learning multiple times -  In the begining the agent learns slow because lack of good examples, but after 200-300 episodes the agent starts learning faster. Increasing the number of learning iterations at this point can speed up learning further. Hence after 200 episodes, for each agent step we take 5 learning steps thereby sampling even more experiences from the replay buffer. 


### Trained Weights

The final trained weights are stored in `checkpoints/agentX_actor_local_weights.pth`

### Future Work

1. We would like to test the convergence speed of more advanced algorithms like  D4PG.

2. We would like to try Prioritized Experience Replay. In the beginning learning is very slow and it picks up after it has good examples in the buffer. This shows that applying PER in the beginning can be beneficial.