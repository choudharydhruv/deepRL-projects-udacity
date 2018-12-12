[//]: # (Image References)

[image1]: https://github.com/choudharydhruv/deepRL-projects-udacity/blob/master/project2_continous_control/plots/demo.gif "Trained Agent"

[image2]: https://github.com/choudharydhruv/deepRL-projects-udacity/blob/master/project2_continous_control/plots/DDPG_rewards.png

# Project 2: Continous Control

### Introduction

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers(action space=4), corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Learning Algorithm

We use Deep Deterministic Policy Gradient(DDPG) to solve this environment. DDPG is a actor-crtic method that uses a an actor to explicitly learn the policy and a critic that estimates the value for a given state and action. Both actor and critic use a target network to make the learning tractable

We attempted to solve both the single agent and multi agent environment.

###### Network Architecture

Actor: We use a neural net with 2 Fully Connected (FC) layers with 128 and 128 neurons. Input state space has a dimensionality of 33 and output actions space has 4 dimensions for 4 actions. Hence the networ architecture is 37->128->128->4. The output nodes have a non linearity of tanh to bound them to -1,1 range

Critic: 
The first hidden layer maps from state size to 128, the second layer takes a concatenated output of first layer and the action vector and maps that to 128 neurons. The output for the critic is a single dimension (the action value). We use network initializations identical to the original DDPG paper

### Evaluation

We used the following hyperparamenters for both single agent and multi agent training:

1. Learning Rate: Actor and critic both use 1e-4
2. Batch Size 128
3. Buffer Size for Shared memory  1e6
4. OUNoise variance 0.15 - we find that the training is sensitive to this noise variance. 0.15 seems to perform best.

Multi-agent: Final reward achieved over 200 episodes is 35.89.

Single-agent: Final reward achieved over 600 episodes is 9.16

![Plot comparing on-policy rewards][image2]


For the off-policy, we achieve an average of 35.59.

The training was very sensitive to a few parameters
a) Gradient clipping for the critic - This was very important otherwise the training does not converge.
b) The OUNoise decides the amount of exploration, but we do not decrease the exploration over time as we do with the DQN and hence in later stages the training becomes unstable. Redicing the noise variance to 0.15 performed much better overall.


### Trained Weights

The final trained weights are stored in `checkpoints/agent_ddpg_multi_actor_local_weights.pth`

### Future Work

1. We would like to test the convergence speed of more advanced algorithms like TRPO and D4PG.

2. Revisit the explore-exploit tradeoff to understand why noise variance makes the learning unstable.