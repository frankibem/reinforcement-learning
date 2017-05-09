# Reinforcement Learning
This repository is a ongoing culmination of my efforts to understand the various algorithms used to train reinforcement learning (RL) agents to perform various tasks. It is meant to be gentle guide for others who may wish to explore the world of RL as well. Following the content in the order listed below should be the fastest way to get up to speed. Once done, see [here](https://github.com/frankibem/kung-fu-master) for a medium scale project which combines several rl concepts to solve an interesting problem.

## Content
### 1. Bandits
This section uses the k-armed bandit problem discussed in [1] to introduced several important RL concepts. I recreate some of the experiments to show that I can obtain similar results and solve some of the exercises. Concepts covered include:
* Value functions
* Epoch-greedy action selection
* Balancing exploration and exploitation
* Upper confidence bounds
* Stationary vs non-stationary problems

### 2. Temporal Difference
This section covers temporal difference methods for RL and demonstrates their performance using examples and exercises from [1].
* SARSA
* Q-Learning

### 3. CartPole
This section introduces some of the new state of the art techniques for solving RL problems. To compare between them, I've opted to use the CartPole enviroment for its simplicity (from OpenAI's Gym).
* Q-Learning with a neural network
* Deep Q-Networks
  * Experience Replay
  * Target Networks
* Double Deep Q-Networks
* Prioritized Experience Replay


## Dependencies
1. Python 3.5
2. numpy
3. matplotlib
4. pandas
5. OpenAI Gym
6. CNTK**

** I've chosen to implement the more complex algorithms using CNTK because very few of such implementations exist and it would force me to understand the little details.

## Resources
### Textbooks:
[1] [Reinforcement Learning: An Introduction by R. Sutton and A. Barto](http://incompleteideas.net/sutton/book/bookdraft2016sep.pdf)


### Papers:
* [Human-level control through deep reinforcement learning](https://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461v3.pdf)
* [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)

### Articles
* [Let's make a DQN](https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/)
* [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
* [Using Keras and Deep Q-Network to Play FlappyBird](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html)
