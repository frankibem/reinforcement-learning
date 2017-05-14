## CartPole
This section compares the performance of recent state of the art algorithms when used to solve the CartPole environment from OpenAI gym. CartPole was chosen for its simplicity - the goal is to balance a pole on a moving cart. There are 2 possible actions - moving left and right - and the game ends when the pole is more than 15 degress from the vertical or the cart moves more than 2.4 units away from the center.

The algorithms and enhancements covered include:
* Q-learning using a neural network
* Deep Q-Networks (DQN)
  * Using a target network
  * Using experience replay
* Double Deep Q-Networks (DDQN)
* Prioritized Experience Replay

As I build up to the more advanced algorithms, I discuss the shortcoming of the previous and how the next solves them. The recommended reading order for the notebooks is:
* q_learning
* target_network
* experience_replay
* dqn
* ddqn
* prioritized_experience_replay
