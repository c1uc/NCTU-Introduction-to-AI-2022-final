# NCTU-Introduction-to-AI-2022-final
## Overview  
Training an agent based on [Atari Breakout](https://www.gymlibrary.ml/environments/atari/breakout/)  game on openAI gym it with CNN+DQN and CNN+Double-DQN.

![gameplay](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Images/DQNep45000.gif)
## Requirements
The conda environment requirements are listed in requirements.txt.

Because I havn't added the condition to configure the device of the network, so a cuda device is also needed.

Installing Atari ROMS can refer to this [link](https://github.com/openai/atari-py#roms).
## Environment
BreakoutNoFrameskip-v4 wrapped with:
- max_episode_steps=10000
- episodic_live=True
- clip_rewards=True
- frame_stack=True

Wrappers are copied from [openai baselines](https://github.com/openai/baselines/tree/master/baselines/common).
## Hyperparameters
- learning rate=0.00025
- gamma=0.99
- batch size=32
- replay buffer size=20000
- start eps=1.0
- end eps=0.1
- Explore steps=50000
- Observe steps=1000000
- total episodes=50000
## Other Specification
- Optimizer: RMSprop
- Train the net every 4 Steps
- (DDQN) Update target network every 5000 steps
## Net Structure
![Net Structure](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Images/Net%20Structure.jpg)
## Experiment Results
### Train
Training Curve:
![Training Curve](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Graphs/compare.png)
### Test
#### DQN
Average Unclipped reward per episode: 38.75

Max Unclipped reward per episode: 59

Average Clipped reward in 100 lives: 
![](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Graphs/DQN_test.png)
#### DDQN
Average Unclipped reward per episode: 14.75

Max Unclipped reward per episode: 18

Average Clipped reward in 100 lives: 
![](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Graphs/DDQN_test.png)

Average Clipped reward when end eps=0.005 in 100 lives:
![](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Graphs/DDQN_with_end_eps_0_005_test.png)
#### Compare
Average Clipped reward in 100 lives:
![Test Curve](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Graphs/compare_test.png)

