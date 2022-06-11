# NCTU-Introduction-to-AI-2022-final
## Overview  
Playing the [Atari Breakout](https://www.gymlibrary.ml/environments/atari/breakout/)  game on openAI gym and trained it with DQN and Double-DQN.

![gameplay](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Images/DQNep45000.gif)
## Requirements
The conda environment requirements are listed in requirements.txt.

Because I havn't add the condition to configure the device of the network, so a cuda device is also needed.
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
## Net Structure
![Net Structure](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Images/Net%20Structure.jpg)
## Experiment Results
Training Curve:
![Training Curve](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Graphs/compare.png)

Average tile broke per life in testing:
![Test Curve](https://github.com/c1uc/NCTU-Introduction-to-AI-2022-final/blob/master/Graphs/compare_test.png)
