# Super Mario RL Agent

This project implements a reinforcement learning agent that learns to play Super Mario Bros using the PPO algorithm from the Stable Baselines3 library. The environment is provided by the `gym_super_mario_bros` library.

## Installation

To run this project, you will need to install the following dependencies:

- Python 3.6 or higher
- gym_super_mario_bros
- stable-baselines3
- nes_py
- torch
- torchvision
- torchaudio

You can install the dependencies using the following commands:

```
!pip3 install torch torchvision torchaudio
!pip install gym_super_mario_bros stable-baselines3[extra] nes_py
```

## Usage

To run the Super Mario RL agent, simply execute the `model.py` script:

```
python model.py
```

This will train the agent for 5000 steps and then render the environment as the agent plays. You can modify the number of training steps by changing the `total_timesteps` parameter in the `model.learn()` function call.

## Acknowledgements

This project was inspired by the article ["Reinforcement learning in Super Mario bros"](https://dev.to/akilesh/reinforcement-learning-in-super-mario-bros-56i9) by Akilesh.