# RL_Testbed

A basic testbed designed to implement and experiment with various reinforcement learning (RL) algorithms using Python and OpenAI's **Gym** environment.

## Features
- **Multiple RL algorithms**: Allows testing and comparison of various RL algorithms in a controlled environment.
- Currently **SAC** and **DDPG** is available. **A3C has memory leak issue**. 
- **OpenAI Gym integration**: Provides a range of environments for RL experiments.
- **Modular codebase**: Structured to easily add new RL algorithms or environments.

## Prerequisites

- **Python 3.7+**
- **OpenAI Gym** for RL environments
- Additional libraries as specified in the `requirements.txt` (if available).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wjsekrud/RL_Testbed.git
   cd RL_Testbed

## Usage
 - Configure the environment and algorithm settings in the code or via configuration files.
 - Run the testbed with your selected algorithm and environment:
```bash
  python main_gui.py
```
 - Observe training metrics and results for algorithm comparison and evaluation.

 ## Directory Structure
 - `modules/agents`: Contains various RL algorithms and utilities.
 - `modules/main_gui.py`: Main script to initialize and run the testbed GUI.
 - `modules/agents/checkpoints`: when learning process completes, the result modle will be saved in here.
