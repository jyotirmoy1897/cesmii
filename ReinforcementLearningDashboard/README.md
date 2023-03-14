# Reinforcement Learning Dashboard

SMEs order often consists of high-mix of parts for manufacturing requiring diverse process routes in low volumes. Moreover, the processing machine due to wear and tear require regular maintenance and if then sudden machine breakdown may occur.

Therefore a reinforcement learning based algorithm is used due to its ability to generalize for dynamic scheduling. The dashboard has the following features:
- Reinforcement learning Training
- Hyperparameter Tuning with integrated [Weights & Biases](https://wandb.ai/site) support.
- Machine Break-down simulation
- Simantha Integration

## Installation Instruction
Install Following Packages:

- Pandas
- Numpy
- Pytorch
- Stable Baselines3
- Wandb
- Gym
- Streamlit
- ImageIO

Please note: Post installation of stable baselines3, the common/policies.py file needs to modified for categorical distribution.

### Wandb integration

Weights and Biases tool provides solution to run multiple hyperparameters and provides comprehensive graphs for comparison. For use of integrated wandb dashboard, a free wandb account has to created and details entered.


