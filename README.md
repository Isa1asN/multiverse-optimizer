# Multi-Verse Optimizer for Hyperparameter Tuning

![Multiverse Artistic Representation](imgs/multiverse_bubble.png)

## Overview
This project implements the **Multi-Verse Optimizer (MVO)** to optimize hyperparameters of **Stochastic Gradient Descent (SGD)** for training a neural network. The goal is to improve model performance on the **Energy Efficiency dataset** by tuning key parameters such as **learning rate, momentum, and weight decay**.

## How It Works
- **MVO Mechanism**: Utilizes exploration (white/black hole selection) and exploitation (wormhole teleportation) to refine hyperparameters.
- **Neural Network Model**: A fully connected **PyTorch neural network** trained for regression tasks.
- **Optimization Process**:
  - Initialize a population of hyperparameter sets (universes).
  - Evaluate performance using validation loss.
  - Evolve universes using MVO strategies.
  - Select the best hyperparameter set.

## Results
The optimized hyperparameters improve model performance, reducing **MSE** and increasing **R² score** compared to standard SGD settings.

## References
For theoretical background, refer to:
- Mirjalili, S., Mirjalili, S. M., & Hatamlou, A. (2016). "Multi-Verse Optimizer: A nature-inspired algorithm for global optimization." *Neural Computing and Applications, 27*(2), 495-513. Available at: [ResearchGate](https://www.researchgate.net/publication/273916757_Multi-Verse_Optimizer_a_nature-inspired_algorithm_for_global_optimization)
- **UCI Machine Learning Repository - Energy Efficiency Dataset.** Available at: [https://archive.ics.uci.edu/dataset/242/energy+efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency).

---