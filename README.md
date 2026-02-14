# Neuroevolution with Gymnasium
## Lunar Lander experiments using SALGA

This repository contains a practical exploration of **neuroevolutionary methods** applied to the
`LunarLander-v2` environment from the **Gymnasium** library.

The project is structured as a progressive experiment:
from human control, to heuristic agents, and finally to a **neuroevolutionary agent**
whose neural controller parameters are optimized using **SALGA (Simple Adaptive Learning Genetic Algorithm)**.

---

## Project overview

The goal of this project is to understand different ways of controlling an agent in a complex environment:

1. **Human control**  
   Manual interaction with the Lunar Lander environment to gain intuition about the task.

2. **Rule-based agent**  
   A handcrafted policy based on thresholds over the observation variables.

3. **Neuroevolutionary agent**  
   A neural controller (single-layer perceptron) whose weights and biases are evolved using SALGA.

Instead of using gradient-based reinforcement learning, this project focuses on **evolving neural network parameters directly**.

---

## Neuroevolution with SALGA

- The agent is represented by a **single-layer perceptron**
- Inputs: 8 (environment observations)
- Outputs: 4 (discrete action space)
- Weights and biases are encoded as a **real-valued chromosome**
- SALGA evolves a population of chromosomes using selection, mutation, and recombination

The fitness of each chromosome is computed by running a complete episode of the Lunar Lander environment
and measuring the accumulated reward.

The best chromosome found by SALGA is later loaded into the model and visualized in the notebook.

---

##  Notebook

### `Neuroevolution-LunarLander.ipynb`

The notebook contains:
- Environment exploration and manual control
- Implementation of a heuristic policy
- Definition of the perceptron model
- Integration of a chromosome obtained with SALGA
- Evaluation and visualization of agent behavior

The notebook is intended for **educational purposes**.

---

## Running the neuroevolution

The file `lunarlander_salga.py` is intended to be executed using **SALGA (Simple Adaptive Learning Genetic Algorithm)**.

This script defines:
- The chromosome representation (perceptron weights and biases)
- The neural controller architecture
- The fitness function, which evaluates each chromosome by running a full episode of the
  `LunarLander-v2` environment and measuring the accumulated reward

When executed with SALGA, the algorithm automatically evolves a population of chromosomes
in order to maximize the fitness score.
After the evolutionary process finishes, the best chromosome found is printed and can be
manually copied into the notebook to evaluate and visualize the resulting agent behavior.

During evolution, environment rendering is disabled to improve performance.
Visualization is performed separately in the notebook using the obtained chromosome.

---

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
