#!/usr/bin/python
# -*- coding: utf-8 -*-

# Neuroevolution for LunarLander-v2 using SALGA
# Chromosomes encode the weights and biases of a perceptron controller

import gymnasium as gym
import numpy as np


# =========================================================
# Neural model (phenotype)
# =========================================================

class Perceptron:
    def __init__(self, ninput, noutput):
        self.ninput = ninput
        self.noutput = noutput
        self.w = np.zeros((ninput, noutput))
        self.b = np.zeros(noutput)

    def forward(self, x):
        u = np.dot(x, self.w) + self.b
        return np.piecewise(u, [u < 0, u >= 0], [0, 1])

    def from_chromosome(self, chromosome):
        w_size = self.ninput * self.noutput
        self.w = np.array(chromosome[:w_size]).reshape(self.ninput, self.noutput)
        self.b = np.array(chromosome[w_size:w_size + self.noutput])


# =========================================================
# Policy
# =========================================================

def policy(observation, model):
    s = model.forward(observation)
    return np.argmax(s)


# =========================================================
# Fitness function (used by SALGA)
# =========================================================

def fitness(chromosome):
    """
    Given a chromosome, evaluates its fitness by running
    one episode of LunarLander-v2.
    """

    env = gym.make("LunarLander-v2")
    observation, _ = env.reset()

    model = Perceptron(8, 4)
    model.from_chromosome(chromosome)

    total_reward = 0.0

    while True:
        action = policy(observation, model)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return total_reward


# =========================================================
# Chromosome definition
# =========================================================

N_INPUTS = 8
N_OUTPUTS = 4
CHROMOSOME_LENGTH = N_INPUTS * N_OUTPUTS + N_OUTPUTS   # 36 genes


# =========================================================
# SALGA parameters
# =========================================================

parameters = {
    'type': 'real',                     # real-valued chromosome
    'length': CHROMOSOME_LENGTH,        # number of genes
    'target': 200,                      # target fitness
    'elitism': True,
    'pmut': 1.0 / CHROMOSOME_LENGTH
}

