#!/usr/bin/python
# -*- coding: utf-8 -*-

# Neuroevolution for LunarLander-v2 using SALGA
# Chromosomes encode the weights and biases of a perceptron controller

import numpy as np
import gymnasium as gym

class Perceptron:
    def __init__(self, ninput, noutput):
        self.ninput = ninput
        self.noutput = noutput
        self.w = np.random.rand(ninput,noutput)-0.5
        self.b = np.random.rand(noutput)-0.5
        
    def forward (self, x): # propaga un vector x y devuelve la salida
        u = np.dot(x, self.w) + self.b              
        return np.piecewise(u, [u<0, u>=0], [0,1])
                   
        
    def update (self, x, d, alpha): # realiza una iteración de entrenamiento
        s = self.forward(x) # propaga
        # Calcula la actualización de los pesos y el sesgo
        error = d - s
        self.w += alpha * np.outer(x,error)
        self.b += error*alpha
        
               
    def RMS (self, X, D): # calcula el error RMS
        S = self.forward(X)
        return np.mean(np.sqrt(np.mean(np.square(S-D),axis=1)))
        
    def accuracy (self, X, D): # calcula el ratio de aciertos
        S = self.forward(X)
        errors = np.mean(np.abs(D-S))
        return 1.0 - errors
    
    def info (self, X, D): # traza de cómno va el entrenamiento
        print('     RMS: %6.5f' % self.RMS(X,D))
        print('Accuracy: %6.5f' % self.accuracy(X,D))
        
    def train (self, X, D, alpha, epochs, trace=0): # entrena usando update
        for e in range(1,epochs+1):
            for i in range(len(X)):
                self.update(X[i],D[i], alpha)
            if trace!=0 and e%trace == 0:
                print('\n   Epoch: %d' % e)
                self.info(X,D)

    def from_chromosome(self, chromosome):
    # Extraer los pesos y bias de la lista del cromosoma
        w_size = self.ninput * self.noutput
        w = np.array(chromosome[:w_size]).reshape(self.ninput, self.noutput)
        b = np.array(chromosome[w_size:w_size+self.noutput])

        # Actualizar los pesos y bias de la red
        self.w = w
        self.b = b


env = gym.make("LunarLander-v3")
def fitness(chromosome):
    # Crear modelo a partir del cromosoma
    model=Perceptron(8,4)
    model.from_chromosome(chromosome)    
    
    lista=0
    for i in range(5):
        observation, info = env.reset()
        ite = 0
        racum = 0
        while True:
            action = np.argmax(model.forward(observation))
            observation, reward, terminated, truncated, info = env.step(action)
            
            
            racum += reward
            
        

            if terminated or truncated:
                break
        racum=(racum+500)/800
        lista+=racum       
    
    return lista/5

parameters={'alphabet':[-3,3], 'type':'floating', 'elitism':True, 'norm':True, 'chromsize':36, 'trace':1,'target':350}


