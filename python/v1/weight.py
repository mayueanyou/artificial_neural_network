import os,sys,random
import numpy as np

class Weight:
    def __init__(self) -> None:
        self.weight = 0.001 * np.random.rand()
        self.learning_rate = 0.1
        #self.learning_rate = 0.1 * np.random.rand()
    
        self.data_in = 0
        self.data_out = 0
        self.loss_in = 0
        self.loss_out = 0
    
    def print(self):
        print(f'data_in: {self.data_in}')
        print(f'data_out: {self.data_out}')
        print(f'loss_in: {self.loss_in}')
        print(f'loss_out: {self.loss_out}')
        print(f'weight: {self.weight}')
        
    
    def inference(self):
        self.data_out = self.data_in * self.weight
    
    def weight_update(self):
        #self.learning_rate = 0.1 * np.random.rand()
        self.loss_out = self.loss_in * self.weight
        self.weight += self.learning_rate * self.loss_in * self.data_in

class GaussianWeight:
    def __init__(self) -> None:
        self.sigma = 0.2
        self.samples = 10
        self.weight = 0.001 * np.random.rand()
        self.learning_rate = 0.1
        #self.learning_rate = 0.1 * np.random.rand()
    
        self.data_in = 0
        self.data_out = 0
        self.loss_in = 0
        self.loss_out = 0
    
    def print(self):
        print(f'data_in: {self.data_in}')
        print(f'data_out: {self.data_out}')
        print(f'loss_in: {self.loss_in}')
        print(f'loss_out: {self.loss_out}')
        print(f'weight: {self.weight}')
        
    
    def inference(self):
        self.data_out = self.data_in * np.mean(np.random.normal(self.weight, self.sigma, self.samples))
        #self.data_out = self.data_in * np.random.normal(self.weight, self.sigma, 1)
    
    def weight_update(self):
        self.loss_out = self.loss_in * self.weight
        self.weight += self.learning_rate * self.loss_in * self.data_in
        self.sigma = abs(self.loss_in)

if __name__ == '__main__':
    w = Weight()
    w.data_in = 1
    w.learning_rate = 0.01
    for i in range(10000):
        w.inference()
        w.loss_in = (1 - w.data_out)**2
        w.weight_update()
        print(w.data_out,w.loss_out)
        