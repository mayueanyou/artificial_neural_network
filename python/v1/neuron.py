import os,sys,random
from weight import Weight
import numpy as np

class Neuron:
    def __init__(self) -> None:
        self.value = 0
        self.loss = 0
        self.bias = 0
        self.learning_rate = 0.1
        self.pre_weight_list = []
        self.post_weight_list = []
    
    def print(self):
        print(self.value,self.loss)
    
    def reset(self):
        self.value = 0
        self.loss = 0
    
    def inference(self):
        for weight in self.pre_weight_list:
            weight.inference()
            self.value += weight.data_out
        self.value += self.bias
            
        for weight in self.post_weight_list:
            weight.data_in = self.value
    
    def back_propagation(self):
        for weight in self.post_weight_list:
            weight.weight_update()
            self.loss += weight.loss_out
        self.loss /= len(self.post_weight_list) + 1
        #self.bias += self.loss * self.bias * self.learning_rate
        
        for weight in self.pre_weight_list:
            weight.loss_in = self.loss

if __name__ == '__main__':
    x = Neuron()
    y = Neuron()
    z = Neuron()
    w1 = Weight()
    w2 = Weight()
    x.post_weight_list = [w1]
    y.pre_weight_list = [w1]
    y.post_weight_list = [w2]
    z.pre_weight_list = [w2]
    
    for i in range(100000):
        #print(f'----------{i}----------')
        x.value = 1
        x.inference()
        y.inference()
        z.inference()
        #print('-------------')
        y.loss = (1 - y.value)
        
        z.back_propagation()
        y.back_propagation()
        x.back_propagation()
        
        x.reset()
        y.reset()
        z.reset()
        print(z.value,z.loss)
        print(z.value,z.loss)
        print(z.value,z.loss)
        input()
        
        