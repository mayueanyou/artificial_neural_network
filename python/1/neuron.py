import os,sys,random
import numpy as np

class Neuron:
    def __init__(self) -> None:
        self.post_neuron_list = []
        self.post_weight_list = []
        self.neuron_value = 0
        self.neuron_loss = 0
        self.learning_rate = 0.001
    
    def reset(self):
        self.neuron_value = 0
        self.neuron_loss = 0
    
    def initialise_weights(self):
        self.post_weight_list = np.random.random((len(self.post_neuron_list)))
        
    def inference(self):
        for i in range(len(self.post_neuron_list)):
            self.post_neuron_list[i].neuron_value += self.post_weight_list[i] * self.neuron_value
    
    def weight_update(self):
        for i in range(len(self.post_neuron_list)):
            self.post_weight_list[i] += self.learning_rate * self.neuron_value * self.post_neuron_list[i].neuron_loss
    
    def calculate_loss(self):
        for i in range(len(self.post_neuron_list)):
            self.neuron_loss += self.post_weight_list[i] * self.neuron_loss

if __name__ == '__main__':
    x = Neuron()
    y = Neuron()
    x.post_neuron_list = [y]
    x.neuron_value = 1
    x.initialise_weights()
    
    for i in range(100):
        x.inference()
        y.neuron_loss = 40 - y.neuron_value
        x.weight_update()
        print(y.neuron_value)
    