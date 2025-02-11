import os,sys
import numpy as np
from neuron import Neuron

class Layer:
    def __init__(self,num=0) -> None:
        self.neuron_list = []
        self.generate_neurons(num)
    
    def generate_neurons(self,num):
        for i in range(num):
            self.neuron_list.append(Neuron())
    
    def reset(self):
        for neuron in self.neuron_list:
            neuron.reset()
    
    def input_data(self,data):
        for i in range(len(self.neuron_list)):
            self.neuron_list[i].neuron_value = data[i]
    
    def read_out_data(self):
        data = []
        for neuron in self.neuron_list:
            data.append(np.around(neuron.neuron_value,decimals=2))
        return data
    
    def input_loss(self,expect):
        for i in range(len(self.neuron_list)):
            self.neuron_list[i].neuron_loss = expect[i] - self.neuron_list[i].neuron_value 
    
    def inference(self):
        for neuron in self.neuron_list:
            neuron.inference()
    
    def weight_update(self):
        for neuron in self.neuron_list:
            neuron.weight_update()
    
    def calculate_loss(self):
        for neuron in self.neuron_list:
            neuron.calculate_loss()
    
    def connect_next_layer(self,next_layer):
        for neuron in self.neuron_list:
            neuron.post_neuron_list = next_layer.neuron_list
            neuron.initialise_weights()


if __name__ == '__main__':
    layer1 = Layer(1)
    layer2 = Layer(1)