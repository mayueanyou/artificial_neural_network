import os,sys,random
import numpy as np
from neuron import Neuron
from weight import Weight

class Layer:
    def __init__(self,num) -> None:
        self.neuron_list = [Neuron() for _ in range(num)]
    
    def reset(self):
        for neuron in self.neuron_list:
            neuron.reset()
    
    def inference(self):
        for neuron in self.neuron_list:
            neuron.inference()
    
    def back_propagation(self):
        for neuron in self.neuron_list:
            neuron.back_propagation()
    
    def input_data(self,data):
        for i in range(len(self.neuron_list)):
            self.neuron_list[i].value = data[i]
    
    def read_out_data(self):
        data = []
        for neuron in self.neuron_list:
            data.append(np.around(neuron.value,decimals=2))
        return data
    
    def input_loss(self,expect):
        for i in range(len(self.neuron_list)):
            self.neuron_list[i].loss = expect[i] - self.neuron_list[i].value
            
if __name__ == '__main__':
    layer_1 = Layer(1)
    layer_2 = Layer(1)
    w = Weight()
    layer_1.neuron_list[0].post_weight_list = [w]
    layer_2.neuron_list[0].pre_weight_list = [w] 
    
    for i in range(10000):
        layer_2.reset()
        layer_1.reset()
        layer_1.input_data([1])
        layer_1.inference()
        layer_2.inference()
        data = layer_2.read_out_data()
        layer_2.neuron_list[0].loss = 1 - data[0]
        layer_2.back_propagation()
        layer_1.back_propagation()
        print(data)