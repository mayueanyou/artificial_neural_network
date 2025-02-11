import os,sys
import numpy as np
from layer import Layer
from weight import Weight,GaussianWeight
from tqdm import trange

class Network:
    def __init__(self) -> None:
        self.layer_list = []
    
    def create_layer(self,num):
        self.layer_list.append(Layer(num))
    
    def connect_neurons(self,pre_neuron,post_neuron):
        #weight = Weight()
        weight = GaussianWeight()
        pre_neuron.post_weight_list.append(weight)
        post_neuron.pre_weight_list.append(weight)
    
    def connect_layers(self):
        for i in range(len(self.layer_list)-1):
            for pre_neuron in self.layer_list[i].neuron_list:
                for post_neuron in self.layer_list[i+1].neuron_list:
                    self.connect_neurons(pre_neuron,post_neuron)
    
    def inference(self,data):
        self.layer_list[0].input_data(data)
        for layer in self.layer_list:
            layer.inference()
        return self.layer_list[-1].read_out_data()
    
    def back_propagation(self):
        for i in reversed(range(len(self.layer_list))):
            self.layer_list[i].back_propagation()
    
    def reset(self):
        for layer in self.layer_list:
            layer.reset()

if __name__ == '__main__':
    net = Network()
    net.create_layer(784)
    net.create_layer(10)
    net.create_layer(1)
    net.connect_layers()
    for i in range(100000):
        net.reset()
        data = net.inference([1]*784)
        net.layer_list[-1].neuron_list[0].loss = 1 - data[0]
        print(net.layer_list[1].neuron_list[0].post_weight_list[0].print())
        net.back_propagation()
        print(data)
        input()