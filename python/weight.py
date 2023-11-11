import os,sys,random

class Weight:
    def __init__(self) -> None:
        self.weight = random.random()
        self.input_neuron = None
        self.output__neuron = None
    
    def inference(self):
        self.input = input
        self.output = self.input * self.weight
    
    def calculate_loss(self,loss_in):
        self.loss_in = loss_in
        self.weight = self.rate * self.input * self.loss_in
        self.loss_out = self.loss_in * self.weight