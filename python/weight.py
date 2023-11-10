import os,sys,random

class Weight:
    def __init__(self) -> None:
        self.weight = random.random()
        self.input = 0
        self.output = 0
        self.loss_in = 0
        self.loss_out = 0
        self.rate = 0.1
    
    def inference(self,input):
        self.input = input
        self.output = self.input * self.weight
    
    def calculate_loss(self,loss_in):
        self.loss_in = loss_in
        self.weight = self.rate * self.input * self.loss_in
        self.loss_out = self.loss_in * self.weight