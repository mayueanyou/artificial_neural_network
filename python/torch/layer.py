import os,sys,torch
from abc import ABC,abstractmethod

class Layer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @abstractmethod
    def inference(self):pass
    
    @abstractmethod
    def backpropagation(self):pass

class SoftmaxLayer(Layer):
    def __init__(self,dim) -> None:
        self.dim = dim
        self.data_in = torch.zeros(dim)
    
    def inference(self,data_in):
        self.data_in = data_in
        data_out = torch.exp(data_in)
        data_out = data_out/torch.sum(data_out)
        return data_out
    
    def backpropagation(self,gradient_in):
        gradient_out = gradient_in * self.data_in * (1-self.data_in)
        return gradient_out

class ReluLayer(Layer):
    def __init__(self,dim) -> None:
        self.dim = dim
        self.data_zeros = torch.zeros(dim)
    
    def inference(self,data_in):
        self.data_in = data_in
        data_out = torch.where(data_in<0, self.data_zeros, data_in)
        return data_out
    
    def backpropagation(self,gradient_in):
        gradient_out = torch.where(self.data_in<0, self.data_zeros, gradient_in)
        return gradient_out

class LinearLayer(Layer):
    def __init__(self,dim_in,dim_out) -> None:
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.intialize_rate = 0.001
        self.weight = torch.rand(dim_in,dim_out) * self.intialize_rate - self.intialize_rate/2
        self.bias = torch.rand(dim_out)
        self.learning_rate = 0.001
        self.weight_ones = torch.ones(dim_in,dim_out)
        self.data_in = torch.zeros(dim_in)
    
    def inference(self,data_in):
        self.data_in = data_in
        data_out = data_in @ self.weight + self.bias
        return data_out
    
    def backpropagation(self,gradient_in):
        gradient_out = gradient_in @ self.weight.T
        self.weight += self.learning_rate * self.weight_ones * gradient_in * torch.unsqueeze(self.data_in, 1)
        self.bias += self.learning_rate * gradient_in
        return gradient_out

class LogisticLayer:
    def __init__(self,dim_in,dim_out) -> None:
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = torch.rand(dim_in,dim_out) * 0.001 + 3.6
        #self.weight = torch.rand(dim_in,dim_out) * 0.001
        self.bias = torch.rand(dim_out)
        self.learning_rate = 0.001
        self.weight_ones = torch.ones(dim_in,dim_out)
        self.data_in = torch.zeros(dim_in)
        self.data_in_ones = torch.ones(dim_in)
    
    def inference(self,data):
        self.data_in = data * (self.data_in_ones - data)
        self.data_in = torch.nan_to_num(self.data_in)
        data_out = (self.data_in @ self.weight)/self.dim_out + self.bias
        return data_out
    
    def backpropagation(self,loss_in):
        loss_in = torch.nan_to_num(loss_in)
        loss_out = loss_in @ self.weight.T
        self.weight += self.learning_rate * self.weight_ones * loss_in * self.data_in.reshape(-1,1)
        self.bias += self.learning_rate * loss_in
        self.weight = torch.clip(self.weight,min=3.6,max=3.9) + torch.rand(self.dim_in,self.dim_out) * 0.001
        return loss_out

class GaussianLayer:
    def __init__(self,dim_in,dim_out) -> None:
        self.mean = torch.rand(dim_in,dim_out) * 0.001
        self.bias = torch.rand(dim_out)
        self.std = torch.rand(dim_in,dim_out) * 0.02
        self.learning_rate = 0.001
        self.mean_ones = torch.ones(dim_in,dim_out)
        self.data_in = torch.zeros(dim_in)
        self.sample = None
    
    def inference(self,data):
        self.data_in = data
        self.sample = torch.normal(self.mean,self.std)
        #print(self.std)
        data_out = data @ self.sample + self.bias
        return data_out
    
    def backpropagation(self,loss_in):
        loss_out = loss_in @ self.sample.T / self.mean.shape[0]
        #self.weight += self.learning_rate * self.weight_ones * loss_in * torch.unsqueeze(self.data_in, 1)
        self.mean += self.learning_rate * self.mean_ones * loss_in * self.data_in.reshape(-1,1)
        self.bias += self.learning_rate * loss_in
        #print(loss_in)
        #self.std = torch.abs(torch.abs(loss_in.expand(self.std.shape[0],-1)) - self.std) * self.learning_rate
        self.std *= torch.abs(loss_in.expand(self.std.shape[0],-1))
        #print(torch.abs(loss_in))
        return loss_out



if __name__ == '__main__':
    layer = Layer(784,3)
    label = torch.ones(10)
    
    for i in range(100):
        data = layer.inference(torch.ones(50))
        layer.backpropagation(label-data)
        print(data)
    