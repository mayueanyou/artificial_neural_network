import os,sys
import numpy as np
from layer import Layer
from neuron import Neuron
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import trange

class Network:
    def __init__(self) -> None:
        self.layer_list = []
    
    def connect_layers(self):
        for i in range(len(self.layer_list)-1):
            self.layer_list[i].connect_next_layer(self.layer_list[i+1])
    
    def inference(self,data):
        self.layer_list[0].input_data(data)
        for layer in self.layer_list:
            layer.inference()
        return self.layer_list[-1].read_out_data()
    
    def weight_update(self):
        for layer in self.layer_list:
            layer.weight_update()
    
    def calculate_loss(self):
        for i in reversed(range(len(self.layer_list))):
            self.layer_list[i].calculate_loss()
    
    def reset(self):
        for layer in self.layer_list:
            layer.reset()
        
def mnist_data():
    def process(data):
        data.data = data.data[:1000]
        data.targets = data.targets[:1000]
        image = np.array(data.data).reshape(len(data.data),-1)/255
        targets = np.array(data.targets)
        new_targets = np.zeros((len(targets),10))
        for i in range(len(new_targets)):
            new_targets[i][targets[i]] = 1
        return image,new_targets
    training_data = datasets.MNIST(root="/home/yma183/datasets",train=True,download=True,transform=ToTensor())
    test_data = datasets.MNIST(root="/home/yma183/datasets",train=False,download=True,transform=ToTensor())
    train_img,train_target = process(training_data)
    test_img,test_target = process(test_data)
    return train_img,train_target,test_img,test_target

def test():
    network = Network()
    network.layer_list.append(Layer(784))
    network.layer_list.append(Layer(32))
    network.layer_list.append(Layer(10))
    network.connect_layers()
    train_img,train_target,test_img,test_target = mnist_data()
    image = train_img[0]
    target = train_target[0]
    for i in range(100):
        data = network.inference(image)
        network.layer_list[-1].input_loss(target)
        network.calculate_loss()
        network.weight_update()
        network.reset()
        print(data)
    
        
def mnist():
    network = Network()
    network.layer_list.append(Layer(784))
    network.layer_list.append(Layer(32))
    network.layer_list.append(Layer(10))
    network.connect_layers()
    train_img,train_target,test_img,test_target = mnist_data()
    train_num = len(train_img)
    test_num = len(test_img)
    for epoch in range(100):
        print(epoch)
        for i in trange(train_num):
            data = network.inference(train_img[i])
            network.layer_list[-1].input_loss(train_target[i])
            network.calculate_loss()
            network.weight_update()
            network.reset()
            #print(data)
            #sys.stdout.flush()
        
        correct = 0
        for i in trange(test_num):
            data = network.inference(test_img[i])
            #network.layer_list[-1].input_loss(test_target[i])
            #network.calculate_loss()
            #network.weight_update()
            network.reset()
            if np.array(data).argmax() == test_target[i].argmax(): correct += 1
        print(correct/test_num)
        sys.stdout.flush()

if __name__ == '__main__':
    mnist()
    #test()