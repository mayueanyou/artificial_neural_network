import os,sys,torch
import torch.nn as nn
from layer import*
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import trange

class Network:
    def __init__(self) -> None:
        self.layer_list = []
    
    def inference(self,x):
        for layer in self.layer_list:
            x = layer.inference(x)
        return x
    
    def backpropagation(self,gradient):
        for layer in reversed(self.layer_list):
            gradient = layer.backpropagation(gradient)
    
    def regularization(self):
        ...
    
    def calculate_loss(self,prediction,label):
        def L1(prediction,label):
            loss = torch.abs(label-prediction)
            gradient = torch.sign(label-prediction)
            return gradient
        
        def L2(prediction,label):
            loss = (label-prediction)**2
            loss_mean = torch.mean(loss)
            gradient = label-prediction
            return gradient

        def corss_entropy(prediction,label):
            prediction_exp = torch.exp(prediction)
            prediction_prob = prediction_exp/torch.sum(prediction_exp)
            label_argmax = torch.argmax(label)
            loss = -torch.log(prediction_prob[label_argmax])
            gradient = label - prediction_prob
            return gradient
        
        #return L1(prediction,label)
        #return L2(prediction,label)
        return corss_entropy(prediction,label)
    
    
def mnist_data():
    def process(data):
        image = data.data / 255
        targets = data.targets
        #image = data.data[:1000] / 255
        #targets = data.targets[:1000]
        new_targets = torch.zeros(len(targets),10)
        for i in range(len(new_targets)):
            new_targets[i][targets[i]] = 1
        return image,new_targets
    
    training_data = datasets.MNIST(root="/home/yma183/datasets",train=True,download=True,transform=ToTensor())
    test_data = datasets.MNIST(root="/home/yma183/datasets",train=False,download=True,transform=ToTensor())
    train_img,train_target = process(training_data)
    test_img,test_target = process(test_data)
    return train_img,train_target,test_img,test_target

def mnist():
    network = Network()
    
    #network.layer_list.append(GaussianLayer(784,3200))
    #network.layer_list.append(GaussianLayer(3200,10))
    #network.layer_list.append(LogisticLayer(784,10))
    #network.layer_list.append(LogisticLayer(10,10))
    #network.layer_list.append(LogisticLayer(10,10))
    network.layer_list.append(LinearLayer(784,32))
    network.layer_list.append(ReluLayer(32))
    network.layer_list.append(LinearLayer(32,10))
    train_img,train_target,test_img,test_target = mnist_data()
    train_num = len(train_img)
    test_num = len(test_img)
    for epoch in range(10):
        print(epoch)
        for i in trange(train_num):
            data = network.inference(torch.flatten(train_img[i]))
            loss = network.calculate_loss(data,train_target[i])
            #print(loss)
            network.backpropagation(loss)
        
        correct = 0
        for i in trange(test_num):
            data = network.inference(torch.flatten(test_img[i]))
            if data.argmax() == test_target[i].argmax(): correct += 1
        print(correct/test_num)
        
        #print(network.layer_list[1].weight)
        #input()
        sys.stdout.flush()

if __name__ == '__main__':
    mnist()