import os,sys
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import trange
from network import Network


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
    training_data = datasets.MNIST(root="~/datasets",train=True,download=True,transform=ToTensor())
    test_data = datasets.MNIST(root="~/datasets",train=False,download=True,transform=ToTensor())
    train_img,train_target = process(training_data)
    test_img,test_target = process(test_data)
    return train_img,train_target,test_img,test_target

def mnist(epoch):
    network = Network()
    network.create_layer(784)
    network.create_layer(32)
    network.create_layer(10)
    network.connect_layers()
    train_img,train_target,test_img,test_target = mnist_data()
    train_num = len(train_img)
    test_num = len(test_img)
    for epoch in range(epoch):
        print(epoch)
        for i in trange(train_num):
            network.reset()
            data = network.inference(train_img[i])
            network.layer_list[-1].input_loss(train_target[i])
            network.back_propagation()
        
        correct = 0
        for i in trange(test_num):
            network.reset()
            data = network.inference(test_img[i])
            if np.array(data).argmax() == test_target[i].argmax(): correct += 1
        print(correct/test_num)
        sys.stdout.flush()

if __name__ ==  '__main__':
    mnist(10)