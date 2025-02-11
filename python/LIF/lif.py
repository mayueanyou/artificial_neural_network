import os,sys,torch,copy
import numpy as np

class Synapse_Filter:
    def __init__(self,dim) -> None:
        self.a = 0.9
        self.b = 0.9
        self.dim = dim
        self.state = np.zeros((dim))
        #self.state = torch.zeros(dim)
    
    def step(self,synapse_in):
        self.state = synapse_in * self.a + self.state * self.b
        return copy.deepcopy(self.state)
    
    def reset(self):
        self.state = np.zeros((self.dim))
        #self.state = torch.zeros(self.dim)


class Membrane_Potential:
    def __init__(self,weight) -> None:
        self.weight = weight
        self.dim_in = len(weight[0])
        self.dim_out = len(weight)
        self.state = np.zeros((self.dim_out))
        self.state_out = np.zeros((self.dim_out))
        #self.state = torch.zeros(self.dim_out)
        #self.state_out = torch.zeros(self.dim_out)
        self.threshhold = 1
        self.l = 0.5
    
    def step(self,data_in):
        self.state = np.sum(data_in * self.weight, axis=1) + self.state * self.l - self.state_out
        #self.state = torch.sum(data_in * self.weight, axis=1) + self.state * self.l - self.state_out
        self.state_out = np.where(self.state >= self.threshhold, 1, 0)
        #self.state_out = torch.where(self.state >= 1, 1, 0)
        return copy.deepcopy(self.state_out)
    
    def reset(self):
        self.state = np.zeros((self.dim_out))
        self.state_out = np.zeros((self.dim_out))
        #self.state = torch.zeros(self.dim_out)
        #self.state_out = torch.zeros(self.dim_out)

def main():
    data_1 = torch.load('./weights/snn1_6_50.pkl',map_location=torch.device('cpu'))['weight.weight'].detach().numpy()
    data_2 = torch.load('./weights/snn2_50_50.pkl',map_location=torch.device('cpu'))['weight.weight'].detach().numpy()
    data_3 = torch.load('./weights/snn3_50_4.pkl',map_location=torch.device('cpu'))['weight.weight'].detach().numpy()
    data_in = torch.load('./weights/input_data').detach().numpy()
    data_out = torch.load('./weights/spike_output').detach().numpy()
    sf_1 = Synapse_Filter(6)
    sf_2 = Synapse_Filter(50)
    sf_3 = Synapse_Filter(50)
    mp_1 = Membrane_Potential(data_1)
    mp_2 = Membrane_Potential(data_2)
    mp_3 = Membrane_Potential(data_3)
    result = []
    for j in range(4):
        sf_1.reset()
        sf_2.reset()
        sf_3.reset()
        mp_1.reset()
        mp_2.reset()
        mp_3.reset()
        idx = j
        s = 0
        for i in range(len(data_in[idx])):
            x = sf_1.step(data_in[idx][i])
            x = mp_1.step(x)
            x = sf_2.step(x)
            x = mp_2.step(x)
            x = sf_3.step(x)
            x = mp_3.step(x)
            print(i,np.array_equal(data_out[idx][i],x),'expect: ',data_out[idx][i],'real: ',x)
            if np.array_equal(data_out[idx][i],x): s+=1
        result.append(s)
        print(s,'/100')
    print(result)


if __name__ == '__main__':
    main()
