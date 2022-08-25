import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VocabModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024,300)
        # self.linear = nn.Linear(256, 2)
        self.vocab = VocabLayer()
        self.fc2 = nn.Linear(80,1024)

    def forward(self, x):

        x = self.fc(x)
        #print("After first linear: ",x.shape)
        x = self.vocab(x)
        x= self.fc2(x)
        return x

class VocabLayer(nn.Module):

    def __init__(self):
        super().__init__()
        
        
        voc = np.loadtxt('vocab/vocabulary_glo.txt', dtype='float32', delimiter=',')
        vec = np.loadtxt('vocab/word_glo.txt', dtype='float32', delimiter=',')
        #vec = vec[:, :65]
        self.vec = torch.from_numpy(vec).to(device)
        self.voc = torch.from_numpy(voc).to(device)
    
        weights = torch.Tensor(self.voc.size(1), self.vec.size(0))
        self.weights = nn.Parameter(weights)   
        torch.nn.init.uniform_(self.weights) 

    def forward(self, inputs):
        #print("Vec shape",self.vec.shape)
        #print("Voc shape",self.voc.shape)
        #print("weight shape",self.weights.shape)
        projection = torch.mm(torch.mm(self.voc, self.weights), self.vec)
        projection = torch.tanh(projection)
        return torch.mm(inputs, projection)
