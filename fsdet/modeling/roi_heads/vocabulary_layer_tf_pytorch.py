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
        self.vocab = VocabLayer()


    def forward(self, x):
        ##print(x.shape)
        x = self.fc(x)
        ##print("Vocab Model After first linear: ",x.shape)
        x = self.vocab(x)
        ##print("Vocab Model After first vocab: ",x.shape)
        return x

class VocabLayer(nn.Module):

    def __init__(self):
        super().__init__()
        #vec = np.loadtxt('vocab/word_glo.txt', dtype='float32', delimiter=',')
        #vec = vec[:, :60]
        
        voc = np.loadtxt('vocab/vocabulary_ftx.txt', dtype='float32', delimiter=',')
        vec_old_coco = np.loadtxt('vocab/word_ftx.txt', dtype='float32', delimiter=',')
        
        
        vec = np.zeros((300,60),dtype="float32")
        vec[:,0] = vec_old_coco[:,5]
        vec[:,1] = vec_old_coco[:,7]
        vec[:,2:4] = vec_old_coco[:,8:10]
        vec[:,4] = vec_old_coco[:,67]
        vec[:,5] = vec_old_coco[:,10]
        vec[:,6] = vec_old_coco[:,16]
        vec[:,7] = vec_old_coco[:,69]
        vec[:,8:14] = vec_old_coco[:,17:23]
        vec[:,14:16] = vec_old_coco[:,70:71]
        vec[:,16] = vec_old_coco[:,23]
        vec[:,17] = vec_old_coco[:,72]
        vec[:,18:25] = vec_old_coco[:,24:31]
        vec[:,25:27] = vec_old_coco[:,32:34]
        vec[:,27] = vec_old_coco[:,73]
        vec[:,28:33] = vec_old_coco[:,34:39]
        vec[:,33] = vec_old_coco[:,74]
        vec[:,34:37] = vec_old_coco[:,39:42]
        vec[:,37] = vec_old_coco[:,75]
        vec[:,38:41] = vec_old_coco[:,42:45]
        vec[:,41] = vec_old_coco[:,48]
        vec[:,42] = vec_old_coco[:,76]
        vec[:,43] = vec_old_coco[:,51]
        vec[:,44] = vec_old_coco[:,77]
        vec[:,45:50] = vec_old_coco[:,52:57]
        vec[:,50] = vec_old_coco[:,78]
        vec[:,51:58] = vec_old_coco[:,57:64]
        vec[:,58] = vec_old_coco[:,79]
        vec[:,59] = vec_old_coco[:,64]
        


        self.vec = torch.from_numpy(vec).to(device)
        self.voc = torch.from_numpy(voc).to(device)
        

        weights = torch.Tensor(self.voc.size(1), self.vec.size(0))
        
        self.weights = nn.Parameter(weights)   
        torch.nn.init.normal_(self.weights) 
        

    def forward(self, inputs):
        ##print("Vec shape",self.vec.shape)
        ##print("Voc shape",self.voc.shape)
        ##print("weight shape",self.weights.shape)
        print(torch.mean(self.weights))
        projection = torch.matmul(torch.matmul(self.voc, self.weights), self.vec)
        projection = F.hardtanh(projection,min_val=-0.5, max_val=0.5)
        probabs = torch.matmul(inputs, projection)
        #probabs = F.tanh(probabs)
        return probabs
