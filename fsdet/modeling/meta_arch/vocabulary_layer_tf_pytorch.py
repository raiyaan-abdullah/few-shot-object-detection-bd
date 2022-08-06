import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VocabLayer(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        voc = np.loadtxt('vocab/vocabulary_glo.txt', dtype='float32', delimiter=',')
        vec = np.loadtxt('vocab/word_glo.txt', dtype='float32', delimiter=',')
        vec = vec[:, :65]
        vec = torch.from_numpy(vec)
        voc = torch.from_numpy(voc)
        self.vec = vec
        self.voc = voc
    
  
        self.kernel = torch.nn.Embedding(num_embeddings=self.voc.shape[1], 
                                     embedding_dim=self.vec.shape[1])  # Embedding layer
        torch.nn.init.uniform_(self.kernel.weight) 

    def forward(self, inputs):
        
        projection = torch.dot(torch.dot(self.voc, self.kernel), self.vec)
        projection = torch.tanh(projection)
        out = torch.dot(inputs, projection)
        return out