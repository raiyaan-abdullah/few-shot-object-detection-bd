import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VocabLayer(nn.Module):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        voc = np.loadtxt('MSCOCO/vocabulary_glo.txt', dtype='float32', delimiter=',')
        vec = np.loadtxt('MSCOCO/word_glo.txt', dtype='float32', delimiter=',')
        vec = vec[:, :65]
        vec = torch.from_numpy(vec)
        voc = torch.from_numpy(voc)
        self.vec = vec
        self.voc = voc
        super(VocabLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(int(self.voc.shape[1]), int(self.vec.shape[0])),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        projection = torch.dot(torch.dot(self.voc, self.kernel), self.vec)
        projection = torch.tanh(projection)
        out = torch.dot(x, projection)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0],self.output_dim)