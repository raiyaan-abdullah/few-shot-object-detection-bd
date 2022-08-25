#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 00:42:40 2022

@author: raiyaan
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

voc = np.loadtxt('/home/raiyaan/GitHub/few-shot-object-detection-bd/vocab/vocabulary_glo.txt', dtype='float32', delimiter=',')
vec = np.loadtxt('/home/raiyaan/GitHub/few-shot-object-detection-bd/vocab/word_glo.txt', dtype='float32', delimiter=',')
#vec = vec[:, :65]
vec = torch.from_numpy(vec)
voc = torch.from_numpy(voc)