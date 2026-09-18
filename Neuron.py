import torch
import torch.nn as nn

#Input
input = torch.tensor([1.0,2.0,3.0])

neuron = nn.Linear(in_features = 3 , out_features = 1)

output = neuron(input)