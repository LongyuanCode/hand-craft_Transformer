import torch
from torch import nn

batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
params = layer_norm.state_dict()
for name, param in params.items():
    print("Param name: {}".format(name))
    print("Param version: {}".format(param._version))
