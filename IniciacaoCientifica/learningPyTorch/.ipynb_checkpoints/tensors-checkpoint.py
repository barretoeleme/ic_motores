# tensors work similarly as vectors and matrixes

import torch

x = torch.empty(1)
print(x)

x = torch.rand(2, 3)
print(x)

x = torch.ones(3, 2, 1)
print(x)
# multidimensional tensor

x = torch.ones(2, 2, dtype = torch.double)
print(x.dtype)
print(x.size())

# you can also make tensors out of arrays
x = torch.tensor([2.5, 0.1])
print(x)
