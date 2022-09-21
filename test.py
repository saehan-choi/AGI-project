import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt
import os

# MSE loss test
'''

a = torch.tensor([1.0, 1.0])
b = torch.tensor([2.0, 1.5])
criterion = nn.MSELoss()

print(criterion(a,b))

print((((2.0-1.0)**2) + ((1.5-1.0)**2))/2.0)

'''




'''
x = torch.randn(10)
print(x)
mask = x.ge(0.1)
# print(mask)

print(torch.masked_fill(x,torch.tensor([False,  True, False, False, False, False,  True,  True,  True, False]),0))
print(torch.rand(100)>0.9)
'''



arr = []
f = open('report.txt')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

for a in f.readlines():
    a = int(a.replace('\n', ''))
    arr.append(a)
    

plt.plot(arr)
plt.show()