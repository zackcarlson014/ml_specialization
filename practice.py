import numpy as np
import matplotlib.pyplot as plt
import copy
import math

np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5
print("tmp_X.shape:", tmp_X.shape)
print("tmp_X:", tmp_X)
print("tmp_w:", tmp_w)
print("tmp_b:", tmp_b)

m, n = tmp_X.shape   
p = np.zeros(m)

print("m:", m)
print("n:", n)
print("p:", p)