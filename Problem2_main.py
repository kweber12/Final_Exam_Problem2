import numpy as np
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import math
import Problem2_methods as p

G = p.GenGraph(10,2)
print(G)
dummy = np.zeros(100)
#Finging the average fraction of verticies in the giant component for 100 graphs
for i in range(100):
    G = p.GenGraph(10,2.2)
    O = p.DFS(G,0)
    a = np.amax(O, axis = 0)
    dummy[i] = np.count_nonzero(a > 0) + 1
av = np.sum(dummy)/len(dummy)