import numpy as np
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import math
import Problem2_methods as p


dummy = 0
for i in range(100):
    G = p.GenGraph(10,2.2)
    s = np.random.rand(10)
    (prev,frac,t) = p.BFS(G,s)
    dummy = dummy + frac
frac = dummy/100 

fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax = fig.add_subplot(111)
plt.plot(t,frac) 


