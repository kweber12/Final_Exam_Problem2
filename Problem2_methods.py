"""
This file stores all the functions we need for the problem
"""
import pandas
import numpy as np
import scipy.sparse.linalg as linalg
import time
import math
from scipy.sparse import csr_matrix

def GenGraph(n,a):
    K = np.round(n**(1/(a-1))) #Maximal Degree
    k = np.arange(1,K)
    p = k**(-a)
    psum = np.sum(p)
    p = p/psum #normalizing our probability distribution 
    aux = np.round(np.cumsum(n*p)).astype(int)# assigning degrees to vertices according to our probability distribution 
    w = np.zeros([n,1])
    prev = 1
    K = K.astype(int)
    for j in range(K-1):
        if aux[j] >= prev:
            w[prev:aux[j]]= j 
            prev = aux[j]+1
    # w = vector of degrees
    wcumsum = np.cumsum(w)
    q = len(wcumsum)
    m = wcumsum(q-1) # total number of edges *2
    if m % 2 == 1:
        q1 = len(w)
        w[q1] = w[q1] + 1
        wcumsum = np.cumsum[w]
    m = wcumsum(q-1) 
    stubs = np.zeros(m)
    prev = 1
    for j in range (n):
        stubs[prev:wcumsum[j]] = j
        prev = wcumsum[j] + 1
    # randomly match stubs
    s0 = np.rand.perm[m]
    e = len(s0)
    s1 = s0[1:m/2]
    s2 = s0[m/2+1:e]
    edges = np.zeros[m/2,2]
    edges[:,1] = stubs[s1]
    edges[:,2] = stubs[s2]
    # remove self loops
    ind = np.find(edges[:,1]==edges[:,2])
    edges[ind,:] = []
    # remove repeated edges
    edges = np.sort(edges,2)
    edges = np.unique(edges,axis = 0)
    # form the adjacency matrix
    G = csr_matrix(edges[:,1],edges[:,2], np.ones([len(edges),1]),n,n)
    G = G.copy().tocsr()
    G.data.fill(1)
    return G 

def DFS(G,t):
    (n,d) = G.shape
    visited = np.array([0])
    out = np.empty((0,n),int)
    for s in range(n):
        st = np.array([s])
        comp = np.array([s])
        while st.size != 0: 
            a = np.where(G[st[-1],:] == 1) 
            a = a[0] 
            q = a.size 
            for i in range(q):
                if i >= a.size:
                    break
                v = np.where(visited == a[i])
                v = v[0]
                if v.size != 0:
                    a = np.delete(a,i)
            q = a.size 
            if q == 0: 
                st = np.delete(st,-1)
                
            else:
                G[st[-1],a[0]] = 0
                st = np.append(st,a[0])
                t = t+1
                visited = np.append(visited,a[0])
                h = np.where(comp == a[0])
                h = h[0]
                if h.size != 0:
                    a = np.delete(a,h)
                comp = np.append(comp,a[0]) 
            if st.size == 0:
                if comp.size == 1:
                    break
                f = comp.size
                w = n - comp.size 
                e = np.zeros(w) 
                comp = np.append(comp,e).astype(int)
                comp = np.array([comp])#comp does not have correct dimestions if I do not do this
                out = np.append(out, comp, axis = 0)
                comp = np.delete(comp,slice(f-1,n))
                
                
    return out

def BFS(G,s):
    t = 1
    q = np.array([s])
    (n,d) = G.shape
    visited = np.array([s])
    prev = []
    frac = np.array([1])
    while q.size != 0:
        t = t+1
        c = np.where(G[q[0],:] == 1)
        k = visited.size
        for i in range(k):
            w = np.where(c == visited[i])
            c = np.delete(c,w)
        q = np.append(q,c)
        
        prev = np.append(prev,q[0])  
        q = np.delete(q,0)
        visited = np.append(visited, c).astype(int)
        vv = len(prev)/n
        frac = np.append(vv, axis = 0)
     
    a = prev.size
    prev = prev.astype(int)
   
    return (prev,frac,t)
    
     