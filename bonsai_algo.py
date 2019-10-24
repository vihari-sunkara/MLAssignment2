#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from scipy import sparse as sps
import utils
import predict
import DecisionTree as dt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import time as tm
import subprocess
# from scipy import io


# In[3]:


subprocess.call(['./sample_run.sh'])


# In[2]:


filename='score_mat.txt'
L=3400
sparse_matrix, _ = load_svmlight_file( "%s" % filename, multilabel = True, n_features = L, offset = 1 )
sparse_matrix


# In[4]:


# arr_ll=sparse_matrix.tolil()
arr_ll=sparse_matrix
arr_ll.indices


# In[11]:


k=1
tic = tm.perf_counter()
yPred = predict.getReco( arr_ll, k )
toc = tm.perf_counter()
print("time for prediction: ",toc-tic)
prec_k = utils.getPrecAtK( arr_ll, yPred, k )
print("prec@",k,": ",prec_k)
mprec_k = utils.getMPrecAtK( arr_ll, yPred, k )
print("mprec@",k,": ",mprec_k)


# In[12]:


k=3
tic = tm.perf_counter()
yPred = predict.getReco( arr_ll, k )
toc = tm.perf_counter()
print("time for prediction: ",toc-tic)
prec_k = utils.getPrecAtK( arr_ll, yPred, k )
print("prec@",k,": ",prec_k)
mprec_k = utils.getMPrecAtK( arr_ll, yPred, k )
print("mprec@",k,": ",mprec_k)


# In[13]:


k=5
tic = tm.perf_counter()
yPred = predict.getReco( arr_ll, k )
toc = tm.perf_counter()
print("time for prediction: ",toc-tic)
prec_k = utils.getPrecAtK( arr_ll, yPred, k )
print("prec@",k,": ",prec_k)
mprec_k = utils.getMPrecAtK( arr_ll, yPred, k )
print("mprec@",k,": ",mprec_k)


# In[ ]:




