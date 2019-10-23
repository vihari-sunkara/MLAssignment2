#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


subprocess.call(['./sample_run.sh'])


# In[2]:


# def kmeans(cent_mat,K):
#     start = tm.time()
#     kmeans = KMeans(n_clusters=100,n_jobs=-1)
#     kmeans.fit(cent_mat)
#     end = tm.time()
#     print("time elapsed: ",end-start)
#     list = kmeans.predict(cent_mat)
#     partitions = [[] for i in range(K)]
#     for i in range(3400):
#         partitions[list[i]].append(i)
#     return partitions
# # BELOW CODE FOR PRINTING THE LABELS IN EACH PARTITION
# #     i=0
# #     for partition in partitions:
# #         print("partition ",i," has labels ",partition)
# #         i+=1


# In[3]:


# def delete_rows_csr(mat, indices):
#     """
#     Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
#     """
#     if not isinstance(mat, scipy.sparse.csr_matrix):
#         raise ValueError("works only for CSR format -- use .tocsr() first")
#     indices = list(indices)
#     mask = numpy.ones(mat.shape[0], dtype=bool)
#     mask[indices] = False
#     return mat[mask]

# def shrinkdata(node,inputX,inputy,centMatrix,n_Y,n_X,n_Xf,n_cXf):
#     node.inputy=shrinkMatrix(inputy,n_Y,n_X)
#     node.inputX=shrinkMatrix(inputX,n_X,n_Xf)
#     node.centMatrix=shrinkMatrix(centMatrix,n_Y,n_cXf)
#     return

# def shrinkMatrix(dataInput,inputDimension,outputDimension):
#     rows=dataInput.shape[0]
#     indicesDelete=[]
#     for i in rows:
        
#     return


# In[ ]:


# def bonsai_train(maxDepth,K):
#     (X, y)=utils.load_data("data",16385,3400)
#     X=normalize(X, norm='l2', axis=1)
#     yt=y.transpose()
#     cent_mat=yt*X
#     cent_mat=normalize(cent_mat, norm='l2', axis=1)
#     root=dt.Node()
#     root.inputX=X
#     root.inputy=yt
#     root.depth=0
#     root.centMatrix=cent_mat
#     root.childNodes=grow(root,maxDepth,K)
#     return root

# def grow(node,maxDepth,K):
#     listOfPartition=kmeans(node.centMatrix,K)
#     childrenNodes=[]
#     for nodeLabelPartition in listOfPartition:
#         tempNode=dt.Node()
#         tempNode.labels=nodeLabelPartition
#         tempNode.depth=node.depth+1
#         n_X=[]
#         n_Xf=[]
#         n_cXf=[]
#         shrinkdata(tempNode,node.inputX,node.inputy,node.centMatrix,nodeLabelPartition,n_X,n_Xf,n_cXf)
#         if(K>=len(nodeLabelPartition) or tempNode.depth>=maxDepth):
#             #implement one-ve-all CSVM for the leaf node
#             continue
#         else:
#             grandChildNodes=grow(tempNode,maxDepth,K)
#             for grandchildnode in grandChildNodes:
#                 tempNode.children.append(grandchildnode)
#         childrenNodes.append(tempNode)
#     #implement one-ve-all CSVM for the non-leaf node
#     return childrenNodes

# bonsai_train(2,3)


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


# In[17]:


#VALIDATION STARTS HERE
(X, y)=utils.load_data("data",16385,3400)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[18]:


X_train.shape


# In[19]:


X_test.shape


# In[23]:


type(y_train)


# In[22]:


y_test.shape


# In[29]:


utils.dump_data(X_train,y_train,"valTrn")


# In[30]:


utils.dump_data(X_test,y_test,"valTest")


# In[ ]:




