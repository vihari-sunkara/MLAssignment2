import numpy as np
from numpy import random as rand
#NEWLY ADDED IMPORTS FOR getReco METHOD
import subprocess
from sklearn.datasets import load_svmlight_file
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# X: n x d matrix in csr_matrix format containing d-dim (sparse) features for n test data points
# k: the number of recommendations to return for each test data point in ranked order

# OUTPUT CONVENTION
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of labels with the i-th row 
# containing k labels which it thinks are most appropriate for the i-th test point. Labels must be returned in 
# ranked order i.e. the label yPred[i][0] must be considered most appropriate followed by yPred[i][1] and so on

# CAUTION: Make sure that you return (yPred below) an n x k numpy (nd) array and not a numpy/scipy/sparse matrix
# The returned matrix will always be a dense matrix and it terribly slows things down to store it in csr format
# The evaluation code may misbehave and give unexpected results if an nd-array is not returned
def getTopK(row_data,ind_data):
    ids = (row_data).argsort()[::-1]
    return row_data[ids], ind_data[ids]

def getReco( X, k ):
    # Find out how many data points we have
    subprocess.call(['./predict.sh'])
    # this has to be using the sir's file
    #not our test files from assignment data, change the script and the paths
    filename='score_mat.txt'
    L=3400 #hardcoded , get this avoided if possible 
    sparse_matrix, _ = load_svmlight_file( "%s" % filename, multilabel = True, n_features = L, offset = 1 )
    arr_ll=sparse_matrix

    n = arr_ll.shape[0]
    L = arr_ll.shape[1]
        
    print("n: ",n)
    print("L: ",L)
    
    yPred = np.zeros( (n, k) )
    #--MAX LEN IS 100 FOR EACH OF THE ROWS..RECHECK THIS ONCE AND FIND OUT
    for i in range(n):
        data,row=getTopK(np.array(arr_ll[i].data),np.array(arr_ll[i].indices))
        arr_ll[i].data = data
        arr_ll[i].indices = row
        #length = len(row)
        yPred[i] = row[0:k]
        
    
    #np.savetxt("yPred.txt",yPred.astype(int), fmt='%i')#FORMATTING TO INTEGER VALUED LABELS
    # Load and unpack the dummy model
    # The dummy model simply stores the labels in decreasing order of their popularity
#     npzModel = np.load( "model.npz" )
#     model = npzModel[npzModel.files[0]]
    # Let us predict a random subset of the 2k most popular labels no matter what the test point
#     shortList = model[0:2*k]
    # Make sure we are returning a numpy nd-array and not a numpy matrix or a scipy sparse matrix
#     yPred = np.zeros( (n, k) )
#     for i in range( n ):
#         yPred[i,:] = rand.permutation( shortList )[0:k]
    print("yPred of type ",type(yPred)," is returned here")
    return yPred