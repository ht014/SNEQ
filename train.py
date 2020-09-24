import scipy.sparse as sp
from model import Graph2Gauss
from utils import load_dataset, score_link_prediction, score_node_classification
import os
import sys
import scipy
from utils import *
import scipy.io
import numpy as np
from numpy.lib.stride_tricks import as_strided
# from netmf import netmf_large
import copy
import itertools
import npdistance as nd

print ('dataset name | training label ratio | input type | number of hops | GPU id | batch size')

name = sys.argv[1]  
 

if input_X =='attribute':
    g2g = Graph2Gauss(input=input_X,A=A, z= zz,data_ = name,X=X,L=128,batch = int(sys.argv[6]),
                      p_semi = p_lab, K=int(sys.argv[4]), p_val=0.05, p_test=0.1, p_nodes=0.0)
if input_X =='adjacent':
    g2g = Graph2Gauss(input=input_X,A=A, z= zz, data_=name,batch = int(sys.argv[6]),
                      X=A + F * sp.eye(A.shape[0]), p_semi = p_lab,L=128,  K=int(sys.argv[4]), p_val=0.05,
                      p_test=0.1, p_nodes=0.0)
if isinstance(z,np.ndarray) == False:
    z= z.toarray()
sess = g2g.train(z,sys.argv[5])
