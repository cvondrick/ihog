import sys
import numpy as np
import scipy
import scipy.sparse as ssp

import spams
import time
from test_utils import *

def test_fistaFlat():
    param = {'numThreads' : 1,'verbose' : True,
             'lambda1' : 0.05, 'it0' : 10, 'max_it' : 200,
             'L0' : 0.1, 'tol' : 1e-3, 'intercept' : False,
             'pos' : False}
    np.random.seed(0)
    m = 100;n = 200
    X = np.asfortranarray(np.random.normal(size = (m,n)))
    X = np.asfortranarray(X - np.tile(np.mean(X,0),(X.shape[0],1)))
    X = spams.normalize(X)
    Y = np.asfortranarray(np.random.normal(size = (m,1)))
    Y = np.asfortranarray(Y - np.tile(np.mean(Y,0),(Y.shape[0],1)))
    Y = spams.normalize(Y)
    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="FORTRAN")
    # Regression experiments 
    # 100 regression problems with the same design matrix X.
    print '\nVarious regression experiments'
    param['compute_gram'] = True
    print '\nFISTA + Regression l1'
    param['loss'] = 'square'
    param['regul'] = 'l1'
    # param.regul='group-lasso-l2';
    # param.size_group=10;
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
##    print "XX %s" %str(optim_info.shape);return None
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:],0),np.mean(optim_info[3,:],0))
###
    print '\nISTA + Regression l1'
    param['ista'] = True
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f\n' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
##
    print '\nSubgradient Descent + Regression l1'
    param['ista'] = False
    param['subgrad'] = True
    param['a'] = 0.1
    param['b'] = 1000 # arbitrary parameters
    max_it = param['max_it']
    it0 = param['it0']
    param['max_it'] = 500
    param['it0'] = 50
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f\n' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
    param['subgrad'] = False
    param['max_it'] = max_it
    param['it0'] = it0

###
    print '\nFISTA + Regression l2'
    param['regul'] = 'l2'
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f\n' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
###
    print '\nFISTA + Regression l2 + sparse feature matrix'
    param['regul'] = 'l2';
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,ssp.csc_matrix(X),W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
###########

    print '\nFISTA + Regression Elastic-Net'
    param['regul'] = 'elastic-net'
    param['lambda2'] = 0.1
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[3,:]))

    print '\nFISTA + Group Lasso L2'
    param['regul'] = 'group-lasso-l2'
    param['size_group'] = 2
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:],0),np.mean(optim_info[3,:],0))
    
    print '\nFISTA + Group Lasso L2 with variable size of groups'
    param['regul'] = 'group-lasso-l2'
    param2=param.copy()
    param2['groups'] = np.array(np.random.random_integers(1,5,X.shape[1]),dtype = np.int32)
    param2['lambda1'] *= 10
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:],0),np.mean(optim_info[3,:],0))

    print '\nFISTA + Trace Norm'
    param['regul'] = 'trace-norm-vec'
    param['size_group'] = 5
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:],0),np.mean(optim_info[3,:]))
    
####    
   
    print '\nFISTA + Regression Fused-Lasso'
    param['regul'] = 'fused-lasso'
    param['lambda2'] = 0.1
    param['lambda3'] = 0.1; #
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[3,:]))
    
    print '\nFISTA + Regression no regularization'
    param['regul'] = 'none'
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[3,:]))
    
    
    print '\nFISTA + Regression l1 with intercept '
    param['intercept'] = True
    param['regul'] = 'l1'
    x1 = np.asfortranarray(np.concatenate((X,np.ones((X.shape[0],1))),1))
    W01 = np.asfortranarray(np.concatenate((W0,np.zeros((1,W0.shape[1]))),0))
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,x1,W01,True,**param)',locals()) # adds a column of ones to X for the intercept,True)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
    
    print '\nFISTA + Regression l1 with intercept+ non-negative '
    param['pos'] = True
    param['regul'] = 'l1'
    x1 = np.asfortranarray(np.concatenate((X,np.ones((X.shape[0],1))),1))
    W01 = np.asfortranarray(np.concatenate((W0,np.zeros((1,W0.shape[1]))),0))
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,x1,W01,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[3,:]))
    param['pos'] = False
    param['intercept'] = False

    print '\nISTA + Regression l0'
    param['regul'] = 'l0'
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[3,:]))
    
# Classification
    
    print '\nOne classification experiment'
#*    Y = 2 * double(randn(100,1) > 0)-1
    Y = np.asfortranarray(2 * np.asarray(np.random.normal(size = (100,1)) > 0,dtype='float64') - 1)
    print '\nFISTA + Logistic l1'
    param['regul'] = 'l1'
    param['loss'] = 'logistic'
    param['lambda1'] = 0.01
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
# can be used of course with other regularization functions, intercept,...
    param['regul'] = 'l1'
    param['loss'] = 'weighted-logistic'
    param['lambda1'] = 0.01
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
# can be used of course with other regularization functions, intercept,...
#!    pause
    
    print '\nFISTA + Logistic l1 + sparse matrix'
    param['loss'] = 'logistic'
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,ssp.csc_matrix(X),W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
# can be used of course with other regularization functions, intercept,...
    

# Multi-Class classification
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size = (100,1000))) - 1)
    param['loss'] = 'multi-logistic'
    print '\nFISTA + Multi-Class Logistic l1'
    nclasses = np.max(Y[:])+1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=np.float64,order="FORTRAN")
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())

    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
# can be used of course with other regularization functions, intercept,...
    
    
# Multi-Task regression
    Y = np.asfortranarray(np.random.normal(size = (100,100)))
    Y = np.asfortranarray(Y - np.tile(np.mean(Y,0),(Y.shape[0],1)))
    Y = spams.normalize(Y)
    param['compute_gram'] = False
    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="FORTRAN")
    param['loss'] = 'square'
    print '\nFISTA + Regression l1l2 '
    param['regul'] = 'l1l2'
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
    
    print '\nFISTA + Regression l1linf '
    param['regul'] = 'l1linf'
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
    
    
    print '\nFISTA + Regression l1l2 + l1 '
    param['regul'] = 'l1l2+l1'
    param['lambda2'] = 0.1
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[3,:]))
    
    
    print '\nFISTA + Regression l1linf + l1 '
    param['regul'] = 'l1linf+l1'
    param['lambda2'] = 0.1
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[3,:]))
    
    
    print '\nFISTA + Regression l1linf + row + columns '
    param['regul'] = 'l1linf-row-column'
    param['lambda2'] = 0.1
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
    
# Multi-Task Classification
    
    print '\nFISTA + Logistic + l1l2 '
    param['regul'] = 'l1l2'
    param['loss'] = 'logistic'
#*    Y = 2*double(randn(100,100) > 0)-1
    Y = np.asfortranarray(2 * np.asarray(np.random.normal(size = (100,100)) > 1,dtype='float64') - 1)
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
# Multi-Class + Multi-Task Regularization
    
    
    print '\nFISTA + Multi-Class Logistic l1l2 '
#*    Y = double(ceil(5*rand(100,1000))-1)
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size = (100,1000))) - 1)
    Y = spams.normalize(Y)
    param['loss'] = 'multi-logistic'
    param['regul'] = 'l1l2'
    nclasses = np.max(Y[:])+1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=np.float64,order="FORTRAN")
    (W, optim_info) = Xtest1('spams','spams.fistaFlat(Y,X,W0,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),np.mean(optim_info[3,:]))
# can be used of course with other regularization functions, intercept,...
    
    
#############
def test_fistaGraph():
    np.random.seed(0)
    num_threads = -1 # all cores (-1 by default)
    verbose = False   # verbosity, false by default
    lambda1 = 0.1 # regularization ter
    it0 = 1      # frequency for duality gap computations
    max_it = 100 # maximum number of iterations
    L0 = 0.1
    tol = 1e-5
    intercept = False
    pos = False

    eta_g = np.array([1, 1, 1, 1, 1],dtype=np.float64)

    groups = ssp.csc_matrix(np.array([[0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0]],dtype=np.bool),dtype=np.bool)

    groups_var = ssp.csc_matrix(np.array([[1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0],
                           [1, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0],
                           [0, 1, 0, 1, 0],
                           [0, 1, 0, 0, 1],
                           [0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1],
                           [0, 0, 1, 0, 0]],dtype=np.bool),dtype=np.bool)

    graph = {'eta_g': eta_g,'groups' : groups,'groups_var' : groups_var}

    verbose = True
    X = np.asfortranarray(np.random.normal(size = (100,10)))
    X = np.asfortranarray(X - np.tile(np.mean(X,0),(X.shape[0],1)))
    X = spams.normalize(X)
    Y = np.asfortranarray(np.random.normal(size = (100,1)))
    Y = np.asfortranarray(Y - np.tile(np.mean(Y,0),(Y.shape[0],1)))
    Y = spams.normalize(Y)
    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="FORTRAN")
    # Regression experiments 
    # 100 regression problems with the same design matrix X.
    print '\nVarious regression experiments'
    compute_gram = True
#
    print '\nFISTA + Regression graph'
    loss = 'square'
    regul = 'graph'
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
#
    print '\nADMM + Regression graph'
    admm = True
    lin_admm = True
    c = 1
    delta = 1
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
#
    admm = False
    max_it = 5
    it0 = 1
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
#
#  works also with non graph-structured regularization. graph is ignored
    print '\nFISTA + Regression Fused-Lasso'
    regul = 'fused-lasso'
    lambda2 = 0.01
    lambda3 = 0.01
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,
        lambda2 = lambda2,lambda3 = lambda3,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),t,np.mean(optim_info[3,:]))
#
    print '\nFISTA + Regression graph with intercept'
    regul = 'graph'
    intercept = True
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,
        lambda2 = lambda2,lambda3 = lambda3,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
    intercept = False

# Classification
    print '\nOne classification experiment'
    Y = np.asfortranarray( 2 * np.asfortranarray(np.random.normal(size = (100,Y.shape[1])) > 0,dtype = np.float64) -1)
    print '\nFISTA +  Logistic + graph-linf'
    loss = 'logistic'
    regul = 'graph'
    lambda1 = 0.01
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,
        lambda2 = lambda2,lambda3 = lambda3,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
#
# can be used of course with other regularization functions, intercept,...

# Multi-Class classification
    
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size = (100,Y.shape[1]))) - 1)
    loss = 'multi-logistic'
    regul = 'graph'
    print '\nFISTA + Multi-Class Logistic + graph'
    nclasses = np.max(Y) + 1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=np.float64,order="FORTRAN")
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,
        lambda2 = lambda2,lambda3 = lambda3,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
#
# can be used of course with other regularization functions, intercept,...
# Multi-Task regression
    Y = np.asfortranarray(np.random.normal(size = (100,Y.shape[1])))
    Y = np.asfortranarray(Y - np.tile(np.mean(Y,0),(Y.shape[0],1)))
    Y = spams.normalize(Y)
    W0 = W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="FORTRAN")
    compute_gram = False
    verbose = True
    loss = 'square'
    print '\nFISTA + Regression multi-task-graph'
    regul = 'multi-task-graph'
    lambda2 = 0.01
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,
        lambda2 = lambda2,lambda3 = lambda3,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
#
# Multi-Task Classification
    print '\nFISTA + Logistic + multi-task-graph'
    regul = 'multi-task-graph'
    lambda2 = 0.01
    loss = 'logistic'
    Y = np.asfortranarray( 2 * np.asfortranarray(np.random.normal(size = (100,Y.shape[1])) > 0,dtype = np.float64) -1)
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,
        lambda2 = lambda2,lambda3 = lambda3,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
# Multi-Class + Multi-Task Regularization
    verbose = False
    print '\nFISTA + Multi-Class Logistic +multi-task-graph'
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size = (100,Y.shape[1]))) - 1)
    loss = 'multi-logistic'
    regul = 'multi-task-graph'
    nclasses = np.max(Y) + 1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=np.float64,order="FORTRAN")
    tic = time.time()
    (W, optim_info) = spams.fistaGraph(
        Y,X,W0,graph,True,numThreads = num_threads,verbose = verbose,
        lambda1 = lambda1,it0 = it0,max_it = max_it,L0 = L0,tol = tol,
        intercept = intercept,pos = pos,compute_gram = compute_gram,
        loss = loss,regul = regul,admm = admm,lin_admm = lin_admm,c = c,
        lambda2 = lambda2,lambda3 = lambda3,delta = delta)
    tac = time.time()
    t = tac - tic
    print 'mean loss: %f, mean relative duality_gap: %f, time: %f, number of iterations: %f' %(np.mean(optim_info[0,:]),np.mean(optim_info[2,:]),t,np.mean(optim_info[3,:]))
# can be used of course with other regularization functions, intercept,...

    return None

def test_fistaTree():
    param = {'numThreads' : -1,'verbose' : False,
             'lambda1' : 0.001, 'it0' : 10, 'max_it' : 200,
             'L0' : 0.1, 'tol' : 1e-5, 'intercept' : False,
             'pos' : False}
    np.random.seed(0)
    m = 100;n = 10
    X = np.asfortranarray(np.random.normal(size = (m,n)))
    X = np.asfortranarray(X - np.tile(np.mean(X,0),(X.shape[0],1)))
    X = spams.normalize(X)
    Y = np.asfortranarray(np.random.normal(size = (m,m)))
    Y = np.asfortranarray(Y - np.tile(np.mean(Y,0),(Y.shape[0],1)))
    Y = spams.normalize(Y)
    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="FORTRAN")
    own_variables =  np.array([0,0,3,5,6,6,8,9],dtype=np.int32)
    N_own_variables =  np.array([0,3,2,1,0,2,1,1],dtype=np.int32)
    eta_g = np.array([1,1,1,2,2,2,2.5,2.5],dtype=np.float64)
    groups = np.asfortranarray([[0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0]],dtype = np.bool)
    groups = ssp.csc_matrix(groups,dtype=np.bool)
    tree = {'eta_g': eta_g,'groups' : groups,'own_variables' : own_variables,
            'N_own_variables' : N_own_variables}
    print '\nVarious regression experiments'
    param['compute_gram'] = True

    print '\nFISTA + Regression tree-l2'
    param['loss'] = 'square'
    param['regul'] = 'tree-l2'
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[3,:],0))
###
    print '\nFISTA + Regression tree-linf'
    param['regul'] = 'tree-linf'
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:]),np.mean(optim_info[3,:],0))
###
# works also with non tree-structured regularization. tree is ignored
    print '\nFISTA + Regression Fused-Lasso'
    param['regul'] = 'fused-lasso'
    param['lambda2'] = 0.001
    param['lambda3'] = 0.001
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[3,:],0))
###
    print '\nISTA + Regression tree-l0'
    param['regul'] = 'tree-l0'
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[3,:],0))
###
    print '\nFISTA + Regression tree-l2 with intercept'
    param['intercept'] = True
    param['regul'] = 'tree-l2'
    x1 = np.asfortranarray(np.concatenate((X,np.ones((X.shape[0],1))),1))
    W01 = np.asfortranarray(np.concatenate((W0,np.zeros((1,W0.shape[1]))),0))
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,x1,W01,tree,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[3,:],0))
###
    param['intercept'] = False

#    Classification

    print '\nOne classification experiment'
    Y = np.asfortranarray(2 * np.asarray(np.random.normal(size = (100,Y.shape[1])) > 0,dtype='float64') - 1)
    print '\nFISTA + Logistic + tree-linf'
    param['regul'] = 'tree-linf'
    param['loss'] = 'logistic'
    param['lambda1'] = 0.001
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:]),np.mean(optim_info[3,:],0))
###
# can be used of course with other regularization functions, intercept,...

#  Multi-Class classification
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size = (100,Y.shape[1]))) - 1)
    param['loss'] = 'multi-logistic'
    param['regul'] = 'tree-l2'
    print '\nFISTA + Multi-Class Logistic + tree-l2'
    nclasses = np.max(Y[:])+1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=np.float64,order="FORTRAN")
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[3,:],0))
# can be used of course with other regularization functions, intercept,...

# Multi-Task regression
    Y = np.asfortranarray(np.random.normal(size = (100,100)))
    Y = np.asfortranarray(Y - np.tile(np.mean(Y,0),(Y.shape[0],1)))
    Y = spams.normalize(Y)
    param['compute_gram'] = False
    param['verbose'] = True;   # verbosity, False by default
    W0 = np.zeros((X.shape[1],Y.shape[1]),dtype=np.float64,order="FORTRAN")
    param['loss'] = 'square'
    print '\nFISTA + Regression  multi-task-tree'
    param['regul'] = 'multi-task-tree'
    param['lambda2'] = 0.001
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:]),np.mean(optim_info[3,:],0))

# Multi-Task Classification
    print '\nFISTA + Logistic + multi-task-tree'
    param['regul'] = 'multi-task-tree'
    param['lambda2'] = 0.001
    param['loss'] = 'logistic'
    Y = np.asfortranarray(2 * np.asarray(np.random.normal(size = (100,Y.shape[1])) > 0,dtype='float64') - 1)
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:]),np.mean(optim_info[3,:],0))

#  Multi-Class + Multi-Task Regularization
    param['verbose'] = False
    print '\nFISTA + Multi-Class Logistic +multi-task-tree'
    Y = np.asfortranarray(np.ceil(5 * np.random.random(size = (100,Y.shape[1]))) - 1)
    param['loss'] = 'multi-logistic'
    param['regul'] = 'multi-task-tree'
    nclasses = np.max(Y[:])+1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=np.float64,order="FORTRAN")
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X,W0,tree,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:]),np.mean(optim_info[3,:],0))
# can be used of course with other regularization functions, intercept,...

    print '\nFISTA + Multi-Class Logistic +multi-task-tree + sparse matrix'
    nclasses = np.max(Y[:])+1
    W0 = np.zeros((X.shape[1],nclasses * Y.shape[1]),dtype=np.float64,order="FORTRAN")
    X2 = ssp.csc_matrix(X)
    (W, optim_info) = Xtest1('spams','spams.fistaTree(Y,X2,W0,tree,True,**param)',locals())
    print 'mean loss: %f, mean relative duality_gap: %f, number of iterations: %f' %(np.mean(optim_info[0,:],0),np.mean(optim_info[2,:]),np.mean(optim_info[3,:],0))

    return None

def test_proximalFlat():
    param = {'numThreads' : -1,'verbose' : True,
             'lambda1' : 0.1 }
    m = 100;n = 1000
    U = np.asfortranarray(np.random.normal(size = (m,n)))
    
    # test L0
    print "\nprox l0"
    param['regul'] = 'l0'
    param['pos'] = False       # false by default
    param['intercept'] = False # false by default
    alpha = Xtest1('spams','spams.proximalFlat(U,False,**param)',locals())

    # test L1
    print "\nprox l1, intercept, positivity constraint"
    param['regul'] = 'l1'
    param['pos'] = True       # can be used with all the other regularizations
    param['intercept'] = True # can be used with all the other regularizations
    alpha = Xtest1('spams','spams.proximalFlat(U,False,**param)',locals())

    # test L2
    print "\nprox squared-l2"
    param['regul'] = 'l2'
    param['pos'] = False
    param['intercept'] = False
    alpha = Xtest1('spams','spams.proximalFlat(U,False,**param)',locals())

# test elastic-net
    print "\nprox elastic-net"
    param['regul'] = 'elastic-net'
    param['lambda2'] = 0.1
    alpha = Xtest1('spams','spams.proximalFlat(U,**param)',locals())

# test fused-lasso
    print "\nprox fused lasso"
    param['regul'] = 'fused-lasso'
    param['lambda2'] = 0.1
    param['lambda3'] = 0.1
    alpha = Xtest1('spams','spams.proximalFlat(U,**param)',locals())

# test l1l2
    print "\nprox mixed norm l1/l2"
    param['regul'] = 'l1l2'
    alpha = Xtest1('spams','spams.proximalFlat(U,**param)',locals())

# test l1linf
    print "\nprox mixed norm l1/linf"
    param['regul'] = 'l1linf'
    alpha = Xtest1('spams','spams.proximalFlat(U,**param)',locals())

# test l1l2+l1
    print "\nprox mixed norm l1/l2 + l1"
    param['regul'] = 'l1l2+l1'
    param['lambda2'] = 0.1
    alpha = Xtest1('spams','spams.proximalFlat(U,**param)',locals())

# test l1linf+l1
    print "\nprox mixed norm l1/linf + l1"
    param['regul'] = 'l1linf+l1'
    param['lambda2'] = 0.1
    alpha = Xtest1('spams','spams.proximalFlat(U,**param)',locals())

# test l1linf-row-column
    print "\nprox mixed norm l1/linf on rows and columns"
    param['regul'] = 'l1linf-row-column'
    param['lambda2'] = 0.1
    alpha = Xtest1('spams','spams.proximalFlat(U,**param)',locals())

# test none
    print "\nprox no regularization"
    param['regul'] = 'none'
    alpha = Xtest1('spams','spams.proximalFlat(U,**param)',locals())


    return None

def test_proximalGraph():
    np.random.seed(0)
    lambda1 = 0.1 # regularization parameter
    num_threads = -1 # all cores (-1 by default)
    verbose = True   # verbosity, false by default
    pos = False       # can be used with all the other regularizations
    intercept = False # can be used with all the other regularizations     

    U = np.asfortranarray(np.random.normal(size = (10,100)))
    print 'First graph example'
# Example 1 of graph structure
# groups:
# g1= {0 1 2 3}
# g2= {3 4 5 6}
# g3= {6 7 8 9}
    eta_g = np.array([1, 1, 1],dtype=np.float64)
    groups = ssp.csc_matrix(np.zeros((3,3)),dtype = np.bool)
    groups_var = ssp.csc_matrix(
        np.array([[1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1]],dtype=np.bool),dtype=np.bool)
    graph = {'eta_g': eta_g,'groups' : groups,'groups_var' : groups_var}

    print '\ntest prox graph'
    regul='graph'
    alpha = Xtest1('spams','spams.proximalGraph(U,graph,False,lambda1 = lambda1,numThreads  = num_threads ,verbose = verbose,pos = pos,intercept = intercept,regul = regul)',locals())

# Example 2 of graph structure
# groups:
# g1= {0 1 2 3}
# g2= {3 4 5 6}
# g3= {6 7 8 9}
# g4= {0 1 2 3 4 5}
# g5= {6 7 8}
    eta_g = np.array([1, 1, 1, 1, 1],dtype=np.float64)
    groups = ssp.csc_matrix(
        np.array([[0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0]],dtype=np.bool),dtype=np.bool)

    groups_var = ssp.csc_matrix(
        np.array([[1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0],
                  [0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0]],dtype=np.bool),dtype=np.bool)
    graph = {'eta_g': eta_g,'groups' : groups,'groups_var' : groups_var}
    print '\ntest prox graph'
    alpha = Xtest1('spams','spams.proximalGraph(U,graph,False,lambda1 = lambda1,numThreads  = num_threads ,verbose = verbose,pos = pos,intercept = intercept,regul = regul)',locals())
#
    print '\ntest prox multi-task-graph'
    regul = 'multi-task-graph'
    lambda2 = 0.1
    alpha = Xtest1('spams','spams.proximalGraph(U,graph,False,lambda1 = lambda1,lambda2 = lambda2,numThreads  = num_threads ,verbose = verbose,pos = pos,intercept = intercept,regul = regul)',locals())
#
    print '\ntest no regularization'
    regul = 'none'
    alpha = Xtest1('spams','spams.proximalGraph(U,graph,False,lambda1 = lambda1,lambda2 = lambda2,numThreads  = num_threads ,verbose = verbose,pos = pos,intercept = intercept,regul = regul)',locals())
    
    return None

def test_proximalTree():
    param = {'numThreads' : -1,'verbose' : True,
             'pos' : False, 'intercept' : False, 'lambda1' : 0.1 }
    m = 10;n = 1000
    U = np.asfortranarray(np.random.normal(size = (m,n)))
    print 'First tree example'
    # Example 1 of tree structure
    # tree structured groups:
    # g1= {0 1 2 3 4 5 6 7 8 9}
    # g2= {2 3 4}
    # g3= {5 6 7 8 9}
    own_variables =  np.array([0,2,5],dtype=np.int32) # pointer to the first variable of each group
    N_own_variables =  np.array([2,3,5],dtype=np.int32) # number of "root" variables in each group
    # (variables that are in a group, but not in its descendants).
    # for instance root(g1)={0,1}, root(g2)={2 3 4}, root(g3)={5 6 7 8 9}
    eta_g = np.array([1,1,1],dtype=np.float64) # weights for each group, they should be non-zero to use fenchel duality
    groups = np.asfortranarray([[0,0,0],
                                [1,0,0],
                                [1,0,0]],dtype = np.bool) 
    # first group should always be the root of the tree
    # non-zero entriees mean inclusion relation ship, here g2 is a children of g1,
    # g3 is a children of g1
    groups = ssp.csc_matrix(groups,dtype=np.bool)
    tree = {'eta_g': eta_g,'groups' : groups,'own_variables' : own_variables,
            'N_own_variables' : N_own_variables}
    print '\ntest prox tree-l0'
    param['regul'] = 'tree-l2'
    alpha = Xtest1('spams','spams.proximalTree(U,tree,False,**param)',locals())

    print '\ntest prox tree-linf'
    param['regul'] = 'tree-linf'
    alpha = Xtest1('spams','spams.proximalTree(U,tree,False,**param)',locals())

    print 'Second tree example'
# Example 2 of tree structure
# tree structured groups:
# g1= {0 1 2 3 4 5 6 7 8 9}    root(g1) = { };
# g2= {0 1 2 3 4 5}            root(g2) = {0 1 2};
# g3= {3 4}                    root(g3) = {3 4};
# g4= {5}                      root(g4) = {5};
# g5= {6 7 8 9}                root(g5) = { };
# g6= {6 7}                    root(g6) = {6 7};
# g7= {8 9}                    root(g7) = {8};
# g8 = {9}                     root(g8) = {9};
    own_variables =  np.array([0, 0, 3, 5, 6, 6, 8, 9],dtype=np.int32)
    N_own_variables =  np.array([0,3,2,1,0,2,1,1],dtype=np.int32)
    eta_g = np.array([1,1,1,2,2,2,2.5,2.5],dtype=np.float64)
    groups = np.asfortranarray([[0,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,1,0]],dtype = np.bool)
    groups = ssp.csc_matrix(groups,dtype=np.bool)
    tree = {'eta_g': eta_g,'groups' : groups, 'own_variables' : own_variables,
            'N_own_variables' : N_own_variables}
    print '\ntest prox tree-l0'
    param['regul'] = 'tree-l0'
    alpha = Xtest1('spams','spams.proximalTree(U,tree,False,**param)',locals())
    
    print '\ntest prox tree-l2'
    param['regul'] = 'tree-l2'
    alpha = Xtest1('spams','spams.proximalTree(U,tree,False,**param)',locals())

    print '\ntest prox tree-linf'
    param['regul'] = 'tree-linf'
    alpha = Xtest1('spams','spams.proximalTree(U,tree,False,**param)',locals())

# mexProximalTree also works with non-tree-structured regularization functions
    print '\nprox l1, intercept, positivity constraint'
    param['regul'] = 'l1'
    param['pos'] = True       # can be used with all the other regularizations
    param['intercept'] = True # can be used with all the other regularizations     
    alpha = Xtest1('spams','spams.proximalTree(U,tree,False,**param)',locals())

    print '\nprox multi-task tree'
    param['pos'] = False
    param['intercept'] = False
    param['lambda2'] = param['lambda1']
    param['regul'] = 'multi-task-tree'
    alpha = Xtest1('spams','spams.proximalTree(U,tree,False,**param)',locals())
    return None



tests = [
    'fistaFlat' , test_fistaFlat,
    'fistaGraph' , test_fistaGraph,
    'fistaTree' , test_fistaTree,
    'proximalFlat' , test_proximalFlat,
    'proximalGraph' , test_proximalGraph,
    'proximalTree' , test_proximalTree,
    ]
