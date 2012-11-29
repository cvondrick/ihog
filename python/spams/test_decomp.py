import sys
import numpy as np
import scipy
import scipy.sparse as ssp

import spams
import time
from test_utils import *

def test_sparseProject():
    np.random.seed(0)
    X = np.asfortranarray(np.random.normal(size = (20000,100)))
    #* matlab : X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
    param = {'numThreads' : -1, # number of processors/cores to use (-1 => all cores)
             'pos' : False,
             'mode': 1, # projection on the l1 ball
             'thrs' : 2}
    print "\n  Projection on the l1 ball"
    tic = time.time()
    X1 = spams.sparseProject(X,**param)
    tac = time.time()
    t = tac - tic
    print "  Time : ", t
    if (t != 0):
        print "%f signals of size %d projected per second" %((X.shape[1] / t),X.shape[0])
    s = np.abs(X1).sum(axis=0)
    print "Checking constraint: %f, %f" %(min(s),max(s))

    print "\n  Projection on the Elastic-Net"
    param['mode'] = 2  # projection on the Elastic-Net
    param['lambda1'] = 0.15
    tic = time.time()
    X1 = spams.sparseProject(X,**param)
    tac = time.time()
    t = tac - tic
    print "  Time : ", t
    if (t != 0):
        print "%f signals of size %d projected per second" %((X.shape[1] / t),X.shape[0])
    constraints = (X1*X1).sum(axis=0) + param['lambda1'] * np.abs(X1).sum(axis=0)
    print 'Checking constraint: %f, %f (Projection is approximate : stops at a kink)' %(min(constraints),max(constraints))
    
    print "\n  Projection on the FLSA"
    param['mode'] = 6       # projection on the FLSA
    param['lambda1'] = 0.7
    param['lambda2'] = 0.7
    param['lambda3'] = 1.0
    X = np.asfortranarray(np.random.random(size = (2000,100)))
    #* matlab : X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
    tic = time.time()
    X1 = spams.sparseProject(X,**param)
    tac = time.time()
    t = tac - tic
    print "  Time : ", t
    if (t != 0):
        print "%f signals of size %d projected per second" %((X.shape[1] / t),X.shape[0])
    constraints = 0.5 * param['lambda3'] * (X1*X1).sum(axis=0) + param['lambda1'] * np.abs(X1).sum(axis=0) + \
    param['lambda2'] * np.abs(X1[2:,] - X1[1:-1,]).sum(axis=0)
    print 'Checking constraint: %f, %f (Projection is approximate : stops at a kink)' %(min(constraints),max(constraints))
    return None

def test_cd():
    np.random.seed(0)
    X = np.asfortranarray(np.random.normal(size = (64,100)))
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
    D = np.asfortranarray(np.random.normal(size = (64,100)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    # parameter of the optimization procedure are chosen
    lambda1 = 0.015
    mode = spams.PENALTY
    tic = time.time()
    alpha = spams.lasso(X,D,lambda1 = lambda1,mode = mode,numThreads = 4)
    tac = time.time()
    t = tac - tic
    xd = X - D * alpha
    E = np.mean(0.5 * (xd * xd).sum(axis=0) + lambda1 * np.abs(alpha).sum(axis=0))
    print "%f signals processed per second for LARS" %(X.shape[1] / t)
    print 'Objective function for LARS: %g' %E
    tol = 0.001
    itermax = 1000
    tic = time.time()
#    A0 = ssp.csc_matrix(np.empty((alpha.shape[0],alpha.shape[1])))
    A0 = ssp.csc_matrix((alpha.shape[0],alpha.shape[1]))
    alpha2 = spams.cd(X,D,A0,lambda1 = lambda1,mode = mode,tol = tol, itermax = itermax,numThreads = 4)
    tac = time.time()
    t = tac - tic
    print "%f signals processed per second for CD" %(X.shape[1] / t)
    xd = X - D * alpha2
    E = np.mean(0.5 * (xd * xd).sum(axis=0) + lambda1 * np.abs(alpha).sum(axis=0))
    print 'Objective function for CD: %g' %E
    print 'With Random Design, CD can be much faster than LARS'

    return None

def test_l1L2BCD():
    np.random.seed(0)
    X = np.asfortranarray(np.random.normal(size = (64,100)))
    D = np.asfortranarray(np.random.normal(size = (64,200)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    ind_groups = np.array(xrange(0,X.shape[1],10),dtype=np.int32) #indices of the first signals in each group
    # parameters of the optimization procedure are chosen
    itermax = 100
    tol = 1e-3
    mode = spams.PENALTY
    lambda1 = 0.15 # squared norm of the residual should be less than 0.1
    numThreads = -1 # number of processors/cores to use the default choice is -1
                    # and uses all the cores of the machine
    alpha0 = np.zeros((D.shape[1],X.shape[1]),dtype=np.float64,order="FORTRAN")
    tic = time.time()
    alpha = spams.l1L2BCD(X,D,alpha0,ind_groups,lambda1 = lambda1,mode = mode,itermax = itermax,tol = tol,numThreads = numThreads)
    tac = time.time()
    t = tac - tic
    print "%f signals processed per second" %(X.shape[1] / t)

    return None

def test_lasso():
    np.random.seed(0)
    print "test lasso"
##############################################
# Decomposition of a large number of signals
##############################################
# data generation
    X = np.asfortranarray(np.random.normal(size=(100,100000)))
    #* X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
    D = np.asfortranarray(np.random.normal(size=(100,200)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    # parameter of the optimization procedure are chosen
#param.L=20; # not more than 20 non-zeros coefficients (default: min(size(D,1),size(D,2)))
    param = {
        'lambda1' : 0.15, # not more than 20 non-zeros coefficients
        'numThreads' : -1, # number of processors/cores to use; the default choice is -1
        # and uses all the cores of the machine
        'mode' : spams.PENALTY}        # penalized formulation

    tic = time.time()
    alpha = spams.lasso(X,D = D,return_reg_path = False,**param)
    tac = time.time()
    t = tac - tic
    print "%f signals processed per second\n" %(float(X.shape[1]) / t)
########################################
# Regularization path of a single signal 
########################################
    X = np.asfortranarray(np.random.normal(size=(64,1)))
    D = np.asfortranarray(np.random.normal(size=(64,10)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    (alpha,path) = spams.lasso(X,D = D,return_reg_path = True,**param)
    return None

def test_lassoMask():
    np.random.seed(0)
    print "test lassoMask"
##############################################
# Decomposition of a large number of signals
##############################################
# data generation
    X = np.asfortranarray(np.random.normal(size=(300,300)))
    # X=X./repmat(sqrt(sum(X.^2)),[size(X,1) 1]);
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
    D = np.asfortranarray(np.random.normal(size=(300,50)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    mask = np.asfortranarray((X > 0))  # generating a binary mask
    param = {
        'lambda1' : 0.15, # not more than 20 non-zeros coefficients
        'numThreads' : -1, # number of processors/cores to use; the default choice is -1
        # and uses all the cores of the machine
        'mode' : spams.PENALTY}        # penalized formulation
    tic = time.time()
    alpha = spams.lassoMask(X,D,mask,**param)
    tac = time.time()
    t = tac - tic
    print "%f signals processed per second\n" %(float(X.shape[1]) / t)
   
    return None

def test_lassoWeighted():
    np.random.seed(0)
    print "test lasso weighted"
##############################################
# Decomposition of a large number of signals
##############################################
# data generation
    X = np.asfortranarray(np.random.normal(size=(64,10000)))
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
    D = np.asfortranarray(np.random.normal(size=(64,256)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    param = { 'L' : 20,
        'lambda1' : 0.15, 'numThreads' : 8, 'mode' : spams.PENALTY} 
    W = np.asfortranarray(np.random.random(size = (D.shape[1],X.shape[1])))
    tic = time.time()
    alpha = spams.lassoWeighted(X,D,W,**param)
    tac = time.time()
    t = tac - tic
    print "%f signals processed per second\n" %(float(X.shape[1]) / t)
    
    return None

def test_omp():
    np.random.seed(0)
    print 'test omp'
    X = np.asfortranarray(np.random.normal(size=(64,100000)))
    D = np.asfortranarray(np.random.normal(size=(64,200)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    L = 10
    eps = 0.1
    numThreads = -1
    tic = time.time()
    alpha = spams.omp(X,D,L=L,eps= eps,return_reg_path = False,numThreads = numThreads)
    tac = time.time()
    t = tac - tic
    print "%f signals processed per second\n" %(float(X.shape[1]) / t)
########################################
# Regularization path of a single signal 
########################################
    X = np.asfortranarray(np.random.normal(size=(64,1)))
    D = np.asfortranarray(np.random.normal(size=(64,10)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    L = 5
    (alpha,path) = spams.omp(X,D,L=L,eps= eps,return_reg_path = True,numThreads = numThreads)
    return None

def test_ompMask():
    np.random.seed(0)
    print 'test ompMask'

########################################    
# Decomposition of a large number of signals
########################################    
    X = np.asfortranarray(np.random.normal(size=(300,300)))
    X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)))
    D = np.asfortranarray(np.random.normal(size=(300,50)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    mask = np.asfortranarray((X > 0))  # generating a binary mask
    L = 20
    eps = 0.1
    numThreads=-1
    tic = time.time()
    alpha = spams.ompMask(X,D,mask,L = L,eps = eps,return_reg_path = False,numThreads = numThreads)
    tac = time.time()
    t = tac - tic
    print "%f signals processed per second\n" %(float(X.shape[1]) / t)
    
    return None

def test_somp():
    np.random.seed(0)
    X = np.asfortranarray(np.random.normal(size = (64,10000)))
    D = np.asfortranarray(np.random.normal(size = (64,200)))
    D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)))
    ind_groups = np.array(xrange(0,10000,10),dtype=np.int32)
    tic = time.time()
    alpha = spams.somp(X,D,ind_groups,L = 10,eps = 0.1,numThreads=-1)
    tac = time.time()
    t = tac - tic
    print "%f signals processed per second" %(X.shape[1] / t)
    return None


tests = [
    'sparseProject' , test_sparseProject,
    'cd' , test_cd,
    'l1L2BCD' , test_l1L2BCD,
    'lasso' , test_lasso,
    'lassoMask' , test_lassoMask,
    'lassoWeighted' , test_lassoWeighted,
    'omp' , test_omp,
    'ompMask' , test_ompMask,
    'somp' , test_somp
    ]
